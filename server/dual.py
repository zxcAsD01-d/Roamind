"""Two resident ARDY instances alternate overlapping horizons and immutable commits."""
from __future__ import annotations

import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager

import numpy as np

from ardy_keyframes import prompt_boundary
from overlap import MotionWindow, fuse_windows
from motion_quality import decode_window, match_boundary, overlap_condition, restore_contacts


class DualRunner:
    def __init__(self, runner_factory):
        from dual_worker import ModelProcess
        self.primary = runner_factory()
        # Fake runners used by the deterministic scheduler tests have no model.
        self.secondary = ModelProcess() if hasattr(self.primary, "model") else None
        self.runners = (self.primary, self.secondary)
        self.overlap = self.primary.horizon // 2
        if self.overlap < 4 or self.primary.horizon % 2:
            raise ValueError("双模型需要偶数生成窗口，至少八帧")
        self.locks = (threading.Lock(), threading.Lock())
        self.stream_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ardy-lane")
        self.last_pair = []
        self.next_lane = 0

    def info(self):
        return {"mode": "dual-overlap", "instances": 2, "steps": self.primary.num_denoising_steps,
                "scheduling": "alternating", "nextLane": self.next_lane,
                'gaitLock': True, 'contactCorrection': 'ardy-native',
                "horizonFrames": self.primary.horizon, "overlapFrames": self.overlap,
                "device": self.primary.device, "lastPair": self.last_pair,
                "workerPids": [os.getpid(), self.secondary.process.pid if self.secondary else None]}

    def warmup(self):
        tasks = [self.executor.submit(runner.warmup) for runner in self.runners]
        for task in tasks:
            task.result()

    @contextmanager
    def session(self, cancelled):
        from realtime import GenerationCancelled
        while not self.stream_lock.acquire(timeout=.05):
            if cancelled():
                raise GenerationCancelled()
        try:
            if cancelled():
                raise GenerationCancelled()
            yield
        finally:
            self.stream_lock.release()

    def _infer(self, lane, local, root, prompt, start, origin, plan, cancelled, pending=None):
        from realtime import GenerationCancelled
        if lane == 1:
            with self.locks[lane]:
                if cancelled():
                    raise GenerationCancelled()
                result = self.secondary.infer(local, root, prompt, start, origin, plan, pending)
                if cancelled():
                    raise GenerationCancelled()
                return result
        runner = self.primary
        with self.locks[lane], runner.torch.inference_mode():
            if cancelled():
                raise GenerationCancelled()
            started = time.perf_counter()
            history = runner._encode_history(local, root) if len(root) else None
            if plan is None:
                window = self._free_window(runner, history, prompt, local, root, pending)
            else:
                window = self._plan_window(runner, history, start, origin, plan, cancelled, pending)
            if cancelled():
                raise GenerationCancelled()
            window.stats.update({"lane": lane, "startFrame": start,
                                 "pid": os.getpid(), "startedMs": started*1000,
                                 "finishedMs": time.perf_counter()*1000})
            return window

    @staticmethod
    def _free_window(runner, history, prompt, local, root, pending=None):
        features, mask = runner._prompt_features(prompt)
        observed, motion_mask = overlap_condition(runner, pending, 0 if history is None else history.shape[1])
        motion = runner.model.autoregressive_step(
            num_frames=(0 if history is None else history.shape[1])+runner.horizon,
            num_denoising_steps=runner.num_denoising_steps, motion_mask=motion_mask, observed_motion=observed,
            cfg_weight=(2., 2.), text_feat=features, text_pad_mask=mask, init_history_sequence=history)
        return decode_window(runner, motion[:, -runner.horizon:])

    @staticmethod
    def _plan_window(runner, history, start, origin, plan, cancelled, pending=None):
        from constrained import conditioned_window, next_boundary
        fused = plan.get('world') == 'fused-pointcloud'
        end = prompt_boundary(plan, start) if fused else min(next_boundary(plan, start), plan['frames'])
        indices = np.clip(np.arange(start, start+runner.horizon), 0, end-1)
        track = np.asarray(plan["rootTrack"], dtype=np.float32)[indices]
        heading = np.asarray(plan["headingTrack"], dtype=np.float32)[indices]
        prompt = plan["promptTrack"][start]
        static = plan.get('staticPoses', {})
        return conditioned_window(runner, history, prompt, track, heading,
            cancelled, (origin, None, None), threading.Lock(), pending=pending,
            return_window=True, height_mask=np.asarray(plan.get('heightTrack',
                [False]*plan['frames']), dtype=bool)[indices],
            hand_targets=np.asarray(plan['handTrack'], dtype=np.float32)[indices] if 'handTrack' in plan else None,
            hand_joint=plan.get('handJoint', 10), fused_world=fused,
            static_poses={i: static[frame] for i, frame in enumerate(indices) if frame in static and start+i < end})

    def _wait_window(self, task, cancelled):
        from realtime import GenerationCancelled
        try:
            while True:
                if cancelled():
                    raise GenerationCancelled()
                try:
                    return task.result(timeout=.05)
                except TimeoutError:
                    if task.done():
                        raise
        except Exception as exc:
            # Drain an in-flight request before another stream can use its lane.
            task.cancel()
            if not task.cancelled():
                try:
                    task.result()
                except Exception as worker_error:
                    if worker_error is not exc and not isinstance(worker_error, GenerationCancelled):
                        exc.add_note(f"交替生成分支失败：{worker_error!r}")
            raise

    def _next_window(self, local, root, prompt, frame, origin, plan, cancelled, pending=None):
        lane = self.next_lane
        self.next_lane = 1-lane
        limit = self.primary.history_limit
        task = self.executor.submit(self._infer, lane, local[-limit:], root[-limit:], prompt,
                                    frame, origin, plan, cancelled, pending)
        window = self._wait_window(task, cancelled)
        # Keep the existing diagnostic field, now the two most recent turns.
        self.last_pair = (self.last_pair+[window.stats])[-2:]
        return window

    def _advance_window(self, local, root, pending, prompt, frame, total, origin, plan, cancelled):
        boundary = total
        if plan:
            from constrained import next_boundary
            boundary = next_boundary(plan, frame)
        count = min(self.overlap, total-frame, boundary-frame)
        if pending is not None and total-frame <= self.overlap:
            return pending.slice(0, count), None
        window = self._next_window(local, root, prompt, frame, origin, plan, cancelled, pending)
        head = window.slice(0, self.overlap)
        if pending is not None:
            head = fuse_windows(pending, head)
        retained = None if frame+count == boundary else window.slice(count)
        return head.slice(0, count), retained

    def world_window(self, plan, start, history, world, cancelled):
        """Generate one provisional window, using only acknowledged playback as history."""
        from ardy_world_motion import finish_window
        runner = self.primary
        with self.session(cancelled), runner.torch.inference_mode():
            _, origin, poses, deltas = runner._read_history(history)
            origin = np.asarray([0, .92, 0], dtype=np.float32) if origin is None else origin
            local = poses[0] if poses else np.empty((0, 81), dtype=np.float32)
            root = deltas[0]+origin if deltas else np.empty((0, 3), dtype=np.float32)
            prompt = plan['promptTrack'][start]
            count = min(self.overlap, prompt_boundary(plan, start)-start)
            began = time.perf_counter()
            window = self._next_window(local, root, prompt, start, origin, plan, cancelled)
            output = finish_window(runner, window, local, root, plan, start, count, origin, world)
            event = self._event(f"{plan['id']}_{start}", prompt, 0, start, output, origin, plan, began)
            return {**event, 'version': 1, 'duration': count/runner.fps, 'dataset': 'ARDY · 融合点云'}

    def generate(self, prompt, duration, emit, initial_history=None, cancelled=lambda: False, plan=None):
        with self.session(cancelled), self.primary.torch.inference_mode():
            return self._generate(prompt, duration, emit, initial_history, cancelled, plan)

    def _generate(self, prompt, duration, emit, initial_history, cancelled, plan):
        from realtime import GenerationCancelled
        runner = self.primary
        _, origin, local_chunks, root_chunks = runner._read_history(initial_history)
        total = plan["frames"] if plan else round(duration*runner.fps)
        origin = origin if origin is not None else np.array([0, .92, 0], dtype=np.float32)
        prefix = sum(len(chunk) for chunk in local_chunks)
        local = np.concatenate(local_chunks) if local_chunks else np.empty((0, 81), dtype=np.float32)
        root = np.concatenate(root_chunks)+origin if root_chunks else np.empty((0, 3), dtype=np.float32)
        if plan and prefix and np.linalg.norm(root[-1]-origin+[0, .92, 0]-plan["start"]) > .25:
            raise ValueError("规划起点与播放衔接点不一致")
        name = f"ardy_{'plan' if plan else 'live'}_{time.time_ns()}"
        frame = index = 0
        pending = None
        contacts = []
        while frame < total:
            if cancelled():
                raise GenerationCancelled()
            if plan and 'touch' in plan:
                from touch import validate_live
                validate_live(plan)
            started = time.perf_counter()
            output, pending = self._advance_window(local, root, pending, prompt, frame, total,
                                                   origin, plan, cancelled)
            if frame == 0 and not prefix and plan is None:
                origin = output.root[0].copy()
            output = output.slice(0, min(len(output.root), total-frame))
            future = pending if total-frame > len(output.root) else None
            output = self._prepare_commit(output, local, root, future, origin, plan)
            if plan and 'touch' in plan:
                from touch_ik import refine
                from constrained import PARENTS
                skeleton = runner.model.motion_rep.skeleton
                output = refine(output, origin, plan, frame, local, root,
                                skeleton.neutral_joints.cpu().numpy(), PARENTS)
                self._validate_output(output, origin, plan)
                self._validate_touch(output, origin, plan, frame)
            emit(self._event(name, prompt, index, prefix+frame, output, origin, plan, started))
            local_chunks.append(output.local)
            root_chunks.append(output.root-origin)
            contacts.append(output.contacts)
            local = np.concatenate((local, output.local))[-runner.history_limit:]
            root = np.concatenate((root, output.root))[-runner.history_limit:]
            frame, index = frame+len(output.root), index+1
        result = runner._motion_document(name, prompt, prefix+total, local_chunks, root_chunks, origin)
        result["engine"] = self.info()
        result['restJoints'] = runner.model.motion_rep.skeleton.neutral_joints.cpu().numpy().ravel().round(6).tolist()
        result['footContacts'] = np.concatenate(contacts).astype(int).ravel().tolist()
        if prefix:
            result['footContacts'] = [0]*(prefix*4)+result['footContacts']
        if plan:
            result.update({"planId": plan["id"], "timeline": plan["timeline"], "world": plan["world"]})
            if 'touch' not in plan:
                from action_library import remember_generated
                result['library'] = remember_generated(plan, result, prefix, initial_history)
        return result

    def _validate_touch(self, window, origin, plan, frame):
        from touch import validate_live
        from touch_motion import validate_hand
        validate_live(plan)
        encoded = self.primary._encode_history(window.local, window.root)
        decoded = self.primary.model.motion_rep.inverse(encoded, is_normalized=True)
        shift = origin-np.array([0, .92, 0], dtype=np.float32)
        joints = decoded['posed_joints'][0, -len(window.root):].detach().cpu().numpy()-shift
        base = plan['effector']['chain'][0]
        window.stats.update(validate_hand(joints[:, plan['handJoint']], plan, frame, joints[:, base]))

    def _prepare_commit(self, output, local, root, future, origin, plan):
        output = match_boundary(output, local, root, preserve_root=plan is not None)
        output = restore_contacts(self.primary, output, local, root, future, preserve_root=plan is not None)
        self._validate_output(output, origin, plan)
        return output

    def _event(self, name, prompt, index, start, output, origin, plan, started):
        event = self.primary._chunk_event(name, prompt, index, 0, output.local, output.root-origin,
                                         start-index*self.primary.horizon, origin)
        event.update({"generationMs": round((time.perf_counter()-started)*1000, 2),
                      "engine": self.info(), "overlapFrames": self.overlap,
                      "restJoints": self.primary.model.motion_rep.skeleton.neutral_joints.cpu().numpy().ravel().round(6).tolist(),
                      "footContacts": output.contacts.astype(int).ravel().tolist()})
        if plan:
            event.update({"planId": plan["id"], "constraintStats": {**output.stats, "bodyContacts": 0}})
        return event

    def _validate_output(self, window, origin, plan):
        if not all(np.isfinite(values).all() for values in (window.local, window.root)):
            raise ValueError("生成动作包含非有限值，片段未发布")
        if plan is None:
            return
        from collision import body_collisions
        from constrained import PARENTS
        encoded = self.primary._encode_history(window.local, window.root)
        decoded = self.primary.model.motion_rep.inverse(encoded, is_normalized=True)
        shift = origin-np.array([0, .92, 0], dtype=np.float32)
        joints = decoded["posed_joints"][0].detach().cpu().numpy()-shift
        if body_collisions(joints, PARENTS):
            raise ValueError("重叠融合后的身体碰撞复检未通过，片段未发布")
