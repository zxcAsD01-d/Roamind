"""Run timed scene plans with genuine ARDY conditioning and reject unsafe chunks."""
from __future__ import annotations

import time

import numpy as np

from collision import JointClearanceCondition, RootHeightCondition, body_collisions
from motion_quality import decode_window, overlap_condition

PARENTS = [-1,0,1,2,3,4,5,4,7,8,9,10,10,4,13,14,15,16,16,0,19,20,21,0,23,24,25]


def prepare_conditions(runner, track, heading, history_count, corrections, shift, height_mask=None, hand_targets=None, hand_joint=10):
    from ardy.constraints import Root2DConstraintSet
    torch = runner.torch
    frames = torch.arange(history_count, history_count+runner.horizon)
    positions = torch.tensor(track+shift, dtype=torch.float32, device=runner.device)
    angles = torch.tensor(heading, dtype=torch.float32, device=runner.device)
    skeleton = runner.model.motion_rep.skeleton
    constraints = [Root2DConstraintSet(skeleton, frames, positions[:, [0, 2]], angles)]
    if height_mask is not None and np.any(height_mask):
        selected = torch.as_tensor(height_mask, dtype=torch.bool)
        constraints.append(RootHeightCondition(frames[selected], positions[selected.to(runner.device)]))
    if corrections:
        constraints.append(JointClearanceCondition(torch, corrections, history_count, shift, runner.device))
    if hand_targets is not None:
        if hand_joint not in (6, 10, 16, 21, 25):
            raise ValueError('接触约束只支持 Core27 头、左右手和左右脚')
        targets = {(frame, hand_joint): target for frame, target in enumerate(hand_targets)}
        constraints.append(JointClearanceCondition(torch, targets, history_count, shift, runner.device))
    return runner.model.motion_rep.create_conditions_from_constraints_batched(
        constraints, torch.tensor([history_count+runner.horizon], device=runner.device),
        to_normalize=True, device=runner.device)


def conditioned_window(runner, history, prompt, track, heading, cancelled, previous, inference_lock,
                       pending=None, return_window=False, height_mask=None, hand_targets=None, hand_joint=10):
    from realtime import GenerationCancelled
    origin, previous_local, previous_root = previous
    shift = origin - np.array([0, .92, 0], dtype=np.float32)
    count = 0 if history is None else history.shape[1]
    feature, mask = runner._prompt_features(prompt)
    corrections = {}
    largest_root_error = 0.
    for attempt in range(3):
        if cancelled():
            raise GenerationCancelled()
        with inference_lock:
            if cancelled():
                raise GenerationCancelled()
            observed, motion_mask = prepare_conditions(runner, track, heading, count, corrections, shift, height_mask, hand_targets, hand_joint)
            overlap, overlap_mask = overlap_condition(runner, pending, count)
            if overlap is not None:
                # Safety refinement may release a colliding retained keyframe;
                # locking its rotation while requesting another joint position is inconsistent.
                for frame in {frame for frame, _ in corrections}:
                    overlap_mask[:, count+frame] = 0
                if hand_targets is not None:
                    # A new visual target must not be hidden by retained full-body keys.
                    overlap_mask[:, count:] = 0
                observed = runner.torch.where(overlap_mask.bool(), overlap, observed)
                motion_mask = runner.torch.maximum(motion_mask, overlap_mask)
            motion = runner.model.autoregressive_step(
                num_frames=count+runner.horizon, num_denoising_steps=runner.num_denoising_steps,
                motion_mask=motion_mask, observed_motion=observed, cfg_weight=(2., 2.),
                text_feat=feature, text_pad_mask=mask, init_history_sequence=history,
                init_global_translation=runner.torch.tensor(
                    [[(track[0]+shift)[0], 0., (track[0]+shift)[2]]],
                    dtype=runner.torch.float32, device=runner.device),
                init_first_heading_angle=runner.torch.tensor([heading[0]], dtype=runner.torch.float32, device=runner.device))
        window = decode_window(runner, motion[:, -runner.horizon:])
        local, root = window.local, window.root
        largest_root_error = max(largest_root_error,
            float(np.max(np.linalg.norm((root-track-shift)[:, [0, 2]], axis=1))))
        # Dense root conditioning is followed by exact projection into the safe corridor.
        root[:, [0, 2]] = (track+shift)[:, [0, 2]]
        if height_mask is not None:
            root[height_mask, 1] = (track+shift)[height_mask, 1]
        if previous_local is not None:
            runner._blend_boundary(local, root, previous_local, previous_root)
            root[:, [0, 2]] = (track+shift)[:, [0, 2]]
        encoded = runner._encode_history(local, root)
        decoded = runner.model.motion_rep.inverse(encoded, is_normalized=True)
        joints = decoded["posed_joints"][0].detach().cpu().numpy() - shift
        collisions = body_collisions(joints, PARENTS)
        if not collisions:
            window.stats = {"refinementPasses": attempt, "bodyContacts": 0,
                            "rootProjectionMax": round(largest_root_error, 4)}
            return window if return_window else (local, root, window.stats)
        corrections.update(collisions)
    raise ValueError(f"身体碰撞复检未通过（{len(collisions)} 个骨段/帧）；已拒绝播放，请增加净空或改变路线/动作")


def next_boundary(plan, frame):
    now = frame/plan["fps"]
    boundaries = [round(t*plan["fps"]) for step in plan["timeline"] for t in (step["start"], step["end"]) if t > now+1e-6]
    return min(boundaries, default=plan["frames"])


def generate_plan(runner, plan, initial_history, emit, cancelled, inference_lock):
    started = time.perf_counter()
    history, origin, local_chunks, root_chunks = runner._read_history(initial_history)
    if origin is None:
        origin = np.array([0, .92, 0], dtype=np.float32)
    prefix = sum(len(chunk) for chunk in local_chunks)
    if prefix:
        last = root_chunks[-1][-1] + [0, .92, 0]
        if np.linalg.norm(last-np.asarray(plan["start"])) > .25:
            raise ValueError("规划起点与实际播放衔接点不一致；请从当前状态重新规划")
    name, frame, index = f"ardy_plan_{time.time_ns()}", 0, 0
    with runner.torch.inference_mode():
        while frame < plan["frames"]:
            count = min(runner.horizon, next_boundary(plan, frame)-frame, plan["frames"]-frame)
            track = np.asarray(plan["rootTrack"][frame:frame+count], dtype=np.float32)
            angle = np.asarray(plan["headingTrack"][frame:frame+count], dtype=np.float32)
            track = np.pad(track, ((0, runner.horizon-count), (0,0)), mode="edge")
            angle = np.pad(angle, (0, runner.horizon-count), mode="edge")
            previous_local = local_chunks[-1] if local_chunks else None
            previous_root = root_chunks[-1]+origin if root_chunks else None
            window_start = time.perf_counter()
            local, root, stats = conditioned_window(runner, history, plan["promptTrack"][frame], track, angle,
                cancelled, (origin, previous_local, previous_root), inference_lock)
            local, delta = local[:count], (root-origin)[:count]
            event = runner._chunk_event(name, plan["promptTrack"][frame], index, 0, local, delta, prefix+frame-index*runner.horizon, origin)
            event.update({"generationMs": round((time.perf_counter()-window_start)*1000, 2),
                          "constraintStats": stats, "planId": plan["id"]})
            emit(event)
            local_chunks.append(local)
            root_chunks.append(delta)
            history = runner._encode_history(np.concatenate(local_chunks[-5:]), np.concatenate(root_chunks[-5:])+origin)
            frame, index = frame+count, index+1
    result = runner._motion_document(name, "timed-house-plan", prefix+plan["frames"], local_chunks, root_chunks, origin)
    result.update({"planId": plan["id"], "timeline": plan["timeline"], "world": plan["world"],
                   "generationSeconds": round(time.perf_counter()-started, 3)})
    return result
