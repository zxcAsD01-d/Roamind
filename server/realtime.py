"""Resident ARDY inference for chunked, autoregressive motion generation."""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from settings import PROJECT as ROOT, MOTION_TOOLS, ARDY_ROOT, EMBEDDINGS
if str(ARDY_ROOT) not in sys.path:
    sys.path.insert(0, str(ARDY_ROOT))

import core


class GenerationCancelled(Exception):
    """A newer playback request superseded this generation."""


class RealtimeRunner:
    """Keep the motion model resident and emit each generated horizon as JSON data."""

    def __init__(self) -> None:
        if not (ARDY_ROOT/'ardy').is_dir():
            raise ValueError('ARDY_ROOT must point to an installed ARDY source checkout')
        if not EMBEDDINGS.is_file():
            raise ValueError('ARDY_EMBEDDINGS is missing; configure a trusted prompt cache')
        import torch
        from ardy.model import load_model

        torch.set_num_threads(1)
        self.torch = torch
        # The small six-step model is faster on this Apple CPU than MPS;
        # separate CPU workers also avoid serializing on a shared Metal queue.
        self.device = os.environ.get("ARDY_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device not in ("cpu", "mps", "cuda:0"):
            raise ValueError("ARDY_DEVICE 必须是 cpu、mps 或 cuda:0")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("本机不支持 MPS")
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise ValueError("本机不支持 CUDA")
        os.environ.setdefault("LOCAL_CACHE", "True")
        self.model = load_model("core", device=self.device, text_encoder=False).eval()
        cache = np.load(EMBEDDINGS, allow_pickle=True)
        self.embeddings = {
            str(prompt): embedding
            for prompt, embedding in zip(cache["prompts"], cache["emb"])
        }
        self.fps = int(self.model.motion_rep.fps)
        self.horizon = int(self.model.gen_horizon_len)
        training_window = (10 * self.fps // self.model.num_frames_per_token) * self.model.num_frames_per_token
        self.history_limit = max(
            0,
            ((training_window - self.horizon) // self.model.num_frames_per_token)
            * self.model.num_frames_per_token,
        )
        self.num_denoising_steps = 6

    def _prompt_features(self, prompt: str):
        embedding = self.embeddings.get(prompt)
        if embedding is None:
            raise ValueError(f"prompt {prompt!r} is not in the embedding cache")
        feat = self.torch.from_numpy(np.asarray(embedding, dtype=np.float32)).unsqueeze(0).unsqueeze(1)
        feat = feat.to(self.device)
        pad_mask = self.torch.ones((1, feat.shape[1]), dtype=self.torch.bool, device=self.device)
        return feat, pad_mask

    def warmup(self) -> None:
        torch = self.torch
        prompt = "A person walks forward."
        text_feat, text_pad_mask = self._prompt_features(prompt)
        with torch.no_grad():
            first = self.model.autoregressive_step(
                num_frames=self.horizon,
                num_denoising_steps=self.num_denoising_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(2.0, 2.0),
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
            )
            history = first[:, -self.history_limit :].detach()
            self.model.autoregressive_step(
                num_frames=history.shape[1] + self.horizon,
                num_denoising_steps=self.num_denoising_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(2.0, 2.0),
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                init_history_sequence=history,
            )

    def _decode_chunk(self, motion) -> tuple[np.ndarray, np.ndarray]:
        output = self.model.motion_rep.inverse(motion, is_normalized=True)
        local = output["local_rot_mats"][0].detach().cpu().numpy()
        root = output["root_positions"][0].detach().cpu().numpy()
        local_axis_angle = core.quat_to_axis_angle(core.rotmat_to_quat(local)).reshape(len(root), -1)
        return local_axis_angle, root

    def _encode_history(self, local: np.ndarray, root: np.ndarray):
        """Rebuild features from the exact poses used by the player."""
        count = min(len(root), self.history_limit)
        local, root = local[-count:], root[-count:]
        padding = (-count) % self.model.num_frames_per_token
        local = np.pad(local, ((padding, 0), (0, 0)), mode="edge")
        root = np.pad(root, ((padding, 0), (0, 0)), mode="edge")
        matrices = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix()
        matrices = matrices.reshape(1, len(root), core.N_ARDY, 3, 3)
        return self.model.motion_rep(
            self.torch.tensor(matrices, dtype=self.torch.float32, device=self.device),
            self.torch.tensor(root[None], dtype=self.torch.float32, device=self.device),
            to_normalize=True,
        ).detach()

    def _read_history(self, data: dict | None):
        if data is None:
            return None, None, [], []
        if not isinstance(data, dict):
            raise ValueError("history must be an object")
        frames = data.get("frames")
        if type(frames) is not int or not 1 <= frames <= self.history_limit:
            raise ValueError(f"history.frames must be between 1 and {self.history_limit}")
        if data.get("fps") != self.fps or data.get("up") != "yup":
            raise ValueError("history must use ARDY's 20 fps Y-up coordinates")
        if data.get("bones") != core._atf.VRM_BONES or data.get("parents") != core._atf.PARENTS:
            raise ValueError("history must use ARDY CoreSkeleton27 bone order")
        local = np.asarray(data.get("localRot"), dtype=np.float32).reshape(frames, core.N_ARDY * 3)
        delta = np.asarray(data.get("rootDelta"), dtype=np.float32).reshape(frames, 3)
        origin = np.asarray(data.get("rootOrigin", [0, 0.92, 0]), dtype=np.float32)
        if origin.shape != (3,) or not all(np.isfinite(a).all() for a in (local, delta, origin)):
            raise ValueError("history contains invalid coordinates")
        return self._encode_history(local, delta + origin), origin, [local], [delta]

    def _blend_boundary(self, local, root, previous_local, previous_root):
        """Decay pose/velocity residuals over 0.6 s without moving the old prefix."""
        if len(root) == 0 or len(previous_root) == 0 or len(previous_local) == 0:
            return
        count = min(12, len(root))
        last = Rotation.from_rotvec(previous_local[-1].reshape(-1, 3))
        before = Rotation.from_rotvec(previous_local[-2 if len(previous_local) > 1 else -1].reshape(-1, 3))
        angular_step = (before.inv() * last).as_rotvec()
        speed = np.linalg.norm(angular_step, axis=1, keepdims=True)
        # Keep a generated window from inheriting an implausible angular
        # impulse from a noisy final frame of the previous window.
        angular_step *= np.minimum(1, 0.15 / np.maximum(speed, 1e-8))
        predicted = last * Rotation.from_rotvec(angular_step)
        first = Rotation.from_rotvec(local[0].reshape(-1, 3))
        correction = (predicted * first.inv()).as_rotvec()
        velocity = previous_root[-1] - previous_root[-2 if len(previous_root) > 1 else -1]
        offset = previous_root[-1] + velocity - root[0]
        generated_velocity = root[1] - root[0] if len(root) > 1 else velocity
        for frame in range(count):
            t = frame / count
            weight = 1 - 3 * t * t + 2 * t * t * t
            rotation = Rotation.from_rotvec(correction * weight)
            local[frame] = (rotation * Rotation.from_rotvec(local[frame].reshape(-1, 3))).as_rotvec().reshape(-1)
            root[frame] += offset * weight + (velocity - generated_velocity) * frame * (1 - t) ** 2

    def _emitted_history(self, local_chunks, root_chunks, root_origin):
        """Encode the exact post-blend frames that the browser will play."""
        if not local_chunks or root_origin is None:
            return None
        local = np.concatenate(local_chunks[-5:], axis=0)
        root = np.concatenate(root_chunks[-5:], axis=0) + root_origin
        return self._encode_history(local, root)

    def _chunk_event(
        self,
        name: str,
        prompt: str,
        index: int,
        total_steps: int,
        local: np.ndarray,
        root_delta: np.ndarray,
        prefix_frames: int,
        root_origin: np.ndarray,
    ) -> dict:
        return {
            "type": "chunk",
            "id": name,
            "name": name,
            "fps": self.fps,
            "frames": len(local),
            "index": index,
            "startFrame": prefix_frames + index * self.horizon,
            "rootOrigin": root_origin.tolist(),
            "totalChunks": total_steps,
            "prompt": prompt,
            "bones": core._atf.VRM_BONES,
            "parents": core._atf.PARENTS,
            "up": "yup",
            "localRot": local.reshape(-1).round(6).tolist(),
            "rootDelta": root_delta.reshape(-1).round(6).tolist(),
        }

    def _motion_document(
        self,
        name: str,
        prompt: str,
        total_frames: int,
        local_chunks: list[np.ndarray],
        root_chunks: list[np.ndarray],
        root_origin: np.ndarray,
    ) -> dict:
        return {
            "version": 1,
            "id": name,
            "name": name,
            "source": "ardy-service-live",
            "dataset": f"service · {prompt}",
            "up": "yup",
            "fps": self.fps,
            "frames": total_frames,
            "duration": round(total_frames / self.fps, 3),
            "bones": core._atf.VRM_BONES,
            "parents": core._atf.PARENTS,
            "localRot": np.concatenate(local_chunks, axis=0).reshape(-1).round(6).tolist(),
            "rootDelta": np.concatenate(root_chunks, axis=0).reshape(-1).round(6).tolist(),
            "prompt": prompt,
            "rootOrigin": root_origin.tolist(),
        }

    def generate(
        self,
        prompt: str,
        duration: float,
        emit: Callable[[dict], None],
        initial_history: dict | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        inference_lock=None,
    ) -> dict:
        torch = self.torch
        if prompt not in self.embeddings:
            raise ValueError(f"prompt {prompt!r} is not in the embedding cache")

        text_feat, text_pad_mask = self._prompt_features(prompt)
        total_frames = int(duration * self.fps)
        total_steps = (total_frames + self.horizon - 1) // self.horizon
        history, root_origin, local_chunks, root_chunks = self._read_history(initial_history)
        prefix_frames = sum(len(chunk) for chunk in local_chunks)
        name = f"ardy_live_{time.time_ns()}"

        with torch.no_grad():
            for index in range(total_steps):
                window_started = time.perf_counter()
                if cancelled():
                    raise GenerationCancelled()
                with inference_lock if inference_lock is not None else nullcontext():
                    if cancelled():
                        raise GenerationCancelled()
                    motion = self.model.autoregressive_step(
                        num_frames=(0 if history is None else history.shape[1]) + self.horizon,
                        num_denoising_steps=self.num_denoising_steps,
                        motion_mask=None,
                        observed_motion=None,
                        cfg_weight=(2.0, 2.0),
                        text_feat=text_feat,
                        text_pad_mask=text_pad_mask,
                        init_history_sequence=history,
                    )
                chunk = motion[:, -self.horizon :]
                local, root = self._decode_chunk(chunk)
                if cancelled():
                    raise GenerationCancelled()
                frame_count = min(self.horizon, total_frames - index * self.horizon)
                local = local[:frame_count]
                root = root[:frame_count]
                if root_origin is None:
                    root_origin = root[0].copy()
                if local_chunks:
                    self._blend_boundary(local, root, local_chunks[-1], root_chunks[-1] + root_origin)
                root_delta = root - root_origin
                local_chunks.append(local)
                root_chunks.append(root_delta)

                event = self._chunk_event(name, prompt, index, total_steps, local, root_delta, prefix_frames, root_origin)
                event["generationMs"] = round((time.perf_counter()-window_started)*1000, 2)
                emit(event)

                # Feed back the corrected frames, rather than the raw model
                # window. Otherwise each later autoregressive window can
                # reintroduce the residual that was removed at the boundary.
                history = self._emitted_history(local_chunks, root_chunks, root_origin)

        return self._motion_document(name, prompt, prefix_frames + total_frames, local_chunks, root_chunks, root_origin)
