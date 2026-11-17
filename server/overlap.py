"""Fuse windows covering the same timestamps, without modifying published frames."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class MotionWindow:
    local: np.ndarray
    root: np.ndarray
    stats: dict = field(default_factory=dict)
    contacts: np.ndarray | None = None

    def slice(self, start: int, end: int | None = None) -> MotionWindow:
        contacts = None if self.contacts is None else self.contacts[start:end].copy()
        return MotionWindow(self.local[start:end].copy(), self.root[start:end].copy(), self.stats, contacts)


def overlap_weights(count: int) -> np.ndarray:
    if count < 4:
        raise ValueError("重叠融合至少需要四帧")
    # Preserve both endpoint velocities: the first/last TWO samples belong
    # entirely to the old/new window. Quintic easing has zero end derivatives.
    t = np.clip((np.arange(count, dtype=np.float64) - 1) / (count - 3), 0, 1)
    return t**3 * (10 - 15*t + 6*t*t)


def fuse_windows(old: MotionWindow, new: MotionWindow) -> MotionWindow:
    if old.local.shape != new.local.shape or old.root.shape != new.root.shape:
        raise ValueError("重叠片段必须覆盖相同帧数和骨骼")
    if not all(np.isfinite(a).all() for a in (old.local, new.local, old.root, new.root)):
        raise ValueError("重叠片段包含非有限坐标")
    count = len(old.root)
    weights = overlap_weights(count)
    joints = old.local.shape[1] // 3
    first = Rotation.from_rotvec(old.local.reshape(-1, 3))
    second = Rotation.from_rotvec(new.local.reshape(-1, 3))
    residual = (first.inv() * second).as_rotvec()
    rotation = first * Rotation.from_rotvec(residual * np.repeat(weights, joints)[:, None])
    local = rotation.as_rotvec().reshape(count, -1).astype(np.float32)
    root = (old.root * (1-weights[:, None]) + new.root * weights[:, None]).astype(np.float32)
    contacts = new.contacts if old.contacts is None else old.contacts
    if old.contacts is not None and new.contacts is not None:
        contacts = np.where(weights[:, None] < .5, old.contacts, new.contacts)
    return MotionWindow(local, root, merge_stats(old, new), contacts)


def concatenate_windows(first: MotionWindow, second: MotionWindow) -> MotionWindow:
    return MotionWindow(np.concatenate((first.local, second.local)),
                        np.concatenate((first.root, second.root)), merge_stats(first, second))


def merge_stats(first: MotionWindow, second: MotionWindow) -> dict:
    return {"refinementPasses": first.stats.get("refinementPasses", 0) + second.stats.get("refinementPasses", 0),
            "rootProjectionMax": max(first.stats.get("rootProjectionMax", 0), second.stats.get("rootProjectionMax", 0))}


def smooth_window(window: MotionWindow, past: MotionWindow, future: MotionWindow | None,
                  preserve_root: bool = False) -> MotionWindow:
    """Five-tap quaternion filter with real lookahead, avoiding axis-angle wrap."""
    tail = np.repeat(window.local[-1:], 2, axis=0) if future is None else future.local[:2]
    head = past.local[-2:] if len(past.local) else window.local[:1]
    local = np.concatenate((head, window.local, tail))
    roots = np.concatenate((past.root[-len(head):] if len(past.root) else window.root[:1],
                            window.root, np.repeat(window.root[-1:], 2, axis=0) if future is None else future.root[:2]))
    left, right = max(0, 2-len(head)), max(0, 2-len(tail))
    local = np.pad(local, ((left, right), (0, 0)), mode="edge")
    roots = np.pad(roots, ((left, right), (0, 0)), mode="edge")
    quaternions = Rotation.from_rotvec(local.reshape(-1, 3)).as_quat().reshape(len(local), -1, 4)
    count = len(window.root)
    center = quaternions[2:2+count]
    rotations = np.zeros_like(center)
    positions = np.zeros_like(window.root)
    for offset, weight in enumerate((1/16, 4/16, 6/16, 4/16, 1/16)):
        neighbors = quaternions[offset:offset+count]
        sign = np.where(np.sum(center*neighbors, axis=2, keepdims=True) < 0, -1., 1.)
        rotations += neighbors*sign*weight
        positions += roots[offset:offset+count]*weight
    local = Rotation.from_quat(rotations.reshape(-1, 4)).as_rotvec().reshape(count, -1)
    return MotionWindow(local.astype(np.float32), window.root.copy() if preserve_root else positions, window.stats)
