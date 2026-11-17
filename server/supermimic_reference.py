"""Exact G1 reference feature layout, Z-up, quaternions xyzw."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DT = .02


def rotation6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def yaw_matrix(matrix: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return Rotation.from_euler('z', yaw).as_matrix()


@dataclass
class G1Reference:
    position: np.ndarray
    rotation: np.ndarray
    joints: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_velocity: np.ndarray

    @classmethod
    def from_csv(cls, path: Path, fps: float = 30) -> G1Reference:
        raw = np.loadtxt(path, delimiter=',')
        return cls.from_samples(raw[:, :3], raw[:, 3:7], raw[:, 7:36], fps)

    @classmethod
    def from_samples(cls, position: np.ndarray, quaternion: np.ndarray,
                     joints: np.ndarray, fps: float = 50) -> G1Reference:
        count = len(position)
        if not 2 <= count <= 30001 or not .1 <= fps <= 240:
            raise ValueError('Unsupported reference length or FPS')
        for value, shape in [(position, (count, 3)), (quaternion, (count, 4)), (joints, (count, 29))]:
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError('Invalid G1 reference data')
        if np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1)) > .01:
            raise ValueError('Reference quaternion must be unit xyzw')
        source_times = np.arange(count) / fps
        if not DT <= source_times[-1] <= 600:
            raise ValueError('Reference duration must be between .02 and 600 seconds')
        times = np.arange(int(np.floor(source_times[-1] / DT + 1e-8)) + 1) * DT
        times = np.minimum(times, source_times[-1])
        p = np.stack([np.interp(times, source_times, position[:, i]) for i in range(3)], axis=1)
        q = np.stack([np.interp(times, source_times, joints[:, i]) for i in range(29)], axis=1)
        r = Slerp(source_times, Rotation.from_quat(quaternion))(times).as_matrix()
        w = np.zeros_like(p)
        w[1:-1] = Rotation.from_matrix(r[2:] @ r[:-2].transpose(0, 2, 1)).as_rotvec() / (2 * DT)
        w[0] = Rotation.from_matrix(r[1] @ r[0].T).as_rotvec() / DT
        w[-1] = Rotation.from_matrix(r[-1] @ r[-2].T).as_rotvec() / DT
        return cls(p, r, q, np.gradient(p, DT, axis=0), w, np.gradient(q, DT, axis=0))

    def goal(self, frame: int) -> np.ndarray:
        requested = np.arange(frame + 1, frame + 11)
        future = np.minimum(requested, len(self.joints) - 1)
        recorded = (requested < len(self.joints))[:, None]
        inverse = self.rotation[future].transpose(0, 2, 1)
        v = np.einsum('nij,nj->ni', inverse, self.velocity[future]) * recorded
        w = np.einsum('nij,nj->ni', inverse, self.angular_velocity[future]) * recorded
        heading = yaw_matrix(self.rotation[frame])
        delta = (self.position[future] - self.position[frame]) @ heading
        orient = rotation6d(heading.T @ self.rotation[future])
        values = np.concatenate((delta, orient, v, w, self.joints[future],
                                 self.joint_velocity[future] * recorded * .1), axis=1)
        return values.reshape(1, 730).astype(np.float32)


def observation(q: np.ndarray, qd: np.ndarray, rotation: np.ndarray, angular_world: np.ndarray,
                previous: np.ndarray, reference_rotation: np.ndarray, default: np.ndarray) -> np.ndarray:
    return np.concatenate((q - default, qd, rotation.T @ angular_world, -rotation[2], previous,
                           rotation6d(rotation.T @ reference_rotation))).astype(np.float32)
