"""Semantic calibration and bounded orientation fitting between humanoids and G1."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from supermimic_physics import G1Physics

# Both frames are right handed: humanoid +X left/+Y up/+Z forward;
# G1 +X forward/+Y left/+Z up. Apply conjugation to world delta rotations.
HUMAN_TO_G1 = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
BODY_MAP = {
    'hips': 'pelvis', 'chest': 'torso_link',
    'leftUpperLeg': 'left_hip_yaw_link', 'leftLowerLeg': 'left_knee_link', 'leftFoot': 'left_ankle_roll_link',
    'rightUpperLeg': 'right_hip_yaw_link', 'rightLowerLeg': 'right_knee_link', 'rightFoot': 'right_ankle_roll_link',
    'leftUpperArm': 'left_shoulder_yaw_link', 'leftLowerArm': 'left_elbow_link', 'leftHand': 'left_wrist_yaw_link',
    'rightUpperArm': 'right_shoulder_yaw_link', 'rightLowerArm': 'right_elbow_link', 'rightHand': 'right_wrist_yaw_link',
}


def forward(local: np.ndarray, parents: list[int], rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrices = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(*local.shape[:-1], 3, 3)
    world, positions = matrices.copy(), np.zeros(local.shape)
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        world[:, joint] = world[:, parent] @ matrices[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum('nij,j->ni', world[:, parent], rest[joint] - rest[parent])
    return world, positions


@dataclass
class HumanMotion:
    document: dict
    local: np.ndarray
    root: np.ndarray
    rest: np.ndarray
    world: np.ndarray
    positions: np.ndarray
    semantic: dict[str, int]

    @classmethod
    def load(cls, document: dict) -> HumanMotion:
        bones, parents = document.get('bones', []), document.get('parents', [])
        frames, fps = document.get('frames', 0), document.get('fps', 0)
        if not isinstance(bones, list) or not isinstance(parents, list) or not all(isinstance(name, str) for name in bones):
            raise ValueError('Expected bone names and parent indices as lists')
        if (document.get('up') != 'yup' or type(frames) is not int or not 2 <= frames <= 30001
                or type(fps) not in (int, float) or not 1 <= fps <= 240 or (frames - 1) / fps > 600
                or not 1 <= len(bones) <= 128 or len(parents) != len(bones)):
            raise ValueError('Expected a finite Y-up humanoid motion of at most 600 seconds')
        if type(parents[0]) is not int or parents[0] != -1 or any(type(p) is not int or not 0 <= p < i for i, p in enumerate(parents[1:], 1)):
            raise ValueError('Skeleton must have one topologically ordered root')
        if not all(key in document for key in ('localRot', 'rootDelta', 'restJoints')):
            raise ValueError('Humanoid motion requires localRot, rootDelta and restJoints')
        local = np.asarray(document['localRot'], dtype=float).reshape(frames, len(bones), 3)
        root = np.asarray(document['rootDelta'], dtype=float).reshape(frames, 3)
        rest = np.asarray(document['restJoints'], dtype=float).reshape(len(bones), 3)
        if not all(np.isfinite(value).all() for value in (local, root, rest)):
            raise ValueError('Humanoid motion contains nonfinite values')
        semantic = {name: i for i, name in enumerate(bones) if name}
        if any(name not in semantic for name in BODY_MAP) or semantic['hips'] != 0:
            raise ValueError('Missing humanoid semantic joints required for G1 calibration')
        for side in ('left', 'right'):
            for limb in ('Leg', 'Arm'):
                names = (side + 'Upper' + limb, side + 'Lower' + limb, side + ('Foot' if limb == 'Leg' else 'Hand'))
                lengths = np.linalg.norm(np.diff(rest[[semantic[name] for name in names]], axis=0), axis=1)
                if np.min(lengths) < 1e-5:
                    raise ValueError('Humanoid limb segments must have nonzero length')
        world, positions = forward(local, parents, rest)
        return cls(document, local, root, rest, world, positions, semantic)

    def resample(self, fps: float, frames: int | None = None) -> HumanMotion:
        if type(fps) not in (int, float) or not np.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError('Resampling FPS must be finite and between 1 and 240')
        if frames is not None and (type(frames) is not int or not 2 <= frames <= 30001):
            raise ValueError('Resampling requires 2 to 30001 frames')
        source = np.arange(len(self.root)) / self.document['fps']
        count = int(np.floor(source[-1] * fps + 1e-8)) + 1 if frames is None else frames
        target = np.arange(count) / fps
        samples = np.minimum(target, source[-1])
        local = np.stack([Slerp(source, Rotation.from_rotvec(self.local[:, j]))(samples).as_rotvec()
                          for j in range(len(self.rest))], axis=1)
        root = np.stack([np.interp(samples, source, self.root[:, i]) for i in range(3)], axis=1)
        document = {**self.document, 'fps': fps, 'frames': len(target), 'duration': float(target[-1]),
                    'localRot': local.ravel().tolist(), 'rootDelta': root.ravel().tolist()}
        document.pop('footContacts', None)
        return HumanMotion.load(document)


class G1Skeleton:
    def __init__(self, physics: G1Physics, human: HumanMotion):
        self.sim, self.human = physics, human
        self.keys = list(BODY_MAP)
        self.human_indices = np.array([human.semantic[name] for name in self.keys])
        self.body_ids = np.array([physics.model.body(BODY_MAP[name]).id for name in self.keys])
        self.rest_q = np.zeros(29)
        for name, value in [('left_shoulder_roll_joint', np.pi / 2), ('right_shoulder_roll_joint', -np.pi / 2),
                            ('left_elbow_joint', np.pi / 2), ('right_elbow_joint', np.pi / 2)]:
            index = np.flatnonzero(physics.q_indices == physics.model.joint(name).qposadr[0])[0]
            self.rest_q[index] = value
        self.rest_rotation, self.rest_position = self.pose(self.rest_q)
        self.scale = self._scale()
        self.weights = np.array([0, .7, 1, 1, 1.4, 1, 1, 1.4, .6, .6, .4, .6, .6, .4])

    def _scale(self) -> float:
        lengths = []
        for side in ('left', 'right'):
            hip, knee, foot = [self.keys.index(side + part) for part in ('UpperLeg', 'LowerLeg', 'Foot')]
            human = self.human.rest[self.human_indices[[hip, knee, foot]]]
            robot = self.rest_position[[hip, knee, foot]].copy()
            robot[0] = self.sim.data.xpos[self.sim.model.body(side + '_hip_pitch_link').id]
            lengths.append(np.linalg.norm(np.diff(robot, axis=0), axis=1).sum() /
                           np.linalg.norm(np.diff(human, axis=0), axis=1).sum())
        scale = float(np.mean(lengths))
        if not .2 <= scale <= 3:
            raise ValueError('Humanoid/G1 leg calibration outside supported range')
        return scale

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data, model = self.sim.data, self.sim.model
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[self.sim.q_indices] = q
        mujoco.mj_kinematics(model, data)
        return data.xmat[self.body_ids].reshape(-1, 3, 3).copy(), data.xpos[self.body_ids].copy()

    def desired(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        world = self.human.world[frame, self.human_indices]
        robot_world = HUMAN_TO_G1 @ world @ HUMAN_TO_G1.T
        root = robot_world[0]
        relative = root.T @ robot_world
        return relative @ self.rest_rotation, root

    def fit(self, frame: int, previous: np.ndarray) -> tuple[np.ndarray, float, int]:
        target, _ = self.desired(frame)
        feet = np.array([self.keys.index(side + 'Foot') for side in ('left', 'right')])
        human = self.human
        relative = (human.positions[frame, self.human_indices] @ human.world[frame, 0]) @ HUMAN_TO_G1.T
        resting = (human.rest[self.human_indices] - human.rest[0]) @ HUMAN_TO_G1.T
        positions = self.rest_position + (relative - resting) * self.scale

        def residual(q: np.ndarray) -> np.ndarray:
            rotation, position = self.pose(q)
            error = Rotation.from_matrix(target @ rotation.transpose(0, 2, 1)).as_rotvec() * self.weights[:, None]
            foot_error = (position[feet] - positions[feet]) * 8
            return np.concatenate((error.ravel(), foot_error.ravel(), .025 * (q - previous)))

        low, high = self.sim.limits[:, 0] + 1e-5, self.sim.limits[:, 1] - 1e-5
        result = least_squares(residual, np.clip(previous, low, high), bounds=(low, high),
                               max_nfev=35, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        if not result.success:
            raise ValueError(f'G1 fitting did not converge at frame {frame}: {result.message}')
        error = residual(result.x)[:len(self.keys) * 3].reshape(-1, 3)
        return result.x, float(np.sqrt(np.mean(np.sum(error**2, axis=1)))), result.nfev
