"""Length-preserving two-bone reconstruction from actual G1 foot trajectories."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_skeleton import HumanMotion, forward


def align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if min(np.linalg.norm(source), np.linalg.norm(target)) < 1e-8:
        raise ValueError('Cannot align zero-length vectors')
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross, dot = np.cross(source, target), float(source @ target)
    if dot > 1 - 1e-10:
        return np.eye(3)
    if dot < -1 + 1e-8:
        helper = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, helper); axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + dot)


def knee_point(hip: np.ndarray, ankle: np.ndarray, guide: np.ndarray,
               upper_length: float, lower_length: float) -> tuple[np.ndarray, float]:
    vector = ankle - hip
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError('Degenerate human leg target')
    direction = vector / length
    distance = np.clip(length, abs(upper_length - lower_length) + 1e-6, upper_length + lower_length - 1e-6)
    along = (upper_length**2 - lower_length**2 + distance**2) / (2 * distance)
    perpendicular = guide - hip - direction * ((guide - hip) @ direction)
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.array([0., 0., 1.]) - direction * direction[2]
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.array([1., 0., 0.]) - direction * direction[0]
    perpendicular /= np.linalg.norm(perpendicular)
    knee = hip + direction * along + perpendicular * np.sqrt(max(0, upper_length**2 - along**2))
    return knee, max(0., length - upper_length - lower_length)


def reconstruct_feet(human: HumanMotion, foot_targets: np.ndarray) -> tuple[np.ndarray, dict]:
    if np.shape(foot_targets) != (len(human.root), 2, 3) or not np.isfinite(foot_targets).all():
        raise ValueError('Expected two finite foot targets for each human frame')
    local, errors, unreachable = human.local.copy(), [], []
    parents, rest = human.document['parents'], human.rest
    for frame in range(len(local)):
        world, positions = forward(local[frame:frame + 1], parents, rest)
        world, positions = world[0], positions[0] + human.root[frame]
        for side, label in enumerate(('left', 'right')):
            upper, knee, foot = [human.semantic[label + part] for part in ('UpperLeg', 'LowerLeg', 'Foot')]
            upper_vector, lower_vector = rest[knee] - rest[upper], rest[foot] - rest[knee]
            target_knee, excess = knee_point(positions[upper], foot_targets[frame, side], positions[knee],
                                             np.linalg.norm(upper_vector), np.linalg.norm(lower_vector))
            upper_world = align_vector(world[upper] @ upper_vector, target_knee - positions[upper]) @ world[upper]
            actual_knee = positions[upper] + upper_world @ upper_vector
            lower_world = align_vector(world[knee] @ lower_vector, foot_targets[frame, side] - actual_knee) @ world[knee]
            local[frame, upper] = Rotation.from_matrix(world[parents[upper]].T @ upper_world).as_rotvec()
            local[frame, knee] = Rotation.from_matrix(upper_world.T @ lower_world).as_rotvec()
            local[frame, foot] = Rotation.from_matrix(lower_world.T @ world[foot]).as_rotvec()
            actual = actual_knee + lower_world @ lower_vector
            errors.append(float(np.linalg.norm(actual - foot_targets[frame, side])))
            unreachable.append(excess)
    return local, {'footTargetMaxErrorM': max(errors), 'unreachableMaxM': max(unreachable)}
