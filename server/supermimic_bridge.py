"""Human -> calibrated G1 -> physical policy tracking -> human rotation residuals."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic_physics import G1Physics
from supermimic_policy import SuperMimicPolicy
from supermimic_reference import DT, G1Reference, observation
from supermimic_skeleton import G1Skeleton, HumanMotion, HUMAN_TO_G1
from supermimic_human_ik import reconstruct_feet


@dataclass
class AdaptedMotion:
    human: HumanMotion
    skeleton: G1Skeleton
    reference: G1Reference
    fit_errors: list[float]
    fit_ms: list[float]


def sole_height(skeleton: G1Skeleton, joints: np.ndarray, rotation: np.ndarray) -> float:
    skeleton.pose(joints)
    sim = skeleton.sim
    body_ids = [sim.model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    heights = []
    for i in range(sim.model.ngeom):
        if sim.model.geom_bodyid[i] not in body_ids or sim.model.geom_contype[i] == 0:
            continue
        body = sim.model.geom_bodyid[i]
        point = sim.data.xpos[body] + sim.data.xmat[body].reshape(3, 3) @ sim.model.geom_pos[i]
        heights.append(float((rotation @ point)[2] - sim.model.geom_size[i, 0]))
    if not heights:
        raise ValueError('G1 foot collision geometry is missing')
    return min(heights)


def adapt_human(human: HumanMotion, policy: SuperMimicPolicy) -> AdaptedMotion:
    skeleton = G1Skeleton(G1Physics(policy.config['control']['joint_names']), human)
    previous, joints, errors, durations = skeleton.rest_q, [], [], []
    for frame in range(len(human.root)):
        started = time.perf_counter()
        previous, error, _ = skeleton.fit(frame, previous)
        joints.append(previous); errors.append(error)
        durations.append((time.perf_counter() - started) * 1000)
    rotations = HUMAN_TO_G1 @ human.world[:, 0] @ HUMAN_TO_G1.T
    floor_height = -sole_height(skeleton, joints[0], rotations[0])
    position = (human.root - human.root[0]) @ HUMAN_TO_G1.T * skeleton.scale
    position[:, 2] += floor_height
    duration = (len(human.root) - 1) / human.document['fps']
    count = int(np.ceil(duration * 50 - 1e-8)) + 1
    mapped = human.resample(50, count)
    source = np.arange(len(human.root)) / human.document['fps']
    target = np.arange(count) / 50
    positions = np.stack([np.interp(target, source, position[:, i]) for i in range(3)], axis=1)
    joint_values = np.stack([np.interp(target, source, np.array(joints)[:, i]) for i in range(29)], axis=1)
    rotation_values = HUMAN_TO_G1 @ mapped.world[:, 0] @ HUMAN_TO_G1.T
    reference = G1Reference.from_samples(positions, Rotation.from_matrix(rotation_values).as_quat(), joint_values, 50)
    return AdaptedMotion(human, skeleton, reference, errors, durations)


def track_adapted(adapted: AdaptedMotion, policy: SuperMimicPolicy) -> tuple[dict, dict]:
    sim, ref = adapted.skeleton.sim, adapted.reference
    sim.reset(ref.position[0], ref.rotation[0], ref.joints[0], ref.velocity[0], ref.angular_velocity[0], ref.joint_velocity[0])
    history, previous = None, np.zeros(29)
    poses, rotations, joints, times = [ref.position[0]], [ref.rotation[0]], [ref.joints[0]], []
    failure = None
    for frame in range(len(ref.joints) - 1):
        started = time.perf_counter()
        _, r, _, w, q, qd = sim.state()
        observed = observation(q, qd, r, w, previous, ref.rotation[frame], sim.default)
        history = np.repeat(observed[None, None], 10, axis=1) if history is None else np.concatenate((history[:, 1:], observed[None, None]), axis=1)
        previous = policy.predict(history, ref.goal(frame))
        sim.step(previous)
        p, r, _, _, q, _ = sim.state()
        poses.append(p); rotations.append(r); joints.append(q)
        times.append((time.perf_counter() - started) * 1000)
        if abs(p[2] - ref.position[frame + 1, 2]) > .25 or abs(r[2, 2] - ref.rotation[frame + 1, 2, 2]) > .8:
            failure = {'frame': frame + 1, 'reason': 'G1 reference height/tilt termination'}
            break
    return {'completed': failure is None, 'failure': failure, 'steps': len(times),
            'stepP95Ms': float(np.percentile(times, 95)), 'stepMaxMs': max(times)}, {
                'root': np.array(poses), 'rotation': np.array(rotations), 'joints': np.array(joints)}


def reverse_human(adapted: AdaptedMotion, trajectory: dict, reconstruct: bool = True,
                  foot_mode: str = 'residual') -> dict:
    if foot_mode not in ('residual', 'trajectory'):
        raise ValueError('Foot reconstruction must be residual or trajectory')
    skeleton, ref = adapted.skeleton, adapted.reference
    human = adapted.human.resample(50, len(ref.joints))
    count = len(ref.joints)
    if len(trajectory['root']) != count or len(human.root) != count:
        raise ValueError('Incomplete G1 trajectory cannot replace a humanoid clip')
    local, foot_residuals, robot_feet = np.zeros_like(human.local), [], []
    foot_indices = [skeleton.keys.index(side + 'Foot') for side in ('left', 'right')]
    for frame in range(count):
        actual, body_position = skeleton.pose(trajectory['joints'][frame])
        actual_feet = trajectory['root'][frame] + body_position[foot_indices] @ trajectory['rotation'][frame].T
        robot_feet.append(actual_feet)
        expected, expected_position = skeleton.pose(ref.joints[frame])
        expected_feet = ref.position[frame] + expected_position[foot_indices] @ ref.rotation[frame].T
        foot_residuals.append(actual_feet - expected_feet)
        actual = trajectory['rotation'][frame] @ actual
        expected = ref.rotation[frame] @ expected
        delta = HUMAN_TO_G1.T @ (actual @ expected.transpose(0, 2, 1)) @ HUMAN_TO_G1
        correction = {int(index): delta[i] for i, index in enumerate(skeleton.human_indices)}
        world = np.zeros((len(human.rest), 3, 3))
        for joint, parent in enumerate(human.document['parents']):
            if joint in correction:
                world[joint] = correction[joint] @ human.world[frame, joint]
            else:
                world[joint] = world[parent] @ Rotation.from_rotvec(human.local[frame, joint]).as_matrix()
            local_rotation = world[joint] if parent < 0 else world[parent].T @ world[joint]
            local[frame, joint] = Rotation.from_matrix(local_rotation).as_rotvec()
    displacement = (trajectory['root'] - ref.position) @ HUMAN_TO_G1 / skeleton.scale
    document = copy.deepcopy(human.document)
    document.update(name=human.document['name'] + ' · SuperMimic 候选', source=human.document.get('source', '') + '; G1 physical residual',
                    localRot=local.ravel().tolist(), rootDelta=(human.root + displacement).ravel().tolist(), fps=50)
    document.pop('footContacts', None)
    document.pop('search', None)
    if reconstruct:
        feet = human.positions[:, [human.semantic[side + 'Foot'] for side in ('left', 'right')]] + human.root[:, None]
        targets = feet + np.array(foot_residuals) @ HUMAN_TO_G1 / skeleton.scale
        if foot_mode == 'trajectory':
            targets = feet[0] + (np.array(robot_feet) - robot_feet[0]) @ HUMAN_TO_G1 / skeleton.scale
        rebuilt, diagnostics = reconstruct_feet(HumanMotion.load(document), targets)
        document['localRot'] = rebuilt.ravel().tolist()
        document['supermimicReconstruction'] = diagnostics
        document['supermimicReconstruction']['footMode'] = foot_mode
    return sample_residual(adapted.human, HumanMotion.load(document), human)


def sample_residual(original: HumanMotion, corrected: HumanMotion, basis: HumanMotion) -> dict:
    """Resample only corrections, so a zero correction preserves source samples exactly."""
    count = len(original.root)
    source_times = np.arange(len(corrected.root)) / 50
    target_times = np.arange(count) / original.document['fps']
    local = original.local.copy()
    for joint in range(len(original.rest)):
        current = Rotation.from_rotvec(corrected.local[:, joint])
        base = Rotation.from_rotvec(basis.local[:, joint])
        delta = current * base.inv()
        correction = Slerp(source_times, delta)(target_times)
        local[:, joint] = (correction * Rotation.from_rotvec(original.local[:, joint])).as_rotvec()
    delta = corrected.root - basis.root
    root = original.root + np.stack([np.interp(target_times, source_times, delta[:, i]) for i in range(3)], axis=1)
    output = copy.deepcopy(original.document)
    output.update(name=corrected.document['name'], source=corrected.document['source'],
                  localRot=local.ravel().tolist(), rootDelta=root.ravel().tolist())
    output.pop('footContacts', None)
    output.pop('search', None)
    for key in ('quality', 'footLift', 'distance', 'angle', 'heightOffset'):
        output.pop(key, None)
    if 'supermimicReconstruction' in corrected.document:
        output['supermimicReconstruction'] = corrected.document['supermimicReconstruction']
    return output
