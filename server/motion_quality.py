"""Preserve a shared gait phase and restore contacts before publishing motion."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from overlap import MotionWindow


TURN_YAW_THRESHOLD = np.deg2rad(18.0)
TURN_ROOT_TRAVEL_THRESHOLD = 0.08


def decode_window(runner, motion):
    decoded = runner.model.motion_rep.inverse(motion, is_normalized=True)
    matrices = decoded['local_rot_mats'][0].detach().cpu().numpy()
    roots = decoded['root_positions'][0].detach().cpu().numpy()
    contacts = decoded['foot_contacts'][0].detach().cpu().numpy()
    local = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(len(roots), -1)
    return MotionWindow(local.astype(np.float32), roots, contacts=contacts)


def overlap_condition(runner, pending, history_count):
    """The second instance observes the retained poses instead of inventing another stride."""
    if pending is None:
        return None, None
    from ardy.constraints import FullBodyConstraintSet
    decoded = runner.model.motion_rep.inverse(
        runner._encode_history(pending.local, pending.root), is_normalized=True)
    torch, rep = runner.torch, runner.model.motion_rep
    frames = torch.arange(history_count, history_count+len(pending.root))
    constraint = FullBodyConstraintSet(rep.skeleton, frames,
        decoded['posed_joints'][0], decoded['global_rot_mats'][0])
    return rep.create_conditions_from_constraints_batched([constraint],
        torch.tensor([history_count+runner.horizon], device=runner.device),
        to_normalize=True, device=runner.device)


def match_boundary(output, previous_local, previous_root, preserve_root=False):
    """Apply one short inertial correction, leaving the rest of the stride intact."""
    if not len(previous_root):
        return output
    count = min(4, len(output.root))
    last = Rotation.from_rotvec(previous_local[-1].reshape(-1, 3))
    before = Rotation.from_rotvec(previous_local[-min(2, len(previous_local))].reshape(-1, 3))
    step = (before.inv()*last).as_rotvec()
    step *= np.minimum(1, .20/np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-8))
    predicted = last*Rotation.from_rotvec(step)
    first = Rotation.from_rotvec(output.local[0].reshape(-1, 3))
    residual = (predicted*first.inv()).as_rotvec()
    velocity = previous_root[-1]-previous_root[-min(2, len(previous_root))]
    offset = previous_root[-1]+velocity-output.root[0]
    for frame in range(count):
        t = frame/max(1, count-1)
        weight = 1-t*t*(3-2*t)
        rotation = Rotation.from_rotvec(residual*weight)
        output.local[frame] = (rotation*Rotation.from_rotvec(output.local[frame].reshape(-1, 3))).as_rotvec().ravel()
        if not preserve_root:
            output.root[frame] += offset*weight
    return output


def _turn_in_place(poses, roots, start, count):
    """Detect a constrained yaw turn without relying on a specific plan type."""
    if count < 2 or start < 0 or start + count > len(roots):
        return False
    segment_roots = roots[start:start + count]
    travel = np.ptp(segment_roots[:, [0, 2]], axis=0).max()
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(len(roots), -1, 3, 3)
    forward = matrices[start:start + count, 0] @ np.array([0.0, 0.0, 1.0])
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    return bool(travel <= TURN_ROOT_TRAVEL_THRESHOLD and
                abs(float(yaw[-1] - yaw[0])) >= TURN_YAW_THRESHOLD)


def restore_contacts(runner, output, local, root, future, preserve_root=False):
    """Use ARDY's native contact solver, with published history pinned as full-body keys."""
    from ardy.constraints import FullBodyConstraintSet, Root2DConstraintSet
    from ardy.postprocess import post_process_motion
    prefix = min(20, len(root))
    suffix = 0 if future is None else min(4, len(future.root))
    poses = np.concatenate((local[-prefix:] if prefix else local[:0], output.local,
                            future.local[:suffix] if suffix else local[:0]))
    roots = np.concatenate((root[-prefix:] if prefix else root[:0], output.root,
                            future.root[:suffix] if suffix else root[:0]))
    torch, rep = runner.torch, runner.model.motion_rep
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(1, len(roots), -1, 3, 3)
    rotations = torch.tensor(matrices, dtype=torch.float32, device=runner.device)
    positions = torch.tensor(roots[None], dtype=torch.float32, device=runner.device)
    decoded = rep.inverse(rep(rotations, positions, to_normalize=True), is_normalized=True)
    contacts = decoded['foot_contacts'].clone()
    if output.contacts is not None:
        contacts[:, prefix:prefix+len(output.root)] = torch.tensor(output.contacts, device=runner.device)
    if _turn_in_place(poses, roots, prefix, len(output.root)):
        output.stats["turnInPlace"] = True
        output.stats["contactPolicy"] = "source-observed"
    constraints = []
    pinned = prefix+1 if prefix else 0
    if pinned:
        constraints.append(FullBodyConstraintSet(rep.skeleton, torch.arange(pinned),
            decoded['posed_joints'][0, :pinned], decoded['global_rot_mats'][0, :pinned]))
    if preserve_root:
        frames = torch.arange(len(roots))
        constraints.append(Root2DConstraintSet(rep.skeleton, frames, positions[0, :, [0, 2]]))
    corrected = post_process_motion(rotations, positions, contacts, rep.skeleton,
                                    constraint_lst=constraints or None)
    end = prefix+len(output.root)
    corrected_local = corrected['local_rot_mats'][0, prefix:end].cpu().numpy()
    output.local = Rotation.from_matrix(corrected_local.reshape(-1, 3, 3)).as_rotvec().reshape(len(output.root), -1).astype(np.float32)
    corrected_root = corrected['root_positions'][0, prefix:end].cpu().numpy()
    if preserve_root:
        corrected_root[:, [0, 2]] = output.root[:, [0, 2]]
    output.root = corrected_root
    output.contacts = contacts[0, prefix:end].cpu().numpy()
    return output
