"""Finalize one ARDY window and validate its pose against the planning map."""
import numpy as np
from scipy.spatial.transform import Rotation

from ardy_keyframes import angle_delta, facing
from constrained import PARENTS
from motion_quality import match_boundary, restore_contacts
from touch_ik import forward


def anchor_poses(local, plan, start):
    poses = plan['staticPoses']
    keys = sorted(poses)
    for index, values in enumerate(local):
        frame = start+index
        yaw = plan['headingTrack'][frame]
        hips = Rotation.from_rotvec(values[:3])
        values[:3] = (Rotation.from_rotvec([0, angle_delta(yaw, facing(values)), 0])*hips).as_rotvec()
        stationary = next((a for a, b in zip(keys, keys[1:]) if a < frame < b and poses[a] == poses[b]
                           and plan['rootTrack'][a] == plan['rootTrack'][b]), None)
        nearest = min(keys, key=lambda key: (abs(frame-key), key))
        distance = abs(frame-nearest)
        if stationary is not None or distance == 0:
            local[index] = poses[nearest if stationary is None else stationary]
        elif distance < 8:
            target = np.asarray(poses[nearest]).reshape(27, 3).copy()
            target[0, 1] = yaw  # keep the compiled turn timing even near an anchor
            blend_pose(values, target, (1-distance/8)**2)


def blend_pose(values, target, weight):
    source = Rotation.from_rotvec(values.reshape(27, 3))
    delta = source.inv()*Rotation.from_rotvec(target)
    values[:] = (source*Rotation.from_rotvec(delta.as_rotvec()*weight)).as_rotvec().ravel()


def finish_window(runner, window, local, root, plan, start, count, origin, world):
    output = match_boundary(window.slice(0, count), local, root, preserve_root=True)
    output = restore_contacts(runner, output, local, root, window.slice(count), preserve_root=True)
    shift = origin-np.array([0, .92, 0], dtype=np.float32)
    output.root[:] = np.asarray(plan['rootTrack'][start:start+count])+shift
    anchor_poses(output.local, plan, start)
    if not np.isfinite(output.local).all() or not np.isfinite(output.root).all():
        raise ValueError('ARDY 窗口包含非有限值，未发布')
    rest = runner.model.motion_rep.skeleton.neutral_joints.cpu().numpy()
    joints = [forward(pose, position-shift, rest, PARENTS)[0] for pose, position in zip(output.local, output.root)]
    if len(root):
        joints.insert(0, forward(local[-1], root[-1]-shift, rest, PARENTS)[0])
    if not world.clear_body(joints, PARENTS):
        raise ValueError('ARDY 生成的骨段与实测点云净空不足，窗口未发布')
    return output
