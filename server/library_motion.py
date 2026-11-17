"""Rigid clip reuse, boundary blending and geometry checks; no foot IK."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from collision import body_collisions
from world import segment_clear

FPS = 20


def arrays(motion):
    frames = motion.get('frames', 0)
    if type(frames) is not int or not 1 <= frames <= 3760:
        raise ValueError('动作帧数无效')
    if motion.get('fps') != FPS or motion.get('up') != 'yup' or len(motion.get('bones', [])) != 27:
        raise ValueError('动作库需要 27 骨骼、20 fps、Y-up 动作')
    local = np.asarray(motion['localRot'], dtype=float).reshape(frames, 27, 3)
    root = np.asarray(motion['rootDelta'], dtype=float).reshape(frames, 3)
    if not np.isfinite(local).all() or not np.isfinite(root).all():
        raise ValueError('动作包含非有限数值')
    return local, root


def yaw(local):
    forward = Rotation.from_rotvec(local[..., 0, :]).apply([0., 0., 1.])
    return np.arctan2(forward[..., 0], forward[..., 2])


def frame_of(plan, history):
    angle = float(yaw(arrays(history)[0][-1])) if history else float(plan['headingTrack'][0])
    return np.asarray(plan['start'], dtype=float)-[0, .92, 0], angle


def profile(plan, history):
    origin, angle = frame_of(plan, history)
    rotation = Rotation.from_euler('y', -angle)
    track = np.asarray(plan['rootTrack'])-np.asarray(plan['start'])
    indices = np.linspace(0, len(track)-1, min(32, len(track))).astype(int)
    headings = np.asarray(plan['headingTrack'])[indices]-angle
    headings = np.arctan2(np.sin(headings), np.cos(headings))
    path = rotation.apply(track[indices]).round(2)
    path[np.abs(path) < .005] = 0
    headings = headings.round(2)
    headings[np.abs(headings) < .005] = 0
    descriptor = {'steps': [(s['action'], s['start'], s['end']) for s in plan['timeline']],
                  'frames': plan['frames'], 'height': round(float(plan['start'][1]), 1),
                  'path': path.tolist(), 'headings': headings.tolist()}
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
    return key, origin, angle


def canonical(plan, motion, prefix, history):
    local, root = arrays(motion)
    if len(root)-prefix != plan['frames']:
        raise ValueError('入库片段时长与规划不一致')
    _, origin, angle = profile(plan, history)
    rotation = Rotation.from_euler('y', -angle)
    local, root = local[prefix:].copy(), rotation.apply(root[prefix:]-origin)
    local[:, 0] = (rotation*Rotation.from_rotvec(local[:, 0])).as_rotvec()
    result = {k: copy.deepcopy(motion[k]) for k in
              ('version', 'name', 'source', 'dataset', 'fps', 'bones', 'parents', 'up', 'restJoints')}
    result.update(frames=len(root), duration=len(root)/FPS, rootOrigin=[0, .92, 0],
                  localRot=local.round(6).ravel().tolist(), rootDelta=root.round(6).ravel().tolist())
    if motion.get('footContacts'):
        result['footContacts'] = motion['footContacts'][prefix*4:]
    return result


def joints(motion):
    local, root = arrays(motion)
    parents = motion['parents']
    if len(parents) != 27 or parents[0] != -1 or any(p < 0 or p >= i for i, p in enumerate(parents[1:], 1)):
        raise ValueError('动作骨骼父子结构无效')
    rest = np.asarray(motion['restJoints'], dtype=float).reshape(27, 3)
    if not np.isfinite(rest).all():
        raise ValueError('动作骨架包含非有限数值')
    matrices = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(-1, 27, 3, 3)
    world, positions = matrices.copy(), np.empty((len(root), 27, 3))
    positions[:, 0] = root+[0, .92, 0]
    for joint, parent in enumerate(parents[1:], 1):
        world[:, joint] = world[:, parent]@matrices[:, joint]
        positions[:, joint] = positions[:, parent]+np.einsum('tij,j->ti', world[:, parent], rest[joint]-rest[parent])
    return positions


def quality(motion):
    local, root = arrays(motion)
    positions = joints(motion)
    if len(root) < 2:
        raise ValueError('动作过短，无法检查连续性')
    rotations = Rotation.from_rotvec(local.reshape(-1, 3)).as_quat().reshape(-1, 27, 4)
    dots = np.abs((rotations[1:]*rotations[:-1]).sum(axis=2))
    angular = 2*np.arccos(np.clip(dots, 0, 1))
    feet = positions[:, [25, 26, 21, 22]]
    speed = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2)*FPS
    floor = np.percentile(feet[:, :, 1], 15, axis=0)
    support = np.maximum(feet[:-1, :, 1], feet[1:, :, 1]) < floor+.025
    support &= np.abs(np.diff(feet[:, :, 1], axis=0))*FPS < .12
    slip = speed[support]
    report = {'angularStepMax': float(angular.max()), 'rootStepMax': float(np.linalg.norm(np.diff(root, axis=0), axis=1).max()),
              'supportSamples': int(support.sum()), 'supportMean': float(slip.mean()) if len(slip) else None,
              'supportP95': float(np.percentile(slip, 95)) if len(slip) else None,
              'footHeightMin': float(feet[:, :, 1].min())}
    if report['angularStepMax'] > .85 or report['rootStepMax'] > .30:
        raise ValueError('动作关节或根节点存在跳变')
    if report['footHeightMin'] < -.16:
        raise ValueError('动作脚部穿地过深，不入播放库')
    if len(slip) and (report['supportMean'] > .22 or report['supportP95'] > .75):
        raise ValueError('动作支撑脚滑动过大，不入播放库')
    return report


def blend_start(local, prefix):
    old, _ = arrays(prefix)
    first = Rotation.from_rotvec(local[0])
    residual = Rotation.from_rotvec(old[-1])*first.inv()
    for frame in range(min(8, len(local))):
        t = (frame+1)/8
        weight = 1-t*t*t*(10+t*(-15+6*t))
        local[frame] = (Rotation.from_rotvec(residual.as_rotvec()*weight)*Rotation.from_rotvec(local[frame])).as_rotvec()


def replay(motion, plan, history):
    local, root = arrays(motion)
    _, origin, angle = profile(plan, history)
    alignment = Rotation.from_euler('y', angle)
    local, root = local.copy(), alignment.apply(root)+origin
    local[:, 0] = (alignment*Rotation.from_rotvec(local[:, 0])).as_rotvec()
    if history:
        blend_start(local, history)
        previous = np.asarray(history['rootDelta'][-3:])
        count = min(8, len(root))
        t = np.arange(1, count+1)/8
        root[:count, 1] += (previous[1]-root[0, 1])*(1-t*t*t*(10+t*(-15+6*t)))
        if np.linalg.norm(root[0]-previous) > .15:
            raise ValueError('动作库接缝根位移过大')
    result = copy.deepcopy(motion)
    result.update(localRot=local.ravel().tolist(), rootDelta=root.ravel().tolist())
    quality(result)
    check_plan(result, plan)
    if history:
        result['localRot'] = history['localRot']+result['localRot']
        result['rootDelta'] = history['rootDelta']+result['rootDelta']
        result['footContacts'] = history.get('footContacts', [0]*(history['frames']*4))+result.get('footContacts', [0]*(result['frames']*4))
        result['frames'] += history['frames']
        result['duration'] = result['frames']/FPS
    return result


def check_plan(motion, plan):
    _, root = arrays(motion)
    track = np.asarray(plan['rootTrack'])
    if len(root) != len(track) or np.max(np.linalg.norm((root+[0, .92, 0]-track)[:, [0, 2]], axis=1)) > .16:
        raise ValueError('库内动作根轨迹与当前路线不匹配')
    world_root = root+[0, .92, 0]
    if any(not segment_clear(a[[0, 2]], b[[0, 2]]) for a, b in zip(world_root, world_root[1:])):
        raise ValueError('库内动作根运动碰撞')
    if body_collisions(joints(motion), motion['parents']):
        raise ValueError('库内动作身体碰撞，需重新生成')
