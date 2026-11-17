"""Geometry-based motion quality metrics; contact slip is separate from continuity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
FOOT_NAMES = ['leftFoot', 'leftToes', 'rightFoot', 'rightToes']


def joints_from_motion(motion):
    rest = np.asarray(motion.get('restJoints',
        json.loads((ROOT/'tests/fixtures/core-rest.json').read_text()))).reshape(-1, 3)
    root = np.asarray(motion['rootDelta']).reshape(-1, 3)+motion.get('rootOrigin', [0, .92, 0])
    count = len(motion['parents'])
    local = Rotation.from_rotvec(np.asarray(motion['localRot']).reshape(-1, 3)).as_matrix().reshape(-1, count, 3, 3)
    world, positions = local.copy(), np.empty((len(root), count, 3))
    positions[:, 0] = root
    for joint, parent in enumerate(motion['parents'][1:], 1):
        world[:, joint] = world[:, parent]@local[:, joint]
        positions[:, joint] = positions[:, parent]+np.einsum('tij,j->ti', world[:, parent], rest[joint]-rest[parent])
    return positions


def support_mask(motion, feet):
    speed = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2)*motion['fps']
    if motion.get('footContacts'):
        contacts = np.asarray(motion['footContacts']).reshape(-1, 4).astype(bool)
        mask = contacts[:-1]&contacts[1:]
    else:
        mask = (feet[:-1, :, 1] < .10)&(feet[1:, :, 1] < .10)&(speed < .15)
    return mask, speed


def motion_metrics(motion):
    joints = joints_from_motion(motion)
    feet = joints[:, [motion['bones'].index(name) for name in FOOT_NAMES]]
    support, speed = support_mask(motion, feet)
    if not support.any():
        raise AssertionError('没有可测量的支撑帧，不能将空集报告为零脚滑')
    root_speed = np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=1)*motion['fps']
    return {'supportSamples': int(support.sum()), 'supportSpeedMean': float(speed[support].mean()),
            'supportSpeedP95': float(np.percentile(speed[support], 95)),
            'rootSpeedMean': float(root_speed.mean()), 'rootHeightRange': float(np.ptp(joints[:, 0, 1])),
            'footHeightMin': float(feet[:, :, 1].min())}


def retarget_metrics(motion, entry):
    joints = joints_from_motion(motion)
    feet = joints[:, [motion['bones'].index(name) for name in FOOT_NAMES]]
    support, source_speed = support_mask(motion, feet)
    target = np.asarray([[frame[name] for name in FOOT_NAMES] for frame in entry['samples']])
    speed = np.linalg.norm(np.diff(target[:, :, [0, 2]], axis=0), axis=2)*motion['fps']
    return {'supportSamples': int(support.sum()), 'sourceSupportSpeedMean': float(source_speed[support].mean()),
            'targetSupportSpeedMean': float(speed[support].mean()),
            'targetSupportSpeedP95': float(np.percentile(speed[support], 95)),
            'footHeightMin': float(target[:, :, 1].min())}


def read_browser_result(path):
    text = Path(path).read_text()
    return json.loads(text.split('### Result\n', 1)[1].splitlines()[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('motion', type=Path)
    parser.add_argument('--browser', type=Path)
    args = parser.parse_args()
    motion = json.loads(args.motion.read_text())
    result = motion_metrics(motion)
    if args.browser:
        result['avatars'] = {entry['file']: retarget_metrics(motion, entry)
                             for entry in read_browser_result(args.browser)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
