"""Build a small, reproducible root-motion bank from local AMASS recordings.

No joint synthesis, contact rewriting, foot locking or runtime inference.
Only coordinate conversion, rigid alignment and lossless body-chain folding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'server'))
from settings import configured_path
DEFAULT_DATA = configured_path('MOTION_AMASS_DIR', 'datasets/amass/raw')
FPS = 20
AXIS = Rotation.from_euler('x', -90, degrees=True)
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25]
BONES = ['hips', 'spine', '', 'chest', 'chest', 'neck', 'head', 'rightShoulder',
         'rightUpperArm', 'rightLowerArm', 'rightHand', '', '', 'leftShoulder',
         'leftUpperArm', 'leftLowerArm', 'leftHand', '', '', 'rightUpperLeg',
         'rightLowerLeg', 'rightFoot', 'rightToes', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes']
SMPL_MAP = {0: 0, 1: 3, 3: 6, 4: 9, 5: 12, 6: 15, 7: 14, 8: 17,
            9: 19, 10: 21, 13: 13, 14: 16, 15: 18, 16: 20,
            19: 2, 20: 5, 21: 8, 22: 11, 23: 1, 24: 4, 25: 7, 26: 10}
REST = np.asarray(json.loads((PROJECT/'server/assets/core-rest.json').read_text()))
FEET = [25, 26, 21, 22]


def load_recording(path):
    with np.load(path, allow_pickle=False) as data:
        fps = float(data['mocap_framerate'] if 'mocap_framerate' in data else data['mocap_frame_rate'])
        indices = np.arange(0, len(data['poses'])-1, fps/FPS).astype(int)
        smpl = data['poses'][indices, :66].reshape(-1, 22, 3)
        root = AXIS.apply(data['trans'][indices])
    local = np.zeros((len(indices), 27, 3))
    for target, source in SMPL_MAP.items():
        local[:, target] = smpl[:, source]
    local[:, 0] = (AXIS*Rotation.from_rotvec(local[:, 0])).as_rotvec()
    return {'local': local, 'root': root, 'source': '/'.join(path.parts[-3:])}


def geometry(recording):
    if '_feet' in recording:
        return recording['_feet']
    local, root = recording['local'], recording['root']
    matrices = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(-1, 27, 3, 3)
    world, positions = matrices.copy(), np.zeros((len(root), 27, 3))
    positions[:, 0] = root
    for joint, parent in enumerate(PARENTS[1:], 1):
        world[:, joint] = world[:, parent]@matrices[:, joint]
        positions[:, joint] = positions[:, parent]+np.einsum('tij,j->ti', world[:, parent], REST[joint]-REST[parent])
    recording['_feet'] = positions[:, FEET]
    return recording['_feet']


def heading(local):
    matrices = Rotation.from_rotvec(local[:, 0]).as_matrix()
    return np.unwrap(np.arctan2(matrices[:, 0, 2], matrices[:, 2, 2]))


def normalized(recording, begin, end):
    local = recording['local'][begin:end+1].copy()
    root = recording['root'][begin:end+1].copy()
    yaw = Rotation.from_euler('y', -heading(local)[0])
    local[:, 0] = (yaw*Rotation.from_rotvec(local[:, 0])).as_rotvec()
    root = yaw.apply(root-root[0])
    return local, root


def document(recording, begin, end, name, kind):
    local, root = normalized(recording, begin, end)
    feet = geometry(recording)[begin:end+1]
    speed = np.linalg.norm(np.gradient(feet, axis=0), axis=2)*FPS
    floor = np.percentile(feet[:, :, 1], 8)
    contacts = (feet[:, :, 1] < floor+.065)&(speed < .24)
    yaw = heading(local)
    distance = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1).sum()
    return {'version': 1, 'name': name, 'source': recording['source'], 'dataset': 'AMASS preset',
            'up': 'yup', 'fps': FPS, 'frames': len(root), 'duration': len(root)/FPS,
            'bones': BONES, 'parents': PARENTS, 'localRot': np.round(local, 6).ravel().tolist(),
            'rootDelta': np.round(root, 6).ravel().tolist(), 'rootOrigin': [0, .92, 0],
            'footContacts': contacts.astype(int).ravel().tolist(), 'restJoints': REST.ravel().tolist(),
            'kind': kind, 'angle': float(yaw[-1]-yaw[0]), 'distance': float(distance),
            'sourceHeight': float(np.median(recording['root'][begin:end+1, 1])),
            'heightOffset': float(recording['root'][begin, 1]-np.median(recording['root'][:20, 1])),
            'sourceFrames': [begin, end], 'footLift': np.ptp(feet[:, :, 1], axis=0).tolist()}


def turn_candidates(recordings, target, count=3):
    candidates = []
    for recording in recordings:
        yaw, feet = heading(recording['local']), geometry(recording)
        floor = np.percentile(feet[:, :, 1], 10)
        speed = np.linalg.norm(np.gradient(feet, axis=0), axis=2)*FPS
        for start in range(50, len(yaw)-16, 2):
            if speed[start:start+2, [0, 2]].max() > .45:
                continue
            for end in range(start+16, min(start+81, len(yaw)), 2):
                angle = yaw[end]-yaw[start]
                if abs(angle-target) > np.deg2rad(9):
                    continue
                endpoint_speed = speed[end-1:end+1, [0, 2]].max()
                if endpoint_speed > .35:
                    continue
                lift = np.ptp(feet[start:end+1, :, 1], axis=0)
                travel = np.linalg.norm(np.diff(recording['root'][start:end+1, [0, 2]], axis=0), axis=1).sum()
                # A twisting torso with planted feet is NOT a turn animation.
                if min(max(lift[:2]), max(lift[2:])) < .05 or max(lift) < .065 or travel > .65:
                    continue
                endpoint_lift = np.maximum(feet[[start, end], :, 1]-floor-.04, 0).mean()
                pose = recording['local'][[start, end]][:, [19, 20, 23, 24]]
                score = travel+endpoint_lift*8+endpoint_speed*3+np.linalg.norm(pose, axis=2).mean()*.15
                score += abs(angle-target)*2
                candidates.append((score, recording, start, end))
    if not candidates:
        raise ValueError(f'找不到有真实抬脚的 {np.rad2deg(target):.0f} 度转身')
    chosen = []
    for _, recording, begin, end in sorted(candidates, key=lambda item: item[0]):
        if any(old['source'] == recording['source'] and min(end, b)-max(begin, a) > 8
               for old, a, b in chosen):
            continue
        chosen.append((recording, begin, end))
        if len(chosen) == count:
            break
    return chosen


def walking_variants(source):
    clips = []
    for take in ('02', '04'):
        recording = load_recording(source/f'KIT/167/walking_slow{take}_poses.npz')
        root = recording['root']
        speed = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1)*FPS
        moving = np.flatnonzero(speed > .18)
        if len(moving) < 60:
            raise ValueError(f'低速行走素材过短：{recording["source"]}')
        first, last = int(moving[0]), int(moving[-1])
        spans = [('idle', 5, first-5), ('start', first-5, first+24),
                 ('walk', first+24, last-8), ('stop', last-24, min(len(root)-1, last+14))]
        for kind, begin, end in spans:
            clip = document(recording, begin, end, f'{kind}_kit_{take}', kind)
            clip['family'] = f'kit_{take}'
            clips.append(clip)
    return clips


def build(source, destination):
    walking = source/'ACCAD/Female1Walking_c3d'
    start = load_recording(walking/'B1 - stand to walk_poses.npz')
    stop = load_recording(walking/'B2 - walk to stand_poses.npz')
    walk = load_recording(walking/'B3 - walk1_poses.npz')
    clips = [document(start, 7, 22, 'idle', 'idle'),
             document(start, 22, 60, 'start', 'start'),
             document(walk, 20, 132, 'walk', 'walk'),
             document(stop, 62, 108, 'stop', 'stop')]
    for clip in clips:
        clip['family'] = 'accad_female1'
    clips.extend(walking_variants(source))
    paths = sorted(source.glob('Eyes_Japan_Dataset/*/turn-01*'))
    paths += sorted(source.glob('Eyes_Japan_Dataset/*/turn-02*'))
    paths += sorted((source/'KIT/513').glob('turn*'))
    turns = [load_recording(path) for path in paths]
    for degrees in (30, 60, 90):
        for sign, side in ((1, 'left'), (-1, 'right')):
            for index, (recording, begin, end) in enumerate(turn_candidates(turns, np.deg2rad(degrees*sign))):
                name = f'turn_{side}_{degrees}' if index == 0 else f'turn_{side}_{degrees}_v{index+1}'
                clips.append(document(recording, begin, end, name, 'turn'))
    destination.mkdir(parents=True, exist_ok=True)
    bank = {'version': 1, 'speed': .5, 'clips': clips}
    (destination/'bank.json').write_text(json.dumps(bank, separators=(',', ':')))
    report = [{key: clip[key] for key in ('name', 'source', 'sourceFrames', 'frames', 'angle', 'distance', 'footLift')} for clip in clips]
    (destination/'sources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=PROJECT/'public/locomotion')
    args = parser.parse_args()
    build(args.source, args.output)
