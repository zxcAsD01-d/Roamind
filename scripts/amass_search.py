"""Curate local AMASS locomotion without synthesizing or modifying recorded motion.

Reject malformed, discontinuous, slipping and redundant clips. Persist frame
features and bounded successor edges for runtime motion/goal matching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_locomotion import (PROJECT, DEFAULT_DATA, FPS, document,
                              geometry, heading, load_recording)

JOINTS = [19, 20, 21, 23, 24, 25]


def quality(clip):
    local = np.asarray(clip['localRot']).reshape(-1, 27, 3)
    root = np.asarray(clip['rootDelta']).reshape(-1, 3)
    if len(root) < 12 or not np.isfinite(local).all() or not np.isfinite(root).all():
        raise ValueError('short_or_nonfinite')
    feet = geometry({'local': local, 'root': root})
    rotations = Rotation.from_rotvec(local.reshape(-1, 3)).as_quat().reshape(-1, 27, 4)
    dots = np.abs((rotations[1:]*rotations[:-1]).sum(axis=2))
    angular = 2*np.arccos(np.clip(dots, 0, 1))
    speed = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1)*FPS
    floor = np.percentile(feet[:, :, 1], 15, axis=0)
    support = np.maximum(feet[:-1, :, 1], feet[1:, :, 1]) < floor+.025
    support &= np.abs(np.diff(feet[:, :, 1], axis=0))*FPS < .12
    slip = (np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2)*FPS)[support]
    report = {'angularStepMax': float(angular.max()), 'rootSpeedMax': float(speed.max()),
              'supportSamples': int(len(slip)), 'supportMean': float(slip.mean()) if len(slip) else None,
              'supportP95': float(np.percentile(slip, 95)) if len(slip) else None,
              'heightRange': float(np.ptp(root[:, 1]))}
    if report['angularStepMax'] > .65 or report['rootSpeedMax'] > 2.5:
        raise ValueError('discontinuous')
    if report['supportSamples'] < 4:
        raise ValueError('insufficient_support')
    if report['supportMean'] > .22 or report['supportP95'] > .65:
        raise ValueError('support_sliding')
    if report['heightRange'] > .22:
        raise ValueError('root_height')
    if clip['kind'] == 'walk':
        direction = root[-1, [0, 2]]
        distance = np.linalg.norm(direction)
        straightness = distance/max(.001, clip['distance'])
        angle = abs(np.arctan2(direction[0], direction[1]))
        if distance < .45 or straightness < .94 or angle > .32 or abs(clip['angle']) > .4:
            raise ValueError('non_forward_walk')
    if clip['kind'] == 'idle' and (clip['distance'] > .10 or abs(clip['angle']) > .12):
        raise ValueError('moving_idle')
    return report


def frame_features(clip):
    local = np.asarray(clip['localRot']).reshape(-1, 27, 3)
    root = np.asarray(clip['rootDelta']).reshape(-1, 3)
    yaw = heading(local)
    velocity = np.gradient(root, axis=0)*FPS
    cosine, sine = np.cos(yaw), np.sin(yaw)
    velocity = np.column_stack((cosine*velocity[:, 0]-sine*velocity[:, 2],
                                sine*velocity[:, 0]+cosine*velocity[:, 2]))
    quats = Rotation.from_rotvec(local[:, JOINTS].reshape(-1, 3)).as_quat().reshape(-1, 6, 4)
    angular = (Rotation.from_rotvec(local[:-1, JOINTS].reshape(-1, 3)).inv()
               *Rotation.from_rotvec(local[1:, JOINTS].reshape(-1, 3))).as_rotvec().reshape(-1, 18)*FPS
    angular = np.vstack((angular[:1], angular))
    contact = np.asarray(clip['footContacts']).reshape(-1, 4)
    support = np.column_stack((contact[:, :2].max(axis=1), contact[:, 2:].max(axis=1)))
    return np.column_stack((velocity, support, quats.reshape(-1, 24), angular)).round(5)


def sample_windows(recording):
    speed = np.linalg.norm(np.diff(recording['root'][:, [0, 2]], axis=0), axis=1)*FPS
    moving = np.flatnonzero(speed > .18)
    if len(moving) < 60:
        return []
    first, last = int(moving[0]), int(moving[-1])
    spans = [('idle', max(0, first-55), first-6),
             ('start', max(0, first-6), first+24),
             ('walk', first+24, min(last-8, first+144)),
             ('stop', last-24, min(len(speed), last+16))]
    return [(kind, begin, end) for kind, begin, end in spans if end-begin >= 12]


def recording_paths(source):
    # Interleave subjects, rather than filling the library from the first actor.
    groups = {}
    for path in sorted(source.glob('KIT/*/walking_slow*_poses.npz')):
        groups.setdefault(path.parent.name, []).append(path)
    maximum = max(map(len, groups.values()), default=0)
    return [paths[index] for index in range(maximum) for paths in groups.values() if index < len(paths)]


def duplicate(clip, clips):
    if any(item['source'] == clip['source'] and item.get('sourceFrames') == clip.get('sourceFrames')
           and item['kind'] == clip['kind'] for item in clips):
        return True
    if clip['kind'] not in ('walk', 'idle'):
        return False
    features = frame_features(clip)
    signature = features[np.linspace(0, len(features)-1, 12).astype(int), 4:28].reshape(-1, 6, 4)
    for item in clips:
        if item['kind'] != clip['kind']:
            continue
        old = frame_features(item)
        old = old[np.linspace(0, len(old)-1, 12).astype(int), 4:28].reshape(-1, 6, 4)
        angles = 2*np.arccos(np.clip(np.abs((old*signature).sum(axis=2)), 0, 1))
        if float(np.sqrt(np.mean(angles**2))) < .045:
            return True
    return False


def curate(source, clips):
    accepted, rejected, scanned = list(clips), [], []
    quotas = {'idle': 10, 'start': 14, 'walk': 14, 'stop': 14}
    for path in recording_paths(source):
        identity = path.relative_to(source).as_posix()
        scanned.append(identity)
        try:
            recording = load_recording(path)
            for kind, begin, end in sample_windows(recording):
                if sum(item['kind'] == kind for item in accepted) >= quotas[kind]:
                    continue
                name = kind+'_'+hashlib.sha256(f'{identity}:{begin}:{end}'.encode()).hexdigest()[:10]
                clip = document(recording, begin, end, name, kind)
                clip['family'] = str(path.relative_to(source).with_suffix(''))
                try:
                    clip['quality'] = quality(clip)
                    if duplicate(clip, accepted):
                        raise ValueError('duplicate')
                    accepted.append(clip)
                except ValueError as error:
                    rejected.append({'source': identity, 'kind': kind, 'frames': [begin, end], 'reason': str(error)})
        except (ValueError, KeyError, OSError) as error:
            rejected.append({'source': identity, 'reason': str(error)})
    return accepted, rejected, scanned


def transition_cost(tail, entries):
    q = entries[:, 4:28].reshape(-1, 6, 4)
    angle = 2*np.arccos(np.clip(np.abs((q*tail[4:28].reshape(6, 4)).sum(axis=2)), 0, 1))
    pose = np.mean(angle**2, axis=1)
    speed = np.sum((entries[:, :2]-tail[:2])**2, axis=1)
    contact = np.mean(np.abs(entries[:, 2:4]-tail[2:4]), axis=1)
    velocity = np.mean((entries[:, 28:]-tail[28:])**2, axis=1)
    return pose+.3*speed+.15*contact+.015*velocity


def index_clips(clips):
    records, summary = {}, []
    for clip in clips:
        features = frame_features(clip)
        entries = list(range(0, max(1, clip['frames']-8), 2))
        clip['search'] = {'version': 1, 'entries': [[frame, *features[frame].tolist()] for frame in entries]}
        records[clip['name']] = (clip, features, entries)
    edges = []
    for name, (clip, features, _) in records.items():
        candidates = []
        for other, (target, entry, frames) in records.items():
            if clip['kind'] == 'walk' and target['kind'] == 'start':
                continue
            costs = transition_cost(features[-1], entry[frames])
            selected = np.argsort(costs)[:2]
            candidates.extend({'to': other, 'frame': frames[int(index)], 'cost': round(float(costs[index]), 5)}
                              for index in selected if costs[index] <= .65)
        candidates.sort(key=lambda item: item['cost'])
        for kind in ('idle', 'start', 'walk', 'stop', 'turn'):
            edges.extend({'from': name, **edge} for edge in
                         [item for item in candidates if records[item['to']][0]['kind'] == kind][:4])
        summary.append({key: clip[key] for key in
                        ('name', 'source', 'sourceFrames', 'kind', 'frames', 'duration', 'quality')})
    return {'version': 1, 'featureLayout': 'velocityXZ,contactLR,legQuaternions6x4,legAngularVelocity6x3',
            'featureJoints': JOINTS, 'clips': summary, 'transitions': edges,
            'policy': 'shortlist then live boundary/goal score; avatar support and collision gates remain mandatory'}


def build(source, base, output):
    if not source.is_dir():
        raise FileNotFoundError(f'AMASS 数据目录不存在：{source}')
    bank = json.loads(base.read_text())
    clips, rejected = [], []
    for clip in bank['clips']:
        try:
            clip['quality'] = quality(clip)
            clips.append(clip)
        except ValueError as error:
            rejected.append({'name': clip['name'], 'source': clip['source'], 'reason': str(error)})
    clips, additions_rejected, scanned = curate(source, clips)
    required = {'idle', 'start', 'walk', 'stop', 'turn'}
    if not required.issubset({clip['kind'] for clip in clips}):
        raise ValueError('清洗后缺少必要动作类别，未写入运行库')
    index = index_clips(clips)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({'version': 1, 'speed': .5, 'searchVersion': 1, 'clips': clips}, separators=(',', ':'))
    index['bankSha256'] = hashlib.sha256(payload.encode()).hexdigest()
    report = {'scannedRecordings': len(scanned), 'scannedSources': scanned, 'accepted': len(clips),
              'kinds': {kind: sum(clip['kind'] == kind for clip in clips) for kind in sorted(required)},
              'rejected': rejected+additions_rejected, 'source': str(source),
              'scope': 'local low-speed locomotion subset; not all 44GB; no rewritten joint/root samples',
              'sourceLicense': 'AMASS and original contributing dataset licenses apply; local use only'}
    (output/'bank.json').write_text(payload)
    (output/'index.json').write_text(json.dumps(index, separators=(',', ':')))
    (output/'cleaning-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ('scannedRecordings', 'accepted', 'kinds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--base', type=Path, default=PROJECT/'public/locomotion/bank.json')
    parser.add_argument('--output', type=Path, default=PROJECT/'public/locomotion')
    args = parser.parse_args()
    build(args.source, args.base, args.output)
