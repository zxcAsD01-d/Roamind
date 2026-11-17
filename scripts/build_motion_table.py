"""Build a versioned native motion table. No root warping or contact rewriting."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from amass_search import quality
from build_locomotion import PROJECT, DEFAULT_DATA, document, load_recording
from motion_table_features import JOINTS, costs, describe


def role(clip):
    return {'walk': 'travel', 'trajectory': 'travel'}.get(clip['kind'], clip['kind'])


def windows(recording):
    size = len(recording['root'])
    for count in (24, 40):
        for start in range(0, size-count, 16):
            yield start, start+count


def source_paths(source):
    groups = defaultdict(list)
    for path in sorted(source.glob('KIT/*/*Circle*_poses.npz')):
        if 'WalkIn' in path.name:
            groups[path.parent.name].append(path)
    for path in sorted(source.glob('ACCAD/*Walking*/*turn*_poses.npz')):
        groups[path.parent.name].append(path)
    return [paths[index] for index in range(max(map(len, groups.values()), default=0))
            for paths in groups.values() if index < len(paths)]


def checked(clip):
    report = quality(clip)
    if report['supportMean'] > .14 or report['supportP95'] > .42:
        raise ValueError('strict_support_sliding')
    if report['angularStepMax'] > .45 or report['rootSpeedMax'] > 1.8:
        raise ValueError('strict_discontinuity')
    return report


def add_clip(clip, entries, motions, buckets, rejected, limit):
    try:
        report, features = checked(clip), describe(clip)
        category = role(clip)
        semantic = clip.get('prompt', clip['name']) if category == 'action' else ''
        bucket = (category, semantic, round(features['yaw']/.12), round(features['length']/.3))
        if category == 'travel' and (features['length'] < .15 or abs(features['yaw']) > 1.2):
            raise ValueError('travel_window_out_of_range')
        if buckets[bucket] >= limit:
            return
        identity = hashlib.sha256((clip['source']+str(clip.get('sourceFrames'))+clip['name']).encode()).hexdigest()[:16]
        if any(entry['id'] == identity for entry in entries):
            return
        entries.append({'id': identity, 'role': category, 'name': clip['name'], 'source': clip['source'],
                        'sourceFrames': clip.get('sourceFrames'), 'prompt': clip.get('prompt', ''), 'fps': clip['fps'], 'frames': clip['frames'],
                        'duration': (clip['frames']-1)/clip['fps'], 'sourceHeight': clip.get('sourceHeight', .92),
                        'heightOffset': clip.get('heightOffset', 0), 'quality': report, **features})
        motions.append(clip)
        buckets[bucket] += 1
    except (ValueError, KeyError) as error:
        rejected.append({'source': clip.get('source'), 'frames': clip.get('sourceFrames'), 'reason': str(error)})


def collect(source, bank, generated, maximum):
    entries, motions, rejected, buckets = [], [], [], Counter()
    for clip in json.loads(bank.read_text())['clips']:
        add_clip(clip, entries, motions, buckets, rejected, 32)
    scanned = []
    for path in source_paths(source)[:maximum]:
        scanned.append(str(path.relative_to(source)))
        try:
            recording = load_recording(path)
            for begin, end in windows(recording):
                clip = document(recording, begin, end, f'{path.stem}_{begin}_{end}', 'trajectory')
                add_clip(clip, entries, motions, buckets, rejected, 12)
        except (OSError, ValueError, KeyError) as error:
            rejected.append({'source': str(path), 'reason': str(error)})
    for path in sorted(generated.glob('*.json')) if generated else []:
        try:
            clip = json.loads(path.read_text())
            if 'localRot' in clip:
                add_clip(clip, entries, motions, buckets, rejected, 32)
        except (OSError, ValueError, KeyError) as error:
            rejected.append({'source': str(path), 'reason': str(error)})
    return entries, motions, rejected, scanned


def graph(entries):
    starts = np.asarray([entry['entry'] for entry in entries])
    edges = []
    for index, entry in enumerate(entries):
        scores, rms, maximum = costs(entry['exit'], starts)
        groups = defaultdict(list)
        for target in np.argsort(scores):
            other = entries[target]
            if rms[target] > .45 or maximum[target] > 1.2 or scores[target] > .65:
                continue
            key = (other['role'], int(np.sign(other['yaw'])) if abs(other['yaw']) > .06 else 0)
            if len(groups[key]) < 3:
                groups[key].append({'to': int(target), 'cost': round(float(scores[target]), 6),
                                    'poseRms': round(float(rms[target]), 6)})
        edges.append([edge for values in groups.values() for edge in values])
    return edges


def publish(output, entries, motions, rejected, scanned):
    packed = []
    for entry, clip in zip(entries, motions):
        entry['offset'] = len(packed)
        packed.extend(clip['localRot'])
        packed.extend(clip['rootDelta'])
        packed.extend(clip['footContacts'])
    binary = np.asarray(packed, dtype='<f4').tobytes()
    digest = hashlib.sha256(binary).hexdigest()
    data_file = f'poses-{digest[:16]}.bin'
    table = {'version': 1, 'fps': 20, 'up': 'yup', 'joints': JOINTS,
             'bones': motions[0]['bones'], 'parents': motions[0]['parents'],
             'restJoints': motions[0]['restJoints'], 'dataFile': data_file, 'dataSha256': digest,
             'dataFloats': len(packed), 'entries': entries, 'successors': graph(entries),
             'policy': 'native-root-SE2; full-body-pose-velocity-contact; bounded-beam; avatar-quality-gated'}
    output.mkdir(parents=True, exist_ok=True)
    (output/data_file).write_bytes(binary)
    temporary = output/'table.json.tmp'
    temporary.write_text(json.dumps(table, separators=(',', ':'), allow_nan=False))
    temporary.replace(output/'table.json')
    report = {'accepted': len(entries), 'roles': dict(Counter(entry['role'] for entry in entries)),
              'sources': len(set(entry['source'] for entry in entries)), 'scanned': scanned, 'rejected': rejected,
              'bytes': len(binary), 'edges': sum(map(len, table['successors'])),
              'qualityGate': {'supportMean': .14, 'supportP95': .42, 'angularStepMax': .45},
              'license': 'AMASS/original source terms apply; generated candidates are never trusted without screening'}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--bank', type=Path, default=PROJECT/'public/locomotion/bank.json')
    parser.add_argument('--generated', type=Path, default=PROJECT/'.local-data/table-candidates')
    parser.add_argument('--output', type=Path, default=PROJECT/'public/motion-table')
    parser.add_argument('--max-recordings', type=int, default=120)
    args = parser.parse_args()
    entries, motions, rejected, scanned = collect(args.source, args.bank, args.generated, args.max_recordings)
    if not entries or not {'travel', 'start', 'stop'}.issubset({entry['role'] for entry in entries}):
        raise ValueError('动作表缺少基本移动能力，未发布')
    report = publish(args.output, entries, motions, rejected, scanned)
    print(json.dumps({key: report[key] for key in ('accepted', 'roles', 'sources', 'bytes', 'edges')}))


if __name__ == '__main__':
    main()
