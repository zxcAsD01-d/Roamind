"""Collect native ARDY candidates; every window must pass the motion-table gates."""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from build_locomotion import BONES, document
from build_motion_table import checked
from motion_table_features import arrays


def generate(base, prompt, seconds):
    body = {'clientId': 'table-'+str(uuid.uuid4()), 'revision': 1,
            'prompt': prompt, 'duration': seconds, 'history': None}
    request = urllib.request.Request(base+'/api/generate/stream', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    entry = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if event['type'] == 'error':
                raise ValueError(event.get('error', 'ARDY generation failed'))
            if event['type'] == 'complete':
                entry = event['entry']
    if not entry or not entry['url'].startswith('/files/'):
        raise ValueError('ARDY did not publish a complete local clip')
    with urllib.request.urlopen(base+entry['url'], timeout=30) as response:
        return json.load(response), entry


def candidates(motion, identity, category):
    if motion.get('bones') != BONES or motion.get('fps') != 20 or motion.get('up') != 'yup':
        raise ValueError('ARDY skeleton incompatible')
    local = np.asarray(motion['localRot']).reshape(-1, 27, 3)
    root = np.asarray(motion['rootDelta']).reshape(-1, 3)
    root += np.asarray(motion.get('rootOrigin', [0, .92, 0]))
    recording = {'local': local, 'root': root, 'source': 'ARDY/'+identity}
    if not np.isfinite(local).all() or not np.isfinite(root).all() or len(root) != len(local):
        raise ValueError('ARDY invalid coordinates')
    if category == 'action':
        clip = document(recording, 0, len(root)-1, identity+'_full', category)
        clip['dataset'] = 'ARDY screened'; clip['prompt'] = motion.get('prompt', '')
        arrays(clip)
        yield clip
    for begin in range(10, len(root)-24, 16):
        clip = document(recording, begin, begin+24, identity+'_'+str(begin), category)
        clip['dataset'] = 'ARDY screened'; clip['prompt'] = motion.get('prompt', '')
        arrays(clip)
        yield clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    parser.add_argument('--output', type=Path, default=Path('.local-data/table-candidates'))
    parser.add_argument('--jobs', type=Path, help='JSON list of {prompt, category: trajectory|action}')
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--duration', type=float, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [('A person walks forward.', 'trajectory'), ('A person walks in a circle.', 'trajectory'),
            ('A person waves their hand.', 'action'), ('A person points forward with their right arm.', 'action'),
            ('A person raises their right arm.', 'action'), ('A person touches something with their right hand.', 'action')]
    if args.jobs:
        rows = json.loads(args.jobs.read_text())
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
                or not isinstance(row.get('prompt'), str) or row.get('category') not in ('trajectory', 'action') for row in rows):
            raise ValueError('jobs must contain prompt and trajectory|action category')
        jobs = [(row['prompt'], row['category']) for row in rows]
    if not 1 <= args.variants <= 100 or not 2 <= args.duration <= 60:
        raise ValueError('variants must be 1–100 and duration must be 2–60 seconds')
    with urllib.request.urlopen(args.base+'/api/prompts', timeout=30) as response:
        cached = set(json.load(response)['prompts'])
    if any(prompt not in cached for prompt, _ in jobs):
        raise ValueError('Only locally cached ARDY prompts can be collected')
    report = []
    for prompt, category in jobs:
        for variant in range(args.variants):
            started = time.monotonic()
            motion, entry = generate(args.base, prompt, args.duration)
            identity = Path(entry['url']).stem
            for clip in candidates(motion, identity, category):
                row = {'prompt': prompt, 'variant': variant, 'name': clip['name']}
                try:
                    row['quality'] = checked(clip)
                    (args.output/(clip['name']+'.json')).write_text(json.dumps(clip, allow_nan=False))
                    row['accepted'] = True
                except ValueError as error:
                    row.update(accepted=False, reason=str(error))
                report.append(row)
            print(json.dumps({'prompt': prompt, 'variant': variant, 'seconds': time.monotonic()-started,
                              'accepted': sum(row.get('accepted', False) for row in report)}, ensure_ascii=False), flush=True)
    (args.output/('collection-report-'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
