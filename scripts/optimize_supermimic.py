"""Calibrate humanoid clips to G1, track physically and reconstruct original bone lengths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_optimizer import optimize_motion
from supermimic_policy import SuperMimicPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', type=Path, help='Existing Y-up motion JSON with restJoints')
    source.add_argument('--bank-names', nargs='+', default=['walk_kit_02', 'start_kit_02', 'turn_left_30', 'idle'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output/supermimic/human-bridge')
    parser.add_argument('--backend', choices=('torch', 'mlx'), default='torch')
    parser.add_argument('--avatar', type=Path, help='Measured VRM rest geometry and per-clip root scales')
    parser.add_argument('--root-scale', type=float, help='Explicit source-to-avatar root scale')
    args = parser.parse_args()
    if args.input:
        clips = [json.loads(args.input.read_text())]
    else:
        bank = json.loads((ROOT / 'public/locomotion/bank.json').read_text())['clips']
        missing = set(args.bank_names) - {clip['name'] for clip in bank}
        if missing:
            parser.error('Unknown bank clips: ' + ', '.join(sorted(missing)))
        clips = [clip for clip in bank if clip['name'] in args.bank_names]
    args.output.mkdir(parents=True, exist_ok=True)
    avatar = json.loads(args.avatar.read_text()) if args.avatar else None
    policy, reports = SuperMimicPolicy(backend=args.backend), []
    for index, clip in enumerate(clips):
        result = optimize_motion(clip, policy, avatar, args.root_scale)
        prefix = f'clip-{index:03}'
        for label, motion in [('source', clip), ('output', result['motion']), ('candidate', result['candidate'])]:
            path = args.output / f'{prefix}-{label}.json'
            if motion is not None:
                path.write_text(json.dumps(motion, ensure_ascii=False, allow_nan=False))
            elif path.exists():
                path.unlink()
        report = {'id': prefix, 'name': clip['name'], **result['report']}
        reports.append(report)
        (args.output / 'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
