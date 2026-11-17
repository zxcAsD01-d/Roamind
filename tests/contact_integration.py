"""Real RGB/stereo -> generic contact command -> ARDY -> VRM playback and tracking."""
import argparse
import json
import time
from pathlib import Path

from touch_integration import request, select_target

BASE = 'http://127.0.0.1:8100'


def exercise(client, part, effector, duration):
    session = next(s for s in request(BASE, '/api/sim/state')['sessions'] if s['id'] == client)
    target = select_target(session['state']['perceptionClientId'], part)
    previous = {j['id'] for j in request(BASE, '/api/contact/jobs')['jobs']}
    command = request(BASE, '/api/sim/contact', {'clientId': client, 'selection': {'targetId': target['targetId']},
                      'effector': effector, 'operation': 'hold', 'duration': duration})
    started = time.monotonic(); samples, states, job = [], [], None
    while time.monotonic()-started < 95:
        status = request(BASE, '/api/sim/commands/'+command['id'])
        if not states or states[-1] != status['status']:
            states.append(status['status'])
        if status['status'] in ('completed', 'failed', 'interrupted', 'superseded'):
            return {'target': target, 'command': status, 'statuses': states, 'job': job, 'samples': samples,
                    'elapsedSeconds': time.monotonic()-started, 'effector': effector}
        jobs = request(BASE, '/api/contact/jobs')['jobs']
        new = [j for j in jobs if j['id'] not in previous]
        if new and new[-1]['status'] in ('generating', 'ready'):
            job = new[-1]
            try:
                observation = request(BASE, '/api/contact/track', {'id': job['id']})
                if not samples or observation['frameId'] != samples[-1]['frameId']:
                    samples.append({key: observation[key] for key in ('position', 'measurementPosition',
                        'frameId', 'trackingSamples', 'sigmaMeters', 'observedAt')})
            except RuntimeError as exc:
                # Keep failures visible; the browser must independently report final execution status.
                samples.append({'error': str(exc), 'frameId': -1})
        time.sleep(.15)
    raise TimeoutError('通用接触命令95秒未结束')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--part', default='left_wrist')
    parser.add_argument('--effector', default='right_hand')
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--require-motion', action='store_true')
    parser.add_argument('--expect-failure', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/contact/live-integration.json'))
    args = parser.parse_args()
    result = exercise(args.client_id, args.part, args.effector, args.duration)
    points = [s['measurementPosition'] for s in result['samples'] if 'measurementPosition' in s]
    distance = max((sum((a-b)**2 for a, b in zip(point, points[0]))**.5 for point in points), default=0)
    result['measuredDisplacementMeters'] = distance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result['command']['status'], result['command'].get('detail'), 'observed motion:', distance)
    expected = 'failed' if args.expect_failure else 'completed'
    if result['command']['status'] != expected or args.require_motion and distance <= .06:
        raise SystemExit('通用接触实测未达到要求，请查看输出记录')
