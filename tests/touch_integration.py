"""Exercise actual YOLO/stereo -> command -> resident ARDY -> browser playback."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base+path, payload, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def select_target(visual, part):
    base = 'http://127.0.0.1:8200'
    frame = request(base, '/api/stereo/frame?clientId='+visual)
    if frame.get('stale') or not frame.get('people'):
        raise ValueError('网页需要保持双目视觉，且所选人物当前可见')
    person = max(frame['people'], key=lambda p: p['confidence'])
    return request(base, '/api/touch/resolve', {'clientId': visual, 'personId': person['id'], 'bodyPart': part})


def await_command(identity):
    deadline = time.monotonic()+100
    while time.monotonic() < deadline:
        command = request('http://127.0.0.1:8100', '/api/sim/commands/'+identity)
        if command['status'] in ('completed', 'failed', 'interrupted', 'superseded'):
            return command
        time.sleep(.25)
    raise TimeoutError('触摸命令 100 秒未结束')


def exercise(client, visual, hand, part, operation):
    target = select_target(visual, part)
    body = {'clientId': client, 'selection': {'targetId': target['targetId']}, 'hand': hand,
            'operation': operation, 'duration': .6, 'amplitude': min(.005, target['localSurfaceRadius']/2)}
    started = time.perf_counter()
    command = request('http://127.0.0.1:8100', '/api/sim/touch', body)
    result = await_command(command['id'])
    jobs = request('http://127.0.0.1:8100', '/api/touch/jobs')['jobs']
    job = request('http://127.0.0.1:8100', '/api/touch/jobs/'+jobs[-1]['id']) if jobs else None
    summary = {'hand': hand, 'part': part, 'operation': operation, 'command': result, 'target': target,
               'elapsedSeconds': time.perf_counter()-started,
               'generation': None if job is None else {k: job[k] for k in ('id', 'status', 'metrics')}}
    print(hand, operation, result['status'], result.get('detail', ''), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--all-operations', action='store_true')
    parser.add_argument('--hand', choices=('left', 'right'), default='right')
    parser.add_argument('--part', default='left_wrist')
    parser.add_argument('--output', default='output/touch/integration.json')
    args = parser.parse_args()
    state = request('http://127.0.0.1:8100', '/api/sim/state')
    session = next(s for s in state['sessions'] if s['id'] == args.client_id)
    visual = session['state']['perceptionClientId']
    operations = ('touch', 'tap', 'hold', 'stroke') if args.all_operations else ('touch',)
    results = [exercise(args.client_id, visual, args.hand, args.part, operation) for operation in operations]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'clientId': args.client_id, 'results': results}, ensure_ascii=False, indent=2))
    if any(result['command']['status'] != 'completed' for result in results):
        raise SystemExit('存在未完成的触摸命令，见报告')


if __name__ == '__main__':
    main()
