"""Observe the actual browser mission; simulation positions are test oracles only."""
import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:8100'+path, payload, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def oracle_room(position, rooms):
    x, _, z = position
    matches = [room for room in rooms if room['bounds'][0] <= x <= room['bounds'][2]
               and room['bounds'][1] <= z <= room['bounds'][3]]
    return min(matches, key=lambda room: (room['bounds'][2]-room['bounds'][0])*
               (room['bounds'][3]-room['bounds'][1]), default={'id': 'outside'})['id']


def monitor(client, seconds, output):
    samples, started = [], time.monotonic()
    rooms = request('/api/world')['rooms']  # Evaluation oracle; never sent to the policy.
    while time.monotonic()-started < seconds:
        world = request('/api/sim/state')
        session = next((item for item in world['sessions'] if item['id'] == client), None)
        if session is None:
            raise ValueError('测试网页不在线')
        goal_state = request('/api/goals/state', {'clientId': client})
        state, goal = goal_state['behavior'], goal_state['goal']
        samples.append({'seconds': round(time.monotonic()-started, 2),
                        'position': session['state']['position'], 'heading': session['state']['heading'],
                        'playing': session['state']['playing'], 'mission': state['mission'],
                        'memory': state['spatialMemory'], 'decisions': state['decisions'],
                        'feedback': state['feedback'], 'fps': session['state'].get('fps'),
                        'perception': session['state'].get('perception'),
                        'oracleRoom': oracle_room(session['state']['position'], rooms),
                        'goal': goal,
                        'commands': [item for item in world['commands']
                                     if item['kind'] == 'visual' and item['clientId'] == client]})
        if goal and goal['status'] == 'failed':
            return {'completed': False, 'failed': True, 'reason': goal['detail'], 'samples': samples}
        if len(samples) % 30 == 0:
            progress = {'completed': False, 'samples': samples, 'summary': summarize(samples)}
            output.write_text(json.dumps(progress, ensure_ascii=False))
            print(json.dumps(progress['summary'], ensure_ascii=False), flush=True)
        if state['mission']['ready']:
            return {'completed': True, 'samples': samples}
        time.sleep(1)
    return {'completed': False, 'samples': samples, 'reason': '观察时间内未达成社交条件'}


def summarize(samples):
    first, last = samples[0], samples[-1]
    commands = {item['id']: item for sample in samples for item in sample['commands']}
    commands = [item for item in commands.values() if item['created'] >= first.get('startedAt', 0)]
    positions = [[sample['position'][0], sample['position'][2]] for sample in samples]
    actions = [next((item['arguments']['action'] for item in reversed(sample['commands'])
                     if item['status'] in ('playing', 'received', 'queued')), None) for sample in samples]
    forward_meters = sum(math.dist(a, b) for index, (a, b) in enumerate(zip(positions, positions[1:]))
                         if 'forward' in actions[index:index+2])
    regions = {tuple(math.floor(value/.7) for value in point) for point in positions}
    perceptions = [sample['perception'] for sample in samples if sample.get('perception')]
    fps = [sample['fps'] for sample in samples if sample.get('fps')]
    elapsed = max(.001, last['seconds']-first['seconds'])
    phases = {sample['mission']['phase'] for sample in samples}
    seen, last_new, longest_idle = set(), first['seconds'], 0.
    for sample, position in zip(samples, positions):
        key = tuple(math.floor(value/.7) for value in position)
        if key not in seen:
            longest_idle = max(longest_idle, sample['seconds']-last_new)
            seen.add(key)
            last_new = sample['seconds']
    longest_idle = max(longest_idle, last['seconds']-last_new)
    return {'seconds': round(elapsed, 2), 'netMeters': round(math.dist(positions[0], positions[-1]), 3),
            'maxRadiusMeters': round(max(math.dist(positions[0], point) for point in positions), 3),
            'longestNoNewRegionSeconds': round(longest_idle, 2),
            'oracleRooms': list(dict.fromkeys(s.get('oracleRoom', 'unrecorded') for s in samples)),
            'sampledPathMeters': round(sum(math.dist(a, b) for a, b in zip(positions, positions[1:])), 3),
            'forwardSampledMeters': round(forward_meters, 3),
            'oracleVisitedRegions': len(regions), 'oracleRegionSizeMeters': .7,
            'visualVisitedRegions': last['memory']['visitedRegions'],
            'newVisualRegionsPerMinute': round((last['memory']['visitedRegions']-first['memory']['visitedRegions'])*60/elapsed, 2),
            'decisions': last['decisions']-first['decisions'], 'commands': len(commands),
            'forwardCommands': sum(item['arguments']['action'] == 'forward' for item in commands),
            'completedCommands': sum(item['status'] == 'completed' for item in commands),
            'interruptedCommands': sum(item['status'] == 'interrupted' for item in commands),
            'failedCommands': sum(item['status'] == 'failed' for item in commands),
            'phases': sorted(phases), 'medianFps': round(statistics.median(fps), 1) if fps else None,
            'trackingSampleRatio': round(sum(p['mapping'] for p in perceptions)/len(perceptions), 3) if perceptions else None,
            'stereoP95Ms': percentile([p['inferenceMs'] for p in perceptions], .95),
            'lastMission': last['mission'], 'planner': last['memory'].get('planner')}


def percentile(values, fraction):
    return round(sorted(values)[min(len(values)-1, int(len(values)*fraction))], 2) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--require-ready', action='store_true')
    parser.add_argument('--min-regions', type=int, default=0)
    parser.add_argument('--min-forward', type=int, default=0)
    parser.add_argument('--min-rooms', type=int, default=0)
    parser.add_argument('--min-radius', type=float, default=0)
    parser.add_argument('--max-idle', type=float, default=float('inf'))
    parser.add_argument('--min-tracking', type=float, default=0)
    parser.add_argument('--start-goal', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/social-navigation/integration.json'))
    args = parser.parse_args()
    if args.start_goal:
        request('/api/goals/submit', {'clientId': args.client_id, 'title': '跨区域自主探索验收',
                                     'steps': [{'kind': 'social', 'timeout': min(600, args.seconds+30)}]})
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = monitor(args.client_id, args.seconds, args.output)
    if args.start_goal and not result['completed'] and not result.get('failed'):
        cleanup = request('/api/goals/cancel', {'clientId': args.client_id})
        result['cleanup'] = cleanup['goal']['status']
    result['samples'][0]['startedAt'] = started
    result['summary'] = summarize(result['samples'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'completed': result['completed'], **result['summary']}, ensure_ascii=False))
    if result.get('failed'):
        raise SystemExit(result['reason'])
    if args.require_ready and not result['completed']:
        raise SystemExit(result['reason'])
    if result['summary']['oracleVisitedRegions'] < args.min_regions:
        raise SystemExit('实际探索覆盖不足')
    if result['summary']['forwardCommands'] < args.min_forward:
        raise SystemExit('有效前进指令不足，不能把转向摆动计为探索')
    summary = result['summary']
    if (len(summary['oracleRooms']) < args.min_rooms or summary['maxRadiusMeters'] < args.min_radius
            or summary['longestNoNewRegionSeconds'] > args.max_idle
            or (summary['trackingSampleRatio'] or 0) < args.min_tracking):
        raise SystemExit('长程探索覆盖、连续进展或视觉定位稳定性未通过')


if __name__ == '__main__':
    main()
