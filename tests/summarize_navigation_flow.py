"""Summarize real browser samples; root positions are evaluation oracles only."""
import collections
import json
import math
import statistics
import sys
from pathlib import Path


def oracle_rooms(samples):
    # Evaluation only; these room bounds never enter the visual controller.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
    from world import get_world
    rooms, visited = get_world()['rooms'], []
    for sample in samples:
        candidates = [room for room in rooms if room['bounds'][0] <= sample['x'] <= room['bounds'][2]
                      and room['bounds'][1] <= sample['z'] <= room['bounds'][3]]
        room = min(candidates, key=lambda item: (item['bounds'][2]-item['bounds'][0])*
                   (item['bounds'][3]-item['bounds'][1]), default={'id': 'outside'})['id']
        if room not in visited:
            visited.append(room)
    return visited


def runs(samples):
    result = []
    for sample in samples:
        if not result or result[-1]['phase'] != sample['phase']:
            result.append({'phase': sample['phase'], 'start': sample['seconds'], 'end': sample['seconds']})
        else:
            result[-1]['end'] = sample['seconds']
    return result


def summary(result):
    samples = [dict(sample) for sample in result['samples']]
    start = 0
    for index in range(1, len(samples)+1):
        first = samples[start]
        if index < len(samples):
            current = samples[index]
            if (current['phase'] == first['phase'] and abs(current['time']-first['time']) < .00001
                    and math.hypot(current['x']-first['x'], current['z']-first['z']) < .00001):
                continue
        if first['phase'] != 'look' and samples[index-1]['seconds']-first['seconds'] >= .35:
            for sample in samples[start:index]:
                sample['phase'] = 'holding'
        start = index
    commands = list(result['commands'].values())
    forward = [command for command in commands if command['arguments'].get('action') == 'forward']
    phases = runs(samples)
    origin = samples[0]
    radius = lambda sample: math.hypot(sample['x']-origin['x'], sample['z']-origin['z'])
    distances = [math.hypot(b['x']-a['x'], b['z']-a['z']) for a, b in zip(samples, samples[1:])]
    forward_meters = sum(distance for index, distance in enumerate(distances)
                         if samples[index]['phase'] in ('start', 'walk') or samples[index+1]['phase'] in ('start', 'walk'))
    walking_stops = sum(phase['phase'] == 'stop' and index > 0 and phases[index-1]['phase'] in ('start', 'walk')
                        for index, phase in enumerate(phases))
    walk_runs, current = [], 0.
    for index, phase in enumerate(phases):
        duration = phase['end']-phase['start']+.1
        if phase['phase'] in ('start', 'walk') or phase['phase'] == 'idle' and duration < .35:
            current += duration
        else:
            walk_runs.append(current)
            current = 0.
    walk_runs.append(current)
    return {'seconds': round(samples[-1]['seconds']-origin['seconds'], 2),
            'oracleRooms': oracle_rooms(samples),
            'maxRadiusMeters': round(max(map(radius, samples)), 3),
            'forwardSampledMeters': round(forward_meters, 3), 'walkingStops': walking_stops,
            'walkingStopsPerMeter': round(walking_stops/max(.001, forward_meters), 3),
            'longestWalkSeconds': round(max(walk_runs), 2),
            'phaseSeconds': {key: round(value/10, 1) for key, value in collections.Counter(sample['phase'] for sample in samples).items()},
            'forwardCommands': len(forward), 'turnCommands': sum(str(command['arguments'].get('action')).startswith('turn') for command in commands),
            'medianForwardPlanMeters': round(statistics.median(command['arguments']['distance'] for command in forward), 3) if forward else 0,
            'continuousPlans': sum(command['arguments'].get('continuous', False) for command in forward),
            'headLookCommands': sum(command['arguments'].get('action') == 'look' for command in commands),
            'prefetchedHandoffs': sum(bool(command['arguments'].get('continuationOf')) for command in commands),
            'failedCommands': [{'action': command['arguments'].get('action'), 'detail': command.get('detail')} for command in commands if command['status'] == 'failed'],
            'trackingRatio': round(sum(sample['tracking'] for sample in samples)/len(samples), 3),
            'goalStatus': result['final']['goal']['status'], 'errors': result['errors']}


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        path = Path(filename)
        result = summary(json.loads(path.read_text()))
        path.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(path.name, json.dumps(result, ensure_ascii=False))
