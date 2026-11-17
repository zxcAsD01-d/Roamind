"""Summarize actual browser social execution, with optional fixture oracles."""
import argparse
import json
import math
from pathlib import Path


def summarize(probe):
    samples = probe['samples']
    commands = list(probe['commands'].values())
    forwards = [item for item in commands if item['arguments'].get('action') == 'forward']
    last = samples[-1]['state']
    final = probe.get('final', last)
    goal = probe.get('cleanup', {}).get('goal') or final['goal']
    greetings = {step['greetingCommand'] for step in goal['steps'] if step.get('greetingCommand')}
    waves = [item for item in commands if item['id'] in greetings]
    ready = [item['state']['behavior']['mission'] for item in samples if item['state']['behavior']['mission']['ready']]
    people = [item['partner'] for item in samples if item.get('partner')]
    positions = [[float(item['root']['rootX']), float(item['root']['rootZ'])] for item in samples]
    result = {'seconds': round(samples[-1]['seconds'], 2), 'done': probe['done'],
              'goalStatus': goal['status'], 'goalDetail': goal['detail'],
              'lastExecutionDetail': last['goal']['detail'],
              'forwardCommands': len(forwards), 'completedForward': sum(item['status'] == 'completed' for item in forwards),
              'waveCommands': len(waves), 'completedWaves': sum(item['status'] == 'completed' for item in waves),
              'plannedPersonUpdates': sum(item['status'] == 'person-route-update' for item in goal['events']),
              'facingConfirmed': any(item['target'].get('attention', {}).get('facingCamera') is True for item in ready),
              'confirmationFrames': max((item['confirmationFrames'] for item in ready), default=0),
              'confirmedDistance': ready[-1]['target']['distance'] if ready else None,
              'maxRadiusMeters': round(max(math.dist(positions[0], point) for point in positions), 3),
              'failedCommands': [item.get('detail') for item in commands if item['status'] == 'failed'],
              'errors': probe['errors']}
    if people:
        result['fixtureMovementSpanMeters'] = round(max(math.dist(a, b) for a in people for b in people), 3)
        result['finalFixtureDistanceMeters'] = round(math.dist(positions[-1], [people[-1][0], people[-1][2]]), 3)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=Path)
    parser.add_argument('--require-greeting', action='store_true')
    parser.add_argument('--require-moving-update', action='store_true')
    args = parser.parse_args()
    result = summarize(json.loads(args.filename.read_text()))
    args.filename.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    if args.require_greeting and not (result['goalStatus'] == 'completed' and result['completedWaves'] == 1
                                     and result['facingConfirmed'] and result['confirmationFrames'] >= 3):
        raise SystemExit('接近、朝向确认和实际挥手闭环未通过')
    if args.require_moving_update and (result.get('fixtureMovementSpanMeters', 0) < .8 or result['plannedPersonUpdates'] < 1):
        raise SystemExit('真实移动目标换路未通过')


if __name__ == '__main__':
    main()
