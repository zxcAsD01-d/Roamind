"""Control a named local browser session; wait for actual playback completion."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class Controller:
    def __init__(self, base, client_id, timeout):
        self.base, self.client_id, self.timeout = base.rstrip('/'), client_id, timeout

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = urllib.request.Request(self.base+path, data, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'无法连接动作服务 {self.base}: {exc.reason}') from exc

    def session(self):
        sessions = [s for s in self.request('/api/sim/state')['sessions'] if s['ageSeconds'] < 5
                    and (not self.client_id or s['id'] == self.client_id)]
        if len(sessions) != 1:
            raise ValueError('请用 --client-id 指定一个在线网页；用 sessions 查看会话 ID')
        self.client_id = sessions[0]['id']
        return sessions[0]['state']

    def execute(self, path, arguments):
        self.session()
        command = self.request(path, {**arguments, 'clientId': self.client_id})
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            current = self.request('/api/sim/commands/'+command['id'])
            if current['status'] in ('failed', 'interrupted', 'superseded'):
                raise RuntimeError(f"命令 {current['status']}: {current.get('detail', '')}")
            if current['status'] == 'completed':
                return {'command': current, 'state': self.session()}
            time.sleep(.2)
        raise TimeoutError(f"命令超时 {self.timeout}s: {command['id']}")

    def calibrate(self):
        report = {'clientId': self.client_id, 'moves': [], 'turns': []}
        try:
            for distance in (.65, 1.5, 3):
                self.execute('/api/sim/interrupt', {'mode': 'reset'})
                start = self.session()['position']
                target = [start[0], start[1], start[2]+distance]
                result = self.execute('/api/sim/move', {'destination': target})['state']
                error = math.hypot(result['position'][0]-target[0], result['position'][2]-target[2])
                report['moves'].append({'requested': distance, 'errorMeters': error, 'state': result})
                if error > .3:
                    raise AssertionError(f'移动误差超限：{error:.3f}m')
            for heading in (30, -60, 90, -90):
                self.execute('/api/sim/interrupt', {'mode': 'reset'})
                result = self.execute('/api/sim/face', {'heading': heading})['state']
                error = abs((result['heading']-heading+180)%360-180)
                report['turns'].append({'requested': heading, 'errorDegrees': error, 'state': result})
                if error > 1 or result['generating']:
                    raise AssertionError(f'转向未通过：{error:.3f}度')
        finally:
            self.execute('/api/sim/interrupt', {'mode': 'reset'})
        return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8100')
    parser.add_argument('--client-id')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--output', type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('sessions', 'state', 'library', 'calibrate'):
        commands.add_parser(name)
    move = commands.add_parser('move'); move.add_argument('destination', help='房间 ID 或 JSON [x,y,z]')
    face = commands.add_parser('face'); face.add_argument('heading', type=float)
    action = commands.add_parser('action'); action.add_argument('action', nargs='?', default='wave')
    action.add_argument('--duration', type=float, default=3); action.add_argument('--clip-id')
    stop = commands.add_parser('stop'); stop.add_argument('--reset', action='store_true')
    return parser.parse_args()


def dispatch(control, args):
    if args.command == 'sessions':
        return control.request('/api/sim/state')['sessions']
    if args.command == 'library':
        return control.request('/api/library')
    if args.command == 'state':
        return control.session()
    if args.command == 'calibrate':
        return control.calibrate()
    if args.command == 'move':
        destination = json.loads(args.destination) if args.destination.startswith('[') else args.destination
        return control.execute('/api/sim/move', {'destination': destination})
    if args.command == 'face':
        return control.execute('/api/sim/face', {'heading': args.heading})
    if args.command == 'action':
        return control.execute('/api/sim/action', {'action': args.action, 'duration': args.duration,
                                                 **({'clipId': args.clip_id} if args.clip_id else {})})
    return control.execute('/api/sim/interrupt', {'mode': 'reset' if args.reset else 'hold'})


def main():
    args = parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('--timeout 必须是正有限数值')
    result = dispatch(Controller(args.url, args.client_id, args.timeout), args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, AssertionError, OSError) as exc:
        print(f'控制失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc
