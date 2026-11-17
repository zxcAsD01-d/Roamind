"""Real stdio MCP touch using current visual evidence and browser completion."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from preset_mcp_integration import call

ROOT = Path(__file__).resolve().parents[1]


async def verify(session, client_id):
    state = await call(session, 'get_simulation_state')
    visual = next(item['state']['perceptionClientId'] for item in state['sessions'] if item['id'] == client_id)
    frame = await call(session, 'get_visual_body_frame', {'visual_client_id': visual})
    if frame.get('stale') or not frame['people']:
        raise ValueError('需要网页开启双目，并看到当前可达的人体目标')
    person = max(frame['people'], key=lambda item: item['confidence'])
    target = await call(session, 'resolve_touch_target', {'visual_client_id': visual,
                        'person_id': person['id'], 'body_part': 'left_wrist'})
    command = await call(session, 'touch_body', {'target_id': target['targetId'], 'hand': 'right',
                         'operation': 'tap', 'duration': .6, 'client_id': client_id})
    statuses = []
    for _ in range(150):
        status = await call(session, 'get_command_status', {'command_id': command['id']})
        if not statuses or statuses[-1] != status['status']:
            statuses.append(status['status'])
        if status['status'] in ('completed', 'failed', 'interrupted', 'superseded'):
            result = {'tool': 'touch_body', 'command': status, 'statuses': statuses,
                      'target': target, 'visualFrameId': frame['frameId']}
            if status['status'] != 'completed':
                raise AssertionError(result)
            return result
        await asyncio.sleep(.2)
    raise TimeoutError(f'MCP 触摸播放未完成：{statuses}')


async def main(client_id):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await verify(session, client_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'output/touch/mcp.json')
    args = parser.parse_args()
    rendered = json.dumps(asyncio.run(main(args.client_id)), ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered)
    print(rendered)
