"""Real stdio MCP -> stored variant -> named browser playback."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from preset_mcp_integration import call

ROOT = Path(__file__).resolve().parents[1]


async def complete(session, command):
    statuses = []
    for _ in range(150):
        current = await call(session, 'get_command_status', {'command_id': command['id']})
        if not statuses or statuses[-1] != current['status']:
            statuses.append(current['status'])
        if current['status'] in ('failed', 'interrupted', 'superseded'):
            raise AssertionError(current)
        if current['status'] == 'completed':
            return statuses
        await asyncio.sleep(.2)
    raise AssertionError(f'动作未完成：{statuses}')


async def verify(session, client_id):
    catalog = await call(session, 'get_action_library')
    clip = next((item for item in catalog['clips'] if item['action'] == 'wave'), None)
    if clip is None:
        raise AssertionError('请先通过网页或控制脚本生成一条合格挥手动作')
    await complete(session, await call(session, 'interrupt_motion', {'mode': 'reset', 'client_id': client_id}))
    try:
        command = await call(session, 'perform_action', {'clip_id': clip['id'], 'client_id': client_id})
        statuses = await complete(session, command)
        sessions = (await call(session, 'get_simulation_state'))['sessions']
        state = next(item['state'] for item in sessions if item['id'] == client_id)
        assert state['locomotion']['mode'] == 'library', state
        assert 'generating' not in statuses and not state['generating'], statuses
        assert any(phase.get('variant') == clip['id'] for phase in state['locomotion']['phases']), state
        return {'clipId': clip['id'], 'statuses': statuses, 'mode': state['locomotion']['mode'],
                'position': state['position'], 'generating': state['generating']}
    finally:
        await complete(session, await call(session, 'interrupt_motion', {'mode': 'reset', 'client_id': client_id}))


async def main(client_id):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await verify(session, client_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(asyncio.run(main(args.client_id)), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)
