"""Real stdio MCP -> look, read, touch, refusal and cancellation in browser."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from preset_mcp_integration import call

ROOT = Path(__file__).resolve().parents[1]


async def terminal(session, command):
    statuses = []
    for _ in range(250):
        state = await call(session, 'get_command_status', {'command_id': command['id']})
        if not statuses or statuses[-1] != state['status']:
            statuses.append(state['status'])
        if state['status'] in ('completed', 'failed', 'interrupted', 'superseded'):
            return {'command': state, 'statuses': statuses}
        await asyncio.sleep(.2)
    raise TimeoutError(statuses)


async def verify(session, client_id):
    tools = {tool.name for tool in (await session.list_tools()).tools}
    assert {'recognize_text', 'read_view_text', 'touch_view_text'} <= tools
    state = await call(session, 'get_simulation_state')
    visual = next(item['state']['perceptionClientId'] for item in state['sessions'] if item['id'] == client_id)
    read = await call(session, 'read_view_text', {'visual_client_id': visual, 'text': '确认', 'direction': 'left'})
    assert read['status'] == 'selected', read
    command = await call(session, 'touch_view_text', {'text': '确认', 'heading': 0, 'direction': 'left', 'client_id': client_id})
    touched = await terminal(session, command)
    assert touched['command']['status'] == 'completed', touched
    missing = await call(session, 'touch_view_text', {'text': '绝不存在的文字937', 'client_id': client_id})
    refused = await terminal(session, missing)
    assert refused['command']['status'] == 'failed', refused
    command = await call(session, 'touch_view_text', {'text': '确认', 'heading': 15, 'client_id': client_id})
    began = None
    for _ in range(50):
        began = await call(session, 'get_command_status', {'command_id': command['id']})
        if began['status'] in ('generating', 'playing'):
            break
        assert began['status'] in ('queued', 'received'), began
        await asyncio.sleep(.1)
    assert began['status'] in ('generating', 'playing'), began
    await call(session, 'interrupt_motion', {'mode': 'hold', 'client_id': client_id})
    cancelled = await terminal(session, command)
    assert cancelled['command']['status'] in ('interrupted', 'superseded'), cancelled
    await asyncio.sleep(3)
    final = await call(session, 'get_command_status', {'command_id': command['id']})
    assert final['status'] == cancelled['command']['status'], final
    return {'read': read, 'touched': touched, 'missing': refused, 'beforeInterrupt': began, 'interrupted': cancelled}


async def main(client_id):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await verify(session, client_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'output/playwright/touch-ocr/mcp.json')
    args = parser.parse_args()
    rendered = json.dumps(asyncio.run(main(args.client_id)), ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered)
