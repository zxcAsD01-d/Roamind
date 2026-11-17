"""Exercise high-level stdio MCP tasks against the real browser runtime."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from preset_mcp_integration import call

ROOT = Path(__file__).resolve().parents[1]


async def verify(session, client):
    names = {tool.name for tool in (await session.list_tools()).tools}
    assert {'submit_character_goal', 'get_character_goal', 'cancel_character_goal'} <= names
    task = await call(session, 'submit_character_goal', {'title': 'MCP组合任务', 'client_id': client,
                      'steps': [{'kind': 'wait', 'duration': .8}, {'kind': 'face', 'heading': 0},
                                {'kind': 'wait', 'duration': .6}]})
    statuses = []
    for _ in range(160):
        state = await call(session, 'get_character_goal', {'client_id': client})
        current = (state['goal']['index'], state['goal']['status'])
        if not statuses or current != statuses[-1]:
            statuses.append(current)
        if state['goal']['status'] in ('completed', 'failed', 'cancelled'):
            break
        await asyncio.sleep(.15)
    assert state['goal']['id'] == task['id'] and state['goal']['status'] == 'completed', state
    invalid = await session.call_tool('submit_character_goal', {'title': '无效输入', 'client_id': client,
                                      'steps': [{'kind': 'text-input', 'text': 'hello'}]})
    assert invalid.isError, invalid
    task = await call(session, 'submit_character_goal', {'title': 'MCP取消验收', 'client_id': client,
                      'steps': [{'kind': 'social'}, {'kind': 'action', 'action': 'wave'}]})
    await asyncio.sleep(.8)
    cancelled = await call(session, 'cancel_character_goal', {'client_id': client})
    await asyncio.sleep(2)
    after = await call(session, 'get_character_goal', {'client_id': client})
    assert after['goal']['status'] == 'cancelled', after
    assert after['goal']['steps'][1]['status'] == 'pending', after
    return {'completed': state['goal'], 'statuses': statuses, 'cancelled': cancelled,
            'afterCancel': after['goal'], 'unsupportedInputRejected': invalid.isError}


async def main(client):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await verify(session, client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    args = parser.parse_args()
    result = asyncio.run(main(args.client_id))
    (ROOT/'output/playwright/character/mcp.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
