"""Real stdio MCP -> HTTP -> browser preset playback, with explicit session."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def call(session, name, arguments=None):
    result = await session.call_tool(name, arguments or {})
    if result.isError:
        raise AssertionError(result.content)
    return json.loads(result.content[0].text)


async def complete(session, command):
    statuses = []
    for _ in range(100):
        result = await call(session, 'get_command_status', {'command_id': command['id']})
        if not statuses or statuses[-1] != result['status']:
            statuses.append(result['status'])
        if result['status'] == 'failed':
            raise AssertionError(result)
        if result['status'] == 'completed':
            return statuses
        await asyncio.sleep(.2)
    raise AssertionError(f'命令未完成：{statuses}')


async def main(client_id):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            reset = await call(session, 'interrupt_motion', {'mode': 'reset', 'client_id': client_id})
            await complete(session, reset)
            face = await call(session, 'face_toward', {'heading': -90, 'client_id': client_id})
            statuses = await complete(session, face)
            state = await call(session, 'get_simulation_state')
            current = next(item['state'] for item in state['sessions'] if item['id'] == client_id)
            assert abs(current['heading']+90) < 1, current
            assert current['locomotion']['mode'] == 'preset' and not current['generating'], current
            assert 'generating' not in statuses, statuses
            print(json.dumps({'statuses': statuses, 'heading': current['heading'],
                              'mode': current['locomotion']['mode'], 'generating': current['generating']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    args = parser.parse_args()
    asyncio.run(main(args.client_id))
