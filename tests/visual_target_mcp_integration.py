"""Real MCP: retrieve camera frame, annotate arbitrary pixel and execute common contact."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from preset_mcp_integration import call
from text_mcp_integration import terminal

ROOT = Path(__file__).resolve().parents[1]


async def select(session, visual):
    attempts = []
    for _ in range(12):
        response = await session.call_tool('get_target_frame', {'visual_client_id': visual})
        if response.isError:
            attempts.append(response.content[0].text)
            await asyncio.sleep(.2)
            continue
        frame = json.loads(response.content[0].text)
        target = await call(session, 'select_contact_target', {'visual_client_id': visual,
                            'pixel': [280., 250.], 'frame_id': frame['frameId'], 'label': 'MCP任意表面'})
        return target, {key: value for key, value in frame.items() if key != 'image'}, attempts
    raise AssertionError(attempts)


async def verify(session, client):
    names = {tool.name for tool in (await session.list_tools()).tools}
    assert {'get_target_frame', 'select_contact_target', 'contact_target'} <= names
    state = await call(session, 'get_simulation_state')
    visual = next(item['state']['perceptionClientId'] for item in state['sessions'] if item['id'] == client)
    target, frame, attempts = await select(session, visual)
    assert target['targetId'].startswith('point-') and 'personId' not in target and 'text' not in target
    command = await call(session, 'contact_target', {'target_id': target['targetId'], 'effector': 'right_hand',
                         'operation': 'hold', 'duration': .6, 'client_id': client})
    touched = await terminal(session, command)
    assert touched['command']['status'] == 'completed', touched
    foreign = await session.call_tool('select_contact_target', {'visual_client_id': 'wrong-camera',
                                      'pixel': [280., 250.], 'frame_id': frame['frameId']})
    assert foreign.isError, foreign
    return {'frame': frame, 'target': target, 'touched': touched, 'staleFrameRetries': attempts,
            'foreignCameraRejected': foreign.isError}


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
    (ROOT/'output/playwright/visual-targets/mcp.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
