"""Read-only MCP validation against an actual browser; does not interrupt its goal."""
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
    tools = {row.name: row for row in (await session.list_tools()).tools}
    required = {'get_object_interactions', 'plan_object_interaction', 'submit_character_goal', 'get_motion_capabilities'}
    assert required <= tools.keys(), tools.keys()
    schema = tools['plan_object_interaction'].inputSchema
    assert schema['properties']['execute']['default'] is False, schema
    motion = await call(session, 'get_motion_capabilities', {})
    assert len(motion['actions']) >= 20 and all(row['overlayLocomotion'] for row in motion['actions']), motion
    before = await call(session, 'get_character_goal', {'client_id': client})
    preview, catalog = None, None
    for _ in range(50):
        catalog = await call(session, 'get_object_interactions', {'client_id': client})
        available = [row for row in catalog['objects'] if row['fresh']]
        if available:
            result = await session.call_tool('plan_object_interaction', {
                'client_id': client, 'object_id': available[0]['objectId'], 'operation': 'inspect'})
            if not result.isError:
                preview = json.loads(next(item.text for item in result.content if item.type == 'text'))
                break
        await asyncio.sleep(.3)
    assert preview is not None, '真实感知没有产生可预览的新鲜物体'
    assert preview['steps'][0]['kind'] == 'interact' and preview['revalidateAtExecution'], preview
    invalid = await session.call_tool('plan_object_interaction', {
        'client_id': client, 'object_id': 'missing-object-test', 'operation': 'touch'})
    assert invalid.isError, invalid
    after = await call(session, 'get_character_goal', {'client_id': client})
    assert (before['goal'] or {}).get('id') == (after['goal'] or {}).get('id'), after
    return {'passed': True, 'tools': sorted(required), 'objects': len(catalog['objects']),
            'motion': motion, 'preview': preview, 'invalidRejected': invalid.isError, 'goalPreserved': True}


async def main(client):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/'server/mcp_server.py')])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await verify(session, client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    result = asyncio.run(main(parser.parse_args().client_id))
    (ROOT/'output/lumen-rebuild/mcp.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
