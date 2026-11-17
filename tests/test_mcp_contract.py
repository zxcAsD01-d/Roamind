"""Real SDK transport, HTTP forwarding and execution feedback without model weights."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import Request, urlopen

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from mcp_server import ActionStep, AgentIntent, CharacterStep, request
import service
import runtime
import goals
import embodied


class SchemaTests(unittest.TestCase):
    def test_invalid_inputs_are_rejected_before_dispatch(self):
        for value in ({'kind': 'teleport'}, {'kind': 'navigate', 'position': [1, 2]},
                      {'kind': 'face'}, {'kind': 'face', 'heading': float('nan')},
                      {'kind': 'socialize', 'secret': 'invalid'}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AgentIntent.model_validate(value)
        with self.assertRaises(ValidationError):
            ActionStep(action='walk', start=2, end=1)
        with self.assertRaises(ValidationError):
            CharacterStep(kind='waypoint', position=[1, 2, 3, 4])

    def test_unavailable_backend_has_actionable_error(self):
        from unittest.mock import patch
        from urllib.error import URLError
        with patch('urllib.request.urlopen', side_effect=URLError('offline')):
            with self.assertRaisesRegex(RuntimeError, '启动对应'):
                request('/health', base='http://127.0.0.1:1')


class TransportTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        runtime.SESSIONS.clear(); runtime.COMMANDS.clear()
        goals.TASKS.clear(); embodied.KERNELS.clear()
        cls.server = service.ThreadingHTTPServer(('127.0.0.1', 0), service.Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=3)

    def post(self, path: str, body: dict) -> dict:
        call = Request(self.base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urlopen(call, timeout=5) as response:
            return json.load(response)

    async def test_sdk_tools_retries_and_actual_feedback(self):
        env = {**os.environ, 'MOTION_LAB_URL': self.base}
        params = StdioServerParameters(command=sys.executable,
            args=[str(ROOT / 'server/mcp_server.py')], env=env)
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                self.assertTrue({'get_agent_context', 'submit_agent_intent', 'get_character_goal',
                    'select_contact_target', 'follow_target', 'touch_view_text'} <= tools)
                failed = await session.call_tool('get_agent_context', {'client_id': 'missing'})
                self.assertTrue(failed.isError)
                self.post('/api/sim/sync', {'clientId': 'mcp-contract', 'state': {}})
                result = await session.call_tool('get_agent_context', {'client_id': 'mcp-contract'})
                self.assertFalse(result.isError)
                context = json.loads(result.content[0].text)
                self.assertEqual(context['coordinateFrame'], 'visual-odometry')
                self.assertNotIn('rooms', context['world'])
                arguments = {'client_id': 'mcp-contract', 'request_id': 'retry-safe',
                             'intent': {'kind': 'face', 'heading': 30}}
                first = await session.call_tool('submit_agent_intent', arguments)
                second = await session.call_tool('submit_agent_intent', arguments)
                self.assertFalse(first.isError); self.assertFalse(second.isError)
                initial, replay = json.loads(first.content[0].text), json.loads(second.content[0].text)
                self.assertEqual(initial['goal']['id'], replay['goal']['id'])
                self.assertTrue(replay['replayed'])
                reports = []
                for _ in range(4):
                    feedback = self.post('/api/sim/sync', {'clientId': 'mcp-contract',
                        'state': {}, 'reports': reports})
                    reports = [{'id': command['id'], 'status': 'completed'} for command in feedback['commands']]
                status = await session.call_tool('get_character_goal', {'client_id': 'mcp-contract'})
                self.assertEqual(json.loads(status.content[0].text)['goal']['status'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
