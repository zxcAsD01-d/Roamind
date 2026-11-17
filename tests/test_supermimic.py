"""Real pretrained inference, reference layout and bounded realtime contract."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_policy import MODEL_DIRECTORY, SuperMimicPolicy, checked_inputs
from supermimic_reference import G1Reference, observation, rotation6d
from supermimic_runtime import PROFILE, RealtimePolicy, RequestFailure
from supermimic_service import Server


def request(identity: str = 'test', sequence: int = 0, action: np.ndarray | None = None) -> dict:
    observed = np.zeros(99, np.float32)
    if action is not None:
        observed[64:93] = action
    return {'profile': PROFILE, 'sessionId': identity, 'sequence': sequence, 'reset': sequence == 0,
            'deadlineMs': 100, 'observation': observed.tolist(), 'goal': [0.] * 730}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = SuperMimicPolicy()

    def test_artifact_and_original_network_match(self):
        source = ROOT / 'external/SuperMimic-reproduction/supermimic/networks.py'
        expected = self.policy.config['source']['runtime_source_sha256']['networks.py']
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
        names = {'mlp', 'RunningNorm', 'run_mlp', 'Policy'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        namespace = {'torch': torch, 'nn': torch.nn, 'NUM_J': 29, 'AMP_DTYPE': None}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        original = namespace['Policy'](99, 730).eval()
        original.load_state_dict(load_file(str(MODEL_DIRECTORY / 'policy.safetensors')), strict=True)
        random = np.random.default_rng(417)
        for _ in range(5):
            h = random.normal(size=(1, 10, 99)).astype(np.float32)
            g = random.normal(size=(1, 730)).astype(np.float32)
            with torch.inference_mode():
                expected_action = original(torch.from_numpy(h), torch.from_numpy(g)).numpy()[0]
            np.testing.assert_allclose(self.policy.predict(h, g), expected_action, atol=1e-5, rtol=1e-5)

    def test_mlx_matches_original_cpu(self):
        mlx = SuperMimicPolicy(backend='mlx')
        h, g = np.zeros((1, 10, 99), np.float32), np.zeros((1, 730), np.float32)
        np.testing.assert_allclose(mlx.predict(h, g), self.policy.predict(h, g), atol=3e-4, rtol=1e-4)

    def test_invalid_shapes_and_nonfinite(self):
        for history, goal in [(np.zeros((1, 10, 98)), np.zeros((1, 730))),
                              (np.zeros((1, 10, 99)), np.full((1, 730), np.nan)),
                              (np.full((1, 10, 99), np.inf), np.zeros((1, 730)))]:
            with self.assertRaises(ValueError):
                checked_inputs(history, goal)


class ReferenceTests(unittest.TestCase):
    def reference(self, fps: float = 50) -> G1Reference:
        p = np.column_stack((np.arange(20) * .02, np.zeros(20), np.full(20, .8)))
        return G1Reference.from_samples(p, np.tile([0, 0, 0, 1], (20, 1)), np.zeros((20, 29)), fps)

    def test_goal_layout_and_held_future(self):
        reference = self.reference()
        first = reference.goal(0).reshape(10, 73)
        np.testing.assert_allclose(first[:, 0], np.arange(1, 11) * .02, atol=1e-6)
        np.testing.assert_allclose(first[0, 3:9], [1, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(first[:, 9], 1, atol=1e-6)
        held = reference.goal(19).reshape(10, 73)
        np.testing.assert_array_equal(held[:, 9:15], 0)
        np.testing.assert_array_equal(held[:, 44:], 0)

    def test_rotation6d_column_order(self):
        r = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        np.testing.assert_allclose(rotation6d(r), [0, -1, 1, 0, 0, 0], atol=1e-6)

    def test_body_frame_observation(self):
        r = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        o = observation(np.ones(29), np.full(29, 2), r, np.array([1, 0, 0]), np.full(29, 3), r, np.zeros(29))
        np.testing.assert_allclose(o[58:61], [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(o[61:64], [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(o[93:], [1, 0, 0, 1, 0, 0], atol=1e-6)

    def test_reject_bad_quaternion_and_duration(self):
        for quat, fps in [(np.zeros((20, 4)), 50), (np.tile([0, 0, 0, 1], (20, 1)), 10000)]:
            with self.assertRaises(ValueError):
                G1Reference.from_samples(np.zeros((20, 3)), quat, np.zeros((20, 29)), fps)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = SuperMimicPolicy()

    def setUp(self):
        self.runtime = RealtimePolicy(self.policy)

    def tearDown(self):
        self.runtime.close()

    def test_history_sequence_and_actual_action_feedback(self):
        first = self.runtime.infer(request())
        action = np.array(first['action'])
        self.runtime.infer(request(sequence=1, action=action))
        session = self.runtime.sessions['test']
        np.testing.assert_allclose(session.history[0, -1, 64:93], action, atol=1e-6)
        with self.assertRaises(RequestFailure):
            self.runtime.infer(request(sequence=1, action=action))
        with self.assertRaises(RequestFailure):
            self.runtime.infer(request(sequence=2, action=np.ones(29)))
        self.assertEqual(session.sequence, 1)

    def test_other_skeleton_rejected(self):
        value = request(); value['profile'] = 'vrm'
        with self.assertRaises(RequestFailure) as failure:
            self.runtime.infer(value)
        self.assertEqual(failure.exception.status, 422)

    def test_deadline_drops_output_without_advancing_history(self):
        with patch.object(self.policy, 'predict', return_value=np.zeros(29)), \
             patch('supermimic_runtime.time.perf_counter', side_effect=[0, 1]):
            with self.assertRaises(RequestFailure) as failure:
                self.runtime.infer(request())
        self.assertEqual(failure.exception.status, 504)
        self.assertFalse(self.runtime.sessions)
        self.assertEqual(self.runtime.late, 1)

    def test_busy_does_not_queue(self):
        with self.runtime.lock:
            with self.assertRaises(RequestFailure) as failure:
                self.runtime.infer(request())
        self.assertEqual(failure.exception.status, 429)

    def test_expired_session_requires_reset(self):
        self.runtime.infer(request())
        self.runtime.sessions['test'].updated = time.monotonic() - 121
        with self.assertRaises(RequestFailure) as failure:
            self.runtime.infer(request(sequence=1))
        self.assertEqual(failure.exception.status, 409)

    def test_bounds_and_session_isolation(self):
        with patch.object(self.policy, 'predict', return_value=np.zeros(29)):
            for i in range(600):
                self.runtime.infer(request(identity=f'session-{i}'))
        self.assertEqual(len(self.runtime.sessions), 16)
        self.assertEqual(len(self.runtime.times), 512)
        self.assertNotIn('session-0', self.runtime.sessions)
        self.assertFalse(self.runtime.health()['vrmEnabled'])

    def test_malformed_request_is_not_committed(self):
        changes = [{'sequence': True}, {'deadlineMs': float('nan')}, {'reset': 'yes'},
                   {'observation': [1]}, {'goal': [[0] * 730]}, {'sessionId': ''}]
        for change in changes:
            value = request(); value.update(change)
            with self.assertRaises((ValueError, RequestFailure)):
                self.runtime.infer(value)
        self.assertFalse(self.runtime.sessions)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server(0, RealtimePolicy(SuperMimicPolicy()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.server.runtime.close()

    def post(self, value: object, headers: dict | None = None):
        return urlopen(Request(self.url + '/infer', data=json.dumps(value).encode(),
                               headers={'Content-Type': 'application/json', **(headers or {})}), timeout=5)

    def test_http_real_inference_and_health(self):
        with self.post(request(identity='http')) as response:
            value = json.load(response)
        self.assertEqual(len(value['action']), 29)
        self.assertFalse(value['appliedToVrm'])
        with urlopen(self.url + '/health', timeout=5) as response:
            self.assertTrue(json.load(response)['ready'])

    def test_http_rejects_browser_and_bad_body(self):
        for value, headers, expected in [(request(), {'Origin': 'https://example.org'}, 403),
                                         ([], {}, 400), ({'padding': 'x' * 70000}, {}, 413)]:
            with self.assertRaises(HTTPError) as failure:
                self.post(value, headers)
            self.assertEqual(failure.exception.code, expected)
            self.assertTrue(json.load(failure.exception)['retainOriginalMotion'])


if __name__ == '__main__':
    unittest.main()
