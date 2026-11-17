"""Integration regression: run against the resident ARDY HTTP service."""

import concurrent.futures
import json
import time
import unittest
import urllib.request
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

BASE = "http://127.0.0.1:8100"


def post(path, body):
    request = urllib.request.Request(
        BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=30)


def generate(client, revision, prompt, history=None, duration=2):
    with post("/api/generate/stream", {
        "clientId": client, "revision": revision, "prompt": prompt,
        "history": history, "duration": duration,
    }) as response:
        events = [json.loads(line) for line in response if line.strip()]
    errors = [event for event in events if event["type"] == "error"]
    if errors:
        raise AssertionError(errors)
    return events


def saved_motion(events):
    entry = next(event["entry"] for event in events if event["type"] == "complete")
    with urllib.request.urlopen(BASE + entry["url"]) as response:
        return json.load(response)


def history_from(motion, frames):
    return {
        **motion, "frames": frames, "duration": frames / motion["fps"],
        "localRot": motion["localRot"][:frames * 81],
        "rootDelta": motion["rootDelta"][:frames * 3],
    }


class ContinuityTest(unittest.TestCase):
    def test_running_history_and_root_coordinates(self):
        client = str(uuid.uuid4())
        walk = saved_motion(generate(client, 1, "A person walks forward.", duration=4))
        # A non-token-aligned splice, translated away from the world origin.
        history = history_from(walk, 47)
        root = np.array(history["rootDelta"]).reshape(-1, 3) + [12, 0, -7]
        history["rootDelta"] = root.reshape(-1).tolist()
        for revision, prompt in [(2, "A person runs."), (3, "A person waves their hand.")]:
            events = generate(client, revision, prompt, history)
            chunk = next(event for event in events if event["type"] == "chunk")
            self.assertEqual(chunk["startFrame"], history["frames"])
            combined = saved_motion(events)
            count = history["frames"]
            np.testing.assert_allclose(combined["rootDelta"][:count * 3], history["rootDelta"], atol=1e-6)
            np.testing.assert_allclose(combined["localRot"][:count * 81], history["localRot"], atol=1e-6)
            prior = np.array(history["rootDelta"]).reshape(-1, 3)
            first = np.array(chunk["rootDelta"]).reshape(-1, 3)[0]
            np.testing.assert_allclose(first, 2 * prior[-1] - prior[-2], atol=2e-6)
            old_pose = Rotation.from_rotvec(np.array(history["localRot"][-81:]).reshape(-1, 3))
            new_pose = Rotation.from_rotvec(np.array(chunk["localRot"][:81]).reshape(-1, 3))
            angle = np.max((old_pose.inv() * new_pose).magnitude())
            self.assertLessEqual(angle, 0.351)
            print(f"{prompt}: first chunk joins at frame {count}, max joint step {angle:.4f} rad")
            history = history_from(combined, min(79, combined["frames"]))

    def test_cancel_supersede_and_stale_requests(self):
        client = str(uuid.uuid4())
        with post("/api/generate/stream", {
            "clientId": client, "revision": 1,
            "prompt": "A person walks forward.", "duration": 60,
        }) as response:
            first = json.loads(response.readline())
            self.assertEqual(first["type"], "chunk")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                old = executor.submit(lambda: [json.loads(line) for line in response if line.strip()])
                start = time.monotonic()
                newest = generate(client, 3, "A person waves their hand.")
                self.assertLess(time.monotonic() - start, 5)
                self.assertTrue(any(e["type"] == "complete" for e in newest))
                self.assertFalse(any(e["type"] == "complete" for e in old.result(timeout=5)))
        self.assertEqual(generate(client, 2, "A person runs."), [])
        with post("/api/generate/cancel", {"clientId": client, "revision": 5}) as response:
            self.assertTrue(json.load(response)["cancelled"])
        self.assertEqual(generate(client, 4, "A person runs."), [])

    def test_single_frame_history(self):
        client = str(uuid.uuid4())
        motion = saved_motion(generate(client, 1, "A person walks forward."))
        history = history_from(motion, 1)
        events = generate(client, 2, "A person runs.", history)
        chunk = next(event for event in events if event["type"] == "chunk")
        np.testing.assert_allclose(chunk["rootDelta"][:3], history["rootDelta"], atol=2e-6)

    def test_every_autoregressive_boundary_is_continuous(self):
        client = str(uuid.uuid4())
        events = generate(client, 1, "A person walks forward.", duration=8)
        chunks = [event for event in events if event["type"] == "chunk"]
        self.assertGreaterEqual(len(chunks), 3)
        previous_local = None
        previous_root = None
        for chunk in chunks:
            local = np.asarray(chunk["localRot"], dtype=np.float32).reshape(chunk["frames"], 27, 3)
            root = np.asarray(chunk["rootDelta"], dtype=np.float32).reshape(chunk["frames"], 3)
            if previous_local is not None and previous_root is not None:
                old_pose = Rotation.from_rotvec(previous_local[-1])
                new_pose = Rotation.from_rotvec(local[0])
                max_step = np.max((old_pose.inv() * new_pose).magnitude())
                self.assertLessEqual(max_step, 0.25)
                incoming = root[0] - previous_root[-1]
                outgoing = previous_root[-1] - previous_root[-2]
                self.assertLess(float(np.linalg.norm(incoming-outgoing)), .05)
            self.assertEqual(chunk["engine"]["mode"], "dual-overlap")
            self.assertEqual(chunk["engine"]["instances"], 2)
            self.assertEqual(chunk["engine"]["scheduling"], "alternating")
            self.assertTrue(chunk['engine']['gaitLock'])
            self.assertEqual(chunk['engine']['contactCorrection'], 'ardy-native')
            self.assertEqual(len(chunk['footContacts']), chunk['frames']*4)
            self.assertEqual(len(chunk['restJoints']), 27*3)
            previous_local = local
            previous_root = root

    def test_model_workers_alternate_without_waiting_for_a_pair(self):
        events = generate(str(uuid.uuid4()), 1, "A person walks forward.", duration=6)
        chunks = [event for event in events if event["type"] == "chunk"]
        turns = [chunk["engine"]["lastPair"][-1] for chunk in chunks[:-1]]
        self.assertEqual([turn["startFrame"] for turn in turns], [0, 20, 40, 60, 80])
        self.assertEqual([chunk["startFrame"] for chunk in chunks], [0, 20, 40, 60, 80, 100])
        for first, second in zip(turns, turns[1:]):
            self.assertNotEqual(first["lane"], second["lane"])
            self.assertNotEqual(first["pid"], second["pid"])
            self.assertLessEqual(first["finishedMs"], second["startedMs"])
        self.assertTrue(all(chunk["frames"] == 20 for chunk in chunks))

    def test_invalid_history_reports_an_error(self):
        with post("/api/generate/stream", {
            "clientId": str(uuid.uuid4()), "revision": 1,
            "prompt": "A person runs.", "duration": 2, "history": [],
        }) as response:
            events = [json.loads(line) for line in response if line.strip()]
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("history must be an object", events[-1]["error"])

    def test_timeline_switches_at_nonsecond_boundary(self):
        with post('/api/plans/timeline', {'steps': [
            {'action': 'walk', 'start': 0, 'end': 1.4},
            {'action': 'wave', 'start': 1.4, 'end': 4.1},
        ]}) as response:
            plan = json.load(response)
        with post('/api/generate/stream', {
            'clientId': str(uuid.uuid4()), 'revision': 1, 'planId': plan['id'],
            'prompt': 'A person walks forward.', 'duration': 4.1,
        }) as response:
            events = [json.loads(line) for line in response if line.strip()]
        self.assertFalse([event for event in events if event['type'] == 'error'])
        chunks = [event for event in events if event['type'] == 'chunk']
        self.assertEqual(sum(chunk['frames'] for chunk in chunks), 82)
        self.assertIn(28, [chunk['startFrame'] for chunk in chunks])
        self.assertTrue(all(not (chunk['startFrame'] < 28 < chunk['startFrame']+chunk['frames'])
                            for chunk in chunks))
        self.assertEqual(saved_motion(events)['timeline'][1]['action'], 'wave')


if __name__ == "__main__":
    unittest.main(verbosity=2)
