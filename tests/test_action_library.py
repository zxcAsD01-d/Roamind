"""Persistent variants, real geometry rejection, rigid replay and immutable history."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server'))
from action_library import ActionLibrary
from planning import timed_plan
import library_motion


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'library.sqlite3'
        self.library = ActionLibrary(self.path)
        idle = json.loads((ROOT/'public/locomotion/bank.json').read_text())['clips'][0]
        self.history = {**idle, 'frames': 1, 'duration': .05, 'localRot': idle['localRot'][:81],
                        'rootDelta': [-5, 0, 3.2], 'footContacts': idle['footContacts'][:4]}
        self.plan = timed_plan({'start': [-5, .92, 3.2], 'steps': [{'action': 'wave', 'start': 0, 'end': 3}]})

    def generated(self, amplitude=.3):
        motion = copy.deepcopy(self.history)
        local = np.tile(np.asarray(motion['localRot']).reshape(1, 27, 3), (61, 1, 1))
        local[1:, 14, 2] += amplitude*np.sin(np.linspace(0, 2*np.pi, 60))
        motion.update(frames=61, duration=3.05, localRot=local.ravel().tolist(),
                      rootDelta=[-5, 0, 3.2]*61, footContacts=motion['footContacts']*61)
        return motion

    def remember(self, amplitude=.3):
        return self.library.remember(self.plan, self.generated(amplitude), 1, self.history)

    def test_missing_generate_persist_and_reuse(self):
        self.assertEqual(self.library.resolve(self.plan, self.history)['reason'], 'missing')
        stored = self.remember()
        self.assertTrue(stored['accepted'], stored)
        reopened = ActionLibrary(self.path)
        replay = reopened.resolve(self.plan, self.history)
        self.assertTrue(replay['hit'], replay)
        self.assertEqual(replay['motion']['localRot'][:81], self.history['localRot'])
        self.assertEqual(replay['motion']['rootDelta'][:3], self.history['rootDelta'])
        self.assertEqual(replay['motion']['frames'], 61)

    def test_frequency_expands_and_recent_variant_is_avoided(self):
        self.library.resolve(self.plan, self.history)
        first = self.remember()
        for _ in range(3):
            self.assertTrue(self.library.resolve(self.plan, self.history)['hit'])
        expand = self.library.resolve(self.plan, self.history)
        self.assertEqual((expand['reason'], expand['targetVariants']), ('expand', 2))
        second = self.remember(.9)
        self.assertTrue(second['accepted'], second)
        a = self.library.resolve(self.plan, self.history)
        b = self.library.resolve(self.plan, self.history)
        self.assertNotEqual(a['clipId'], b['clipId'])
        self.assertEqual({a['clipId'], b['clipId']}, {first['clipId'], second['clipId']})

    def test_duplicate_and_corrupt_motion_are_not_added(self):
        self.library.resolve(self.plan, self.history)
        self.assertTrue(self.remember()['accepted'])
        self.assertFalse(self.remember()['accepted'])
        corrupt = self.generated(); corrupt['localRot'][150] = float('nan')
        self.assertFalse(self.library.remember(self.plan, corrupt, 1, self.history)['accepted'])
        self.assertEqual(len(self.library.catalog()['clips']), 1)

    def test_rigid_translation_and_heading_share_profile(self):
        self.library.resolve(self.plan, self.history); self.remember()
        history = copy.deepcopy(self.history)
        local = np.asarray(history['localRot']).reshape(27, 3)
        local[0] = (Rotation.from_euler('y', np.pi/2)*Rotation.from_rotvec(local[0])).as_rotvec()
        history['localRot'] = local.ravel().tolist(); history['rootDelta'] = [-4, 0, 3.2]
        plan = timed_plan({'start': [-4, .92, 3.2], 'startHeading': 90,
                           'steps': [{'action': 'wave', 'start': 0, 'end': 3}]})
        self.assertEqual(library_motion.profile(plan, history)[0], library_motion.profile(self.plan, self.history)[0])
        result = self.library.resolve(plan, history)
        self.assertTrue(result['hit'], result)
        np.testing.assert_allclose(result['motion']['rootDelta'][-3:], [-4, 0, 3.2])

    def test_collision_and_duration_mismatch_do_not_replay(self):
        self.library.resolve(self.plan, self.history); self.remember()
        plan = copy.deepcopy(self.plan); plan['start'] = [-8.8, .92, 4.7]
        plan['rootTrack'] = [plan['start']]*60
        history = {**self.history, 'rootDelta': [-8.8, 0, 4.7]}
        result = self.library.resolve(plan, history)
        self.assertFalse(result['hit'], result)
        self.assertEqual(result['reason'], 'unsafe_or_incompatible')
        other = timed_plan({'start': [-5, .92, 3.2], 'steps': [{'action': 'wave', 'start': 0, 'end': 4}]})
        self.assertFalse(self.library.resolve(other, self.history)['hit'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
