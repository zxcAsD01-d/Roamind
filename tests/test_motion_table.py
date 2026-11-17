"""Motion-table invariants, artifact integrity and fail-closed curation."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_motion_table import add_clip, graph
from motion_table_features import JOINTS, boundary, costs, describe
from collections import Counter


class MotionTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'public/locomotion/bank.json').is_file():
            raise unittest.SkipTest('Requires locally curated motion data; see docs/ASSETS.md')
        cls.clip = next(clip for clip in json.loads((ROOT/'public/locomotion/bank.json').read_text())['clips']
                        if clip['kind'] == 'walk')

    def test_full_body_not_just_legs(self):
        changed = json.loads(json.dumps(self.clip))
        local = np.asarray(changed['localRot']).reshape(-1, 27, 3)
        local[-1, 8, 0] += .4
        changed['localRot'] = local.ravel().tolist()
        score, rms, maximum = costs(boundary(self.clip, self.clip['frames']-1), [boundary(changed, changed['frames']-1)])
        self.assertGreater(score[0], .01)
        self.assertGreater(maximum[0], .35)
        self.assertIn(3, JOINTS)  # Both chest-chain rotations affect the retargeted body.

    def test_rigid_world_transform_invariant(self):
        changed = json.loads(json.dumps(self.clip))
        local = np.asarray(changed['localRot']).reshape(-1, 27, 3)
        root = np.asarray(changed['rootDelta']).reshape(-1, 3)
        yaw = Rotation.from_euler('y', 2.8)
        local[:, 0] = (yaw*Rotation.from_rotvec(local[:, 0])).as_rotvec()
        changed['localRot'] = local.ravel().tolist()
        changed['rootDelta'] = (yaw.apply(root)+[10, 4, -9]).ravel().tolist()
        np.testing.assert_allclose(boundary(self.clip, 10), boundary(changed, 10), atol=2e-6)
        np.testing.assert_allclose(describe(self.clip)['trajectory'], describe(changed)['trajectory'], atol=2e-6)

    def test_quaternion_sign_equivalence(self):
        original = boundary(self.clip, 10)
        changed = original.copy()
        changed[5:5+len(JOINTS)*4] = [-v for v in changed[5:5+len(JOINTS)*4]]
        self.assertLess(costs(original, [changed])[0][0], 1e-5)

    def test_angular_velocity_sensitive(self):
        original = boundary(self.clip, 10)
        changed = original.copy(); changed[-1] += 3
        self.assertGreater(costs(original, [changed])[0][0], .001)

    def test_nonfinite_and_sliding_rejected(self):
        for invalid in (float('nan'), 100):
            clip = json.loads(json.dumps(self.clip)); clip['rootDelta'][90] = invalid
            entries, rejected = [], []
            add_clip(clip, entries, [], Counter(), rejected, 20)
            self.assertFalse(entries)
            self.assertEqual(len(rejected), 1)

    def test_published_binary_and_graph(self):
        table = json.loads((ROOT/'public/motion-table/table.json').read_text())
        binary = (ROOT/'public/motion-table'/table['dataFile']).read_bytes()
        self.assertEqual(hashlib.sha256(binary).hexdigest(), table['dataSha256'])
        values = np.frombuffer(binary, dtype='<f4')
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(len(values), table['dataFloats'])
        offset = 0
        for entry, edges in zip(table['entries'], table['successors']):
            self.assertEqual(offset, entry['offset']); offset += entry['frames']*88
            self.assertEqual(len(entry['entry']), 5+len(JOINTS)*7)
            for edge in edges:
                self.assertLess(edge['to'], len(table['entries']))
                self.assertLessEqual(edge['cost'], .65)
                self.assertLessEqual(edge['poseRms'], .45)
        self.assertEqual(offset, len(values))
        self.assertTrue(any(entry['source'].startswith('ARDY/') for entry in table['entries']))
        self.assertGreater(sum(entry['role'] == 'travel' for entry in table['entries']), 200)

    def test_graph_refuses_incompatible_upper_body(self):
        clip = describe(self.clip)
        other = json.loads(json.dumps(clip)); other['entry'][5:9] = [0, 1, 0, 0]
        entries = [{**clip, 'role': 'travel'}, {**other, 'role': 'travel'}]
        self.assertFalse(any(edge['to'] == 1 for edge in graph(entries)[0]))


if __name__ == '__main__':
    unittest.main()
