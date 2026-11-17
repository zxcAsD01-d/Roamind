"""AMASS curation contracts, seam features and bounded graph regression."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'server'))
from amass_search import frame_features, quality, transition_cost
from exploration_memory import ExplorationMemory


class AmassSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'public/locomotion/index.json').is_file():
            raise unittest.SkipTest('Requires locally curated motion data; see docs/ASSETS.md')
        cls.payload = (ROOT/'public/locomotion/bank.json').read_bytes()
        cls.bank = json.loads(cls.payload)
        cls.index = json.loads((ROOT/'public/locomotion/index.json').read_text())

    def test_bank_identity_and_required_categories(self):
        self.assertEqual(self.index['bankSha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual({clip['kind'] for clip in self.bank['clips']}, {'idle', 'start', 'walk', 'stop', 'turn'})
        self.assertEqual(len({clip['name'] for clip in self.bank['clips']}), len(self.bank['clips']))

    def test_all_published_clips_pass_quality_and_have_finite_features(self):
        for clip in self.bank['clips']:
            with self.subTest(clip=clip['name']):
                self.assertGreaterEqual(quality(clip)['supportSamples'], 4)
                features = frame_features(clip)
                self.assertEqual(features.shape, (clip['frames'], 46))
                self.assertTrue(np.isfinite(features).all())
                for row in clip['search']['entries']:
                    np.testing.assert_allclose(row[1:], features[row[0]], atol=1e-5)

    def test_nonfinite_and_root_teleport_are_rejected(self):
        clip = copy.deepcopy(self.bank['clips'][0])
        clip['localRot'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            quality(clip)
        clip = copy.deepcopy(self.bank['clips'][0])
        clip['rootDelta'][6] += 10
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            quality(clip)

    def test_transition_graph_only_references_valid_indexed_frames(self):
        clips = {clip['name']: clip for clip in self.bank['clips']}
        counts = {}
        self.assertTrue(self.index['transitions'])
        for edge in self.index['transitions']:
            self.assertIn(edge['from'], clips)
            target = clips[edge['to']]
            self.assertIn(edge['frame'], [row[0] for row in target['search']['entries']])
            self.assertLessEqual(edge['cost'], .65001)
            key = (edge['from'], target['kind'])
            counts[key] = counts.get(key, 0)+1
        self.assertLessEqual(max(counts.values()), 4)

    def test_transition_cost_penalizes_velocity_and_support_mismatch(self):
        features = frame_features(self.bank['clips'][0])
        query = features[0].copy()
        candidate = query.copy()
        candidate[0] += 2
        candidate[2:4] = 1-query[2:4]
        same, changed = transition_cost(query, np.asarray([query, candidate]))
        self.assertAlmostEqual(same, 0, places=3)
        self.assertGreater(changed, 1)

    def test_clearance_cache_reuses_and_invalidates_after_new_obstacle(self):
        memory = ExplorationMemory()
        memory.cells = {(x, z): (-2., 100., 1) for x in range(5) for z in range(5)}
        with patch.object(memory, 'compute_clearance_penalties', wraps=memory.compute_clearance_penalties) as compute:
            first = memory.clearance_penalties()
            self.assertIs(first, memory.clearance_penalties())
            self.assertEqual(compute.call_count, 1)
            memory.cells[(2, 2)] = (2., 101., 1)
            memory.map_revision += 1
            second = memory.clearance_penalties()
            self.assertEqual(compute.call_count, 2)
            self.assertIsNot(first, second)
            self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
