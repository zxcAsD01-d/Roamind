"""Validate physical motion, not merely rewritten contact labels."""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quality_metrics', ROOT/'tests/motion_quality.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def has_turn_steps(motion):
    feet = metrics.joints_from_motion(motion)[:, [25, 26, 21, 22]]
    lift = np.ptp(feet[:, :, 1], axis=0)
    return min(max(lift[:2]), max(lift[2:])) > .025 and max(lift) > .045


class PresetQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = json.loads((ROOT/'public/locomotion/bank.json').read_text())['clips']

    def test_native_structure_and_finite_data(self):
        for clip in self.clips:
            with self.subTest(clip=clip['name']):
                self.assertEqual(len(clip['bones']), 27)
                self.assertEqual(len(clip['localRot']), clip['frames']*81)
                self.assertEqual(len(clip['rootDelta']), clip['frames']*3)
                self.assertEqual(len(clip['footContacts']), clip['frames']*4)
                self.assertTrue(np.isfinite(clip['localRot']).all())
                self.assertTrue(np.isfinite(clip['rootDelta']).all())
                self.assertEqual(clip['sourceFrames'][1]-clip['sourceFrames'][0]+1, clip['frames'])

    def test_real_turn_steps_and_native_root_motion(self):
        for clip in self.clips:
            if clip['kind'] == 'turn':
                with self.subTest(clip=clip['name']):
                    self.assertTrue(has_turn_steps(clip), '旋转髋部不能冒充迈步转身')
                    self.assertGreater(clip['distance'], .05)
                    self.assertLess(clip['distance'], .65)

    def test_contact_flags_cannot_disguise_a_frozen_lower_body(self):
        clip = next(item for item in self.clips if item['kind'] == 'turn').copy()
        poses = np.asarray(clip['localRot']).reshape(-1, 27, 3)
        poses[:] = poses[0]
        angles = np.zeros((len(poses), 3)); angles[:, 1] = np.linspace(0, 1.5, len(poses))
        poses[:, 0] = Rotation.from_rotvec(angles).as_rotvec()
        clip['localRot'] = poses.ravel().tolist()
        clip['rootDelta'] = [0.]*(clip['frames']*3)
        clip['footContacts'] = ([1, 1, 0, 0, 0, 0, 1, 1]*(clip['frames']//2)
                                + [1, 1, 0, 0]*(clip['frames'] % 2))
        self.assertFalse(has_turn_steps(clip))


if __name__ == '__main__':
    unittest.main(verbosity=2)
