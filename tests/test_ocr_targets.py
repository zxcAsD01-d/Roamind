"""Image/depth frame contracts, session isolation, loss and ambiguity refusals."""
import copy
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from ocr_targets import TextTargets, text_surface
from stereo_depth import Calibration, DepthFrame
from ocr_tracking import track_text


class StereoFixture:
    def __init__(self):
        c = Calibration(320, 180, 240., 240., 160., 90., .12)
        z = np.full((180, 320), .8, np.float32)
        rgb = np.full((180, 320, 3), 120, np.uint8)
        self.frame = DepthFrame(rgb, rgb[:, :, 0], z, z > 0, np.zeros_like(z, np.uint8), c)
        self.timestamp, self.frame_id = time.time(), 1
        self.requested = []

    def capture_record(self, client, frame_id=None):
        self.requested.append((client, frame_id))
        if client != 'camera':
            raise ValueError('wrong camera')
        return self.frame, self.timestamp, frame_id or self.frame_id, np.eye(4)


class RecognizerFixture:
    def __init__(self):
        self.matches = [{'text': '确认', 'confidence': .99, 'box': [120, 60, 180, 120], 'center': [150, 90]}]

    def infer(self, body):
        return {'selected': self.matches[0] if len(self.matches) == 1 else None,
                'candidates': copy.deepcopy(self.matches)}


class TextTargetTests(unittest.TestCase):
    def test_visual_tracking_follows_pixels_and_refuses_occlusion(self):
        rng = np.random.default_rng(41)
        previous = rng.integers(0, 255, (180, 320), dtype=np.uint8)
        previous = cv2.GaussianBlur(previous, (3, 3), 0)
        match = {'box': [100, 50, 200, 130], 'confidence': .99, 'text': '确认'}
        transform = np.array([[1., 0, 4], [0, 1, -3]])
        current = cv2.warpAffine(previous, transform, (320, 180))
        tracked = track_text(previous, current, match)
        np.testing.assert_allclose(tracked['box'], [104, 47, 204, 127], atol=.5)
        current[45:132, 100:209] = 0
        with self.assertRaises(ValueError):
            track_text(previous, current, match)

    def setUp(self):
        self.stereo, self.ocr = StereoFixture(), RecognizerFixture()
        self.targets = TextTargets(self.stereo, self.ocr)

    def observe(self):
        return self.targets.observe({'clientId': 'camera', 'text': '确认', 'resolveDepth': True})['target']

    def advance(self):
        self.stereo.timestamp, self.stereo.frame_id = time.time(), 2
        for record in self.targets.targets.values():
            record['target']['observedAt'] -= .4

    def test_pixel_projection_and_same_frame(self):
        target = self.observe()
        np.testing.assert_allclose(target['position'], [-10*.8/240, 0, .8], atol=1e-4)
        self.advance()
        refreshed = self.targets.resolve({'clientId': 'camera', 'targetId': target['targetId']})
        self.assertEqual(refreshed['frameId'], 2)
        self.assertEqual(self.stereo.requested[-1], ('camera', 2))
        self.assertEqual(refreshed['targetId'], target['targetId'])

    def test_session_expiry_and_unseen_frame_refused(self):
        target = self.observe()
        with self.assertRaisesRegex(ValueError, '会话'):
            self.targets.resolve({'clientId': 'other', 'targetId': target['targetId']})
        with self.assertRaisesRegex(ValueError, '新双目帧'):
            self.targets.observe({'clientId': 'camera', 'text': '确认', 'afterObservedAt': time.time()+1})
        self.targets.targets[target['targetId']]['created'] -= 61
        with self.assertRaisesRegex(ValueError, '过期'):
            self.targets.resolve({'clientId': 'camera', 'targetId': target['targetId']})

    def test_lost_and_duplicate_targets_refused(self):
        for duplicates in (False, True):
            with self.subTest(duplicates=duplicates):
                self.setUp()
                target = self.observe()
                self.advance()
                self.ocr.matches = self.ocr.matches*2 if duplicates else []
                with self.assertRaisesRegex(ValueError, '丢失|不明确'):
                    self.targets.resolve({'clientId': 'camera', 'targetId': target['targetId']})

    def test_depth_search_stays_inside_text_and_refuses_missing_surface(self):
        match = self.ocr.matches[0]
        points = []
        def absent(frame, keypoint, rig):
            points.append(keypoint['pixel'])
            raise ValueError('无深度')
        with patch('ocr_targets.surface', side_effect=absent), self.assertRaisesRegex(ValueError, '可靠'):
            text_surface(self.stereo.frame, np.eye(4), match)
        self.assertEqual(points[0], match['center'])
        self.assertTrue(all(120 <= x <= 180 and 60 <= y <= 120 for x, y in points))

    def test_close_range_patch_does_not_mix_remote_background(self):
        frame = self.stereo.frame
        frame.depth[:] = 2.
        frame.depth[82:99, 142:159] = .8
        result = text_surface(frame, np.eye(4), self.ocr.matches[0])
        self.assertAlmostEqual(result['depth'], .8, places=5)
        np.testing.assert_allclose(result['normal'], [0, 0, -1], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
