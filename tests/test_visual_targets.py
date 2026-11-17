"""Generic annotation geometry, exact pixel tracking and fail-closed lifetime."""
import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from visual_targets import VisualTargets
from stereo_depth import Calibration, DepthFrame
from ocr_tracking import track_region
from contact_tracking import ContactTracker


class StereoFixture:
    def __init__(self):
        c = Calibration(320, 180, 240., 240., 160., 90., .12)
        gray = np.random.default_rng(32).integers(0, 255, (180, 320), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        z = np.full((180, 320), .8, np.float32)
        self.frame = DepthFrame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, z, z > 0, np.zeros_like(gray), c)
        self.timestamp, self.frame_id, self.rig = time.time(), 1, np.eye(4)

    def capture_record(self, client, frame_id=None):
        if client != 'camera' or frame_id is not None and frame_id != self.frame_id:
            raise ValueError('会话或同帧图像不存在')
        return self.frame, self.timestamp, self.frame_id, self.rig

    def advance(self, dx=0, dy=0):
        transform = np.array([[1., 0, dx], [0, 1, dy]])
        self.frame.gray[:] = cv2.warpAffine(self.frame.gray, transform, (320, 180))
        self.frame.rgb[:] = cv2.cvtColor(self.frame.gray, cv2.COLOR_GRAY2BGR)
        self.timestamp, self.frame_id = time.time(), self.frame_id+1


class VisualTargetTests(unittest.TestCase):
    def setUp(self):
        self.stereo = StereoFixture()
        self.targets = VisualTargets(self.stereo)

    def select(self, **args):
        return self.targets.select({'clientId': 'camera', 'pixel': [120., 80.], 'frameId': 1, 'label': '任意表面', **args})

    def resolve(self, target, **args):
        return self.targets.resolve({'clientId': 'camera', 'targetId': target['targetId'], **args})

    def test_any_surface_without_person_or_text_and_nonidentity_rig(self):
        self.stereo.rig = np.array([[0., 0., 1., 3.], [0., -1., 0., 1.5], [1., 0., 0., -2.], [0., 0., 0., 1.]])
        target = self.select()
        optical = np.array([-40*.8/240, -10*.8/240, .8])
        np.testing.assert_allclose(target['position'], self.stereo.rig[:3, :3]@optical+self.stereo.rig[:3, 3], atol=1e-4)
        self.assertNotIn('personId', target)
        self.assertEqual(target['label'], '任意表面')
        self.assertEqual(ContactTracker(target).update(target)['trackingSamples'], 1)
        self.assertTrue(self.targets.frame('camera')['image'].startswith('data:image/jpeg;'))

    def test_exact_selected_pixel_follows_affine_not_box_center(self):
        match = {'box': [90, 50, 180, 120], 'pixel': [115., 95.]}
        transform = cv2.getRotationMatrix2D((135, 85), 4, 1.02)
        transform[:, 2] += [3, -2]
        current = cv2.warpAffine(self.stereo.frame.gray, transform, (320, 180))
        result = track_region(self.stereo.frame.gray, current, match)
        expected = cv2.transform(np.array([[match['pixel']]], np.float32), transform)[0, 0]
        np.testing.assert_allclose(result['pixel'], expected, atol=.5)
        self.assertGreater(np.linalg.norm(np.array(result['pixel'])-result['center']), 10)

    def test_capture_cadence_tracks_without_action_polling(self):
        target = self.select()
        for _ in range(5):
            self.stereo.advance(1, -1)
            self.targets.update('camera')
        result = self.resolve(target)
        np.testing.assert_allclose(result['requestedPixel'], [125, 75], atol=.5)
        self.assertEqual(result['frameId'], 6)
        self.assertEqual(result['targetId'], target['targetId'])

    def test_occlusion_invalidates_token_and_does_not_reacquire_background(self):
        target = self.select()
        self.stereo.advance()
        self.stereo.frame.gray[:] = 0
        self.targets.update('camera')
        with self.assertRaisesRegex(ValueError, '丢失'):
            self.resolve(target)
        self.stereo.advance()
        self.stereo.frame.gray[:] = np.random.default_rng(32).integers(0, 255, (180, 320), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, '丢失'):
            self.resolve(target)

    def test_session_expiry_stale_reset_and_missing_frame(self):
        target = self.select()
        with self.assertRaisesRegex(ValueError, '会话'):
            self.resolve(target, clientId='other')
        with self.assertRaises(ValueError):
            self.select(frameId=999)
        self.stereo.timestamp -= 2
        with self.assertRaisesRegex(ValueError, '失效'):
            self.resolve(target)
        self.stereo.timestamp = time.time()
        with self.assertRaisesRegex(ValueError, '失效'):
            self.resolve(target)
        self.targets.targets[target['targetId']]['created'] -= 61
        with self.assertRaisesRegex(ValueError, '过期'):
            self.resolve(target)

    def test_bad_depth_calibration_and_input_refused(self):
        for args in ({'pixel': [float('nan'), 80]}, {'pixel': [True, 80]}, {'pixel': [-10, 80]},
                     {'frameId': None}, {'label': ''}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.select(**args)
        self.stereo.rig = None
        with self.assertRaisesRegex(ValueError, '标定'):
            self.select()
        self.stereo.rig = np.eye(4)
        self.stereo.frame.valid[:] = False
        with self.assertRaises(ValueError):
            self.select()

    def test_large_spatial_jump_and_target_identity_swap_refused(self):
        target = self.select()
        self.stereo.advance()
        self.stereo.rig[0, 3] += .2
        with self.assertRaisesRegex(ValueError, '范围'):
            self.resolve(target)
        with self.assertRaisesRegex(ValueError, '身份'):
            ContactTracker(target).update({**target, 'targetId': 'another'})

    def test_reset_or_release_invalidates_only_own_annotations(self):
        target = self.select()
        self.targets.clear('other-camera')
        self.assertEqual(self.resolve(target)['targetId'], target['targetId'])
        self.targets.clear('camera')
        with self.assertRaisesRegex(ValueError, '过期'):
            self.resolve(target)


if __name__ == '__main__':
    unittest.main()
