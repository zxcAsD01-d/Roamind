import copy
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from stereo_depth import Calibration, DepthFrame
from touch_vision import BodyTargets, body_pixel, read_rig, surface
from touch_motion import operation_track, options, target_unchanged, validate_hand, map_to_source
from touch_ik import forward, solve_frame
from touch_ik import bounded_rotation


class TouchVisionTests(unittest.TestCase):
    def setUp(self):
        c = Calibration(320, 180, 240., 240., 160., 90., .12)
        depth = np.full((180, 320), .8, np.float32)
        rgb = np.full((180, 320, 3), 120, np.uint8)
        self.frame = DepthFrame(rgb, rgb[:, :, 0], depth, depth > 0, np.zeros_like(depth, np.uint8), c)
        self.rig = np.array([[0., 0., 1., 3.], [0., -1., 0., 1.5], [1., 0., 0., -2.], [0., 0., 0., 1.]])
        self.person = {'confidence': .95, 'box': [80, 30, 240, 170], 'opticalPosition': [0, 0, .8],
                       'keypoints': [{'name': 'left_shoulder', 'pixel': [140, 70], 'confidence': .9},
                                     {'name': 'left_elbow', 'pixel': [150, 90], 'confidence': .9},
                                     {'name': 'left_wrist', 'pixel': [170, 110], 'confidence': .9}]}
        for point in self.person['keypoints']:
            point.update(valid=True, depth=.8)

    def tracker(self):
        tracker = BodyTargets()
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), time.time(), 1, self.rig)
        return tracker, people[0]['id']

    def test_nonidentity_camera_transform_and_normal(self):
        target = surface(self.frame, self.person['keypoints'][1], self.rig)
        optical = np.array([-10*.8/240, 0, .8])
        np.testing.assert_allclose(target['position'], self.rig[:3, :3]@optical+self.rig[:3, 3], atol=.0001)
        np.testing.assert_allclose(target['normal'], [-1, 0, 0], atol=.0001)

    def test_calibration_rejects_reflection_scale_nan(self):
        for matrix in (np.diag([1., 1., -1., 1.]), np.diag([2., 1., 1., 1.]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                read_rig({'source': 'ego-kinematics', 'opticalToControl': matrix.ravel().tolist()})

    def test_invalid_center_cannot_borrow_background(self):
        self.frame.valid[87:94, 147:154] = False
        with self.assertRaisesRegex(ValueError, '中心'):
            surface(self.frame, self.person['keypoints'][1], self.rig)

    def test_small_matching_hole_uses_only_two_pixel_support(self):
        self.frame.valid[90, 150] = False
        result = surface(self.frame, self.person['keypoints'][1], self.rig)
        self.assertLessEqual(np.linalg.norm(np.asarray(result['pixel'])-[150, 90]), 2)
        self.assertEqual(result['requestedPixel'], [150, 90])

    def test_center_on_different_depth_rejected(self):
        self.frame.depth[90, 150] = 2.
        with self.assertRaisesRegex(ValueError, '边缘'):
            surface(self.frame, self.person['keypoints'][1], self.rig)

    def test_pixel_outside_body_support_rejected(self):
        with self.assertRaisesRegex(ValueError, '连线'):
            body_pixel(self.person, {'pixel': [230, 60]})
        _, selection = body_pixel(self.person, {'pixel': [160, 100]})
        self.assertEqual(selection['segment'], ['left_elbow', 'left_wrist'])
        self.assertAlmostEqual(selection['fraction'], .5)

    def test_body_target_reobserved_and_person_lost(self):
        tracker, identity = self.tracker()
        target = tracker.resolve({'personId': identity, 'bodyPart': 'left_elbow'})
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), time.time(), 2, self.rig)
        self.assertEqual(people[0]['id'], identity)
        current = tracker.resolve({'targetId': target['targetId']})
        self.assertEqual(current['frameId'], 2)
        tracker.update([], self.frame, np.eye(4), time.time(), 3, self.rig)
        with self.assertRaisesRegex(ValueError, '丢失'):
            tracker.resolve({'targetId': target['targetId']})

    def test_person_identity_ignores_simulator_world_translation(self):
        tracker, identity = self.tracker()
        rig = self.rig.copy()
        rig[:3, 3] += [100, 200, 300]
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), time.time(), 2, rig)
        self.assertEqual(people[0]['id'], identity)
        people = tracker.update([copy.deepcopy(self.person)], self.frame, None, time.time(), 3, rig)
        self.assertNotEqual(people[0]['id'], identity)

    def test_slow_semantic_frame_keeps_unique_identity_but_not_stale_touch(self):
        tracker, identity = self.tracker()
        stamp = tracker.record[2]+1.8
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), stamp, 2, self.rig)
        self.assertEqual(people[0]['id'], identity)
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), stamp+3, 3, self.rig)
        self.assertNotEqual(people[0]['id'], identity)

    def test_ambiguous_nearby_people_not_associated(self):
        tracker, identity = self.tracker()
        second = copy.deepcopy(tracker.people[0]); second['id'] = 'other'; tracker.people.append(second)
        people = tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), time.time(), 2, self.rig)
        self.assertNotIn(people[0]['id'], (identity, 'other'))

    def test_recent_clicked_frame_and_expired_frame(self):
        tracker, identity = self.tracker()
        tracker.update([copy.deepcopy(self.person)], self.frame, np.eye(4), time.time(), 2, self.rig)
        target = tracker.resolve({'personId': identity, 'pixel': [160, 100], 'frameId': 1})
        self.assertEqual(target['frameId'], 1)
        with self.assertRaisesRegex(ValueError, '过期'):
            tracker.resolve({'personId': identity, 'pixel': [160, 100], 'frameId': 999})

    def test_stale_or_unregistered_target_rejected(self):
        tracker, identity = self.tracker()
        with self.assertRaisesRegex(ValueError, '过期'):
            tracker.resolve({'targetId': 'invented'})
        tracker.record = (*tracker.record[:2], time.time()-2, *tracker.record[3:])
        with self.assertRaisesRegex(ValueError, '过期'):
            tracker.resolve({'personId': identity, 'bodyPart': 'left_elbow'})


class TouchMotionTests(unittest.TestCase):
    def target(self):
        return {'position': [.3, 1., .2], 'normal': [0, 0, -1.], 'tangent': [1., 0, 0],
                'localSurfaceRadius': .03, 'personId': 'person', 'frame': 'control-world-y-up'}

    def test_operations_return_to_start_with_bounded_speed(self):
        start = np.array([.1, .9, .1])
        for operation in ('touch', 'tap', 'hold', 'stroke'):
            track, (begin, end) = operation_track(start, self.target(), options({'operation': operation}))
            np.testing.assert_allclose(track[-1], start)
            np.testing.assert_allclose(track[begin-1], track[begin], atol=.0001)
            np.testing.assert_allclose(track[end-1], track[end], atol=.001)
            self.assertLess(np.linalg.norm(np.diff(track, axis=0), axis=1).max()*20, .7)

    def test_invalid_operation_and_nonfinite_options(self):
        for body in ({'operation': 'invent'}, {'hand': 'third'}, {'duration': float('nan')}, {'amplitude': .2}):
            with self.assertRaises(ValueError):
                options(body)

    def test_stroke_stays_in_observed_surface(self):
        with self.assertRaisesRegex(ValueError, '范围'):
            operation_track(np.array([.1, .9, .1]), self.target(), options({'operation': 'stroke', 'amplitude': .04}))

    def test_motion_or_identity_change_interrupts(self):
        initial = self.target()
        for current in ({**initial, 'position': [.5, 1, .2]}, {**initial, 'personId': 'other'}, {**initial, 'normal': [0, 1, 0]}):
            with self.assertRaises(ValueError):
                target_unchanged(initial, current)

    def test_final_hand_check_uses_actual_output_and_frame_offset(self):
        plan = {'handTrack': [[i*.01, 1., 0] for i in range(40)]}
        expected = np.asarray(plan['handTrack'])[20:30]
        report = validate_hand(expected, plan, 20)
        self.assertEqual(report['handErrorMaxMeters'], 0)
        with self.assertRaisesRegex(ValueError, '拒绝播放'):
            validate_hand(expected+[.1, 0, 0], plan, 20)

    def test_own_limb_calibration_preserves_target_direction_and_fraction(self):
        arm = {'shoulder': [2, 1, 3], 'elbow': [2, .8, 3], 'hand': [2, .6, 3]}
        target = {**self.target(), 'position': [2, 1, 3.2]}
        mapped = map_to_source(target, arm, np.array([0, 1.4, 0]), .6)
        np.testing.assert_allclose(mapped['position'], [0, 1.4, .3])
        np.testing.assert_allclose(target['position'], [2, 1, 3.2])

    def test_invalid_own_limb_calibration_is_rejected(self):
        for arm in ({}, {'shoulder': [0, 0, 0], 'elbow': [0, 0, 0], 'hand': [0, 0, 0]}):
            with self.assertRaises(ValueError):
                map_to_source(self.target(), arm, np.array([0, 1.4, 0]), .6)

    def test_two_segment_solver_both_hands_preserves_other_joints(self):
        from constrained import PARENTS
        rest = np.zeros((27, 3))
        rest[8], rest[9], rest[10] = [-.15, 1.4, 0], [-.35, 1.4, 0], [-.55, 1.4, 0]
        rest[14], rest[15], rest[16] = [.15, 1.4, 0], [.35, 1.4, 0], [.55, 1.4, 0]
        for hand, upper, lower, target in [(10, 8, 9, [-.30, 1.2, .10]), (16, 14, 15, [.30, 1.2, .10])]:
            local = np.zeros(81); root = np.zeros(3)
            before, _ = forward(local, root, rest, PARENTS)
            solve_frame(local, root, np.array(target), rest, PARENTS, hand, None)
            posed, _ = forward(local, root, rest, PARENTS)
            np.testing.assert_allclose(posed[hand], target, atol=1e-6)
            other = [j for j in range(27) if j not in (lower, hand, hand+1, hand+2)]
            np.testing.assert_allclose(posed[other], before[other], atol=1e-6)
            unchanged = np.delete(local.reshape(27, 3), [upper, lower], axis=0)
            np.testing.assert_allclose(unchanged, 0)

    def test_rotation_limiter_handles_wrist_flip(self):
        from scipy.spatial.transform import Rotation
        result = bounded_rotation(np.zeros(3), np.array([0., 0., 2.1]), .18)
        self.assertAlmostEqual(Rotation.from_rotvec(result).magnitude(), .18)


if __name__ == '__main__':
    unittest.main(verbosity=2)
