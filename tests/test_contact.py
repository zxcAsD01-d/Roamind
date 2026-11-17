import copy
import json
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server'))
from contact_model import EFFECTORS, capabilities, check_support
from contact_tracking import ContactTracker
from touch_ik import forward, solve_axial
from touch_motion import build_plan, options
from touch_vision import body_pixel, follow_selection
from library_motion import joints


def target(position=None):
    return {'position': position or [0., 1., 0.], 'normal': [0., 0., -1.], 'tangent': [1., 0., 0.],
            'personId': 'same', 'frame': 'control-world-y-up', 'targetId': 'token',
            'observedAt': time.time()-.5, 'frameId': 1, 'sigmaMeters': .01, 'localSurfaceRadius': .03}


class ContactTrackingTests(unittest.TestCase):
    def test_slow_generation_reacquires_but_playback_gap_still_stops(self):
        import touch
        initial = target()
        current = {**initial, 'observedAt': time.time(), 'frameId': 2}
        for status, moved, succeeds in [('generating', .02, True), ('generating', .07, False), ('ready', .02, False)]:
            with self.subTest(status=status, moved=moved):
                tracker = ContactTracker(initial)
                tracker.last['observedAt'] -= 2
                job = {'status': status, 'tracker': tracker}
                plan = {'id': 'slow-test', 'touch': {'clientId': 'camera', 'target': initial}}
                observed = {**current, 'position': [moved, 1., 0.]}
                with patch.dict(touch.JOBS, {'slow-test': job}), patch('touch.resolve', return_value=observed):
                    if succeeds:
                        result = touch.validate_live(plan)
                        self.assertEqual(result['trackingSamples'], 1)
                        np.testing.assert_array_equal(result['velocity'], [0, 0, 0])
                    else:
                        with self.assertRaises(ValueError):
                            touch.validate_live(plan)

    def test_follows_more_than_old_six_cm_without_identity_change(self):
        initial = target(); tracker = ContactTracker(initial)
        result = tracker.update({**initial, 'position': [.09, 1., 0.],
                                 'observedAt': initial['observedAt']+.4, 'frameId': 2})
        self.assertGreater(result['position'][0], .06)
        self.assertEqual(result['trackingSamples'], 2)
        self.assertLessEqual(np.linalg.norm(result['velocity']), .45)

    def test_repeated_or_out_of_order_frame_does_not_train_filter(self):
        initial = target(); tracker = ContactTracker(initial)
        self.assertEqual(tracker.update(initial)['trackingSamples'], 1)
        stale = {**initial, 'position': [.01, 1., 0.], 'frameId': 0, 'observedAt': initial['observedAt']-.1}
        self.assertEqual(tracker.update(stale)['trackingSamples'], 1)
        np.testing.assert_array_equal(tracker.position, initial['position'])

    def test_rejects_identity_jump_normal_stale_and_large_motion(self):
        initial = target()
        for change in ({'personId': 'other'}, {'normal': [0., 1., 0.]}, {'observedAt': time.time()-2},
                       {'position': [.181, 1., 0.]}, {'sigmaMeters': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ContactTracker(initial).update({**initial, **change, 'frameId': 2})

    def test_outlier_speed_is_rejected(self):
        initial = target()
        with self.assertRaisesRegex(ValueError, '跳变'):
            ContactTracker(initial).update({**initial, 'position': [.14, 1., 0.], 'frameId': 2,
                                            'observedAt': initial['observedAt']+.01})

    def test_noisy_measurement_gets_less_weight_than_precise_measurement(self):
        initial = target()
        update = {**initial, 'position': [.03, 1., 0.], 'frameId': 2, 'observedAt': initial['observedAt']+.2}
        precise = ContactTracker(initial).update(update)
        noisy = ContactTracker(initial).update({**update, 'sigmaMeters': .05})
        self.assertGreater(precise['position'][0], noisy['position'][0])


class ContactAnatomyTests(unittest.TestCase):
    def person(self):
        return {'keypoints': [{'name': name, 'pixel': pixel, 'confidence': .9} for name, pixel in
                             [('left_hip', [40, 20]), ('left_knee', [40, 100]),
                              ('nose', [120, 60]), ('left_eye', [110, 40]), ('right_eye', [130, 40])]]}

    def test_limb_click_keeps_actual_surface_offset_and_scales_with_limb(self):
        person = self.person()
        point, selection = body_pixel(person, {'pixel': [44., 60.]})
        np.testing.assert_allclose(point['pixel'], [44, 60])
        person['keypoints'][1]['pixel'] = [40, 180]
        np.testing.assert_allclose(follow_selection(person, selection)['pixel'], [48, 100])

    def test_thigh_is_explicit_inference_not_a_detected_joint(self):
        point, selection = body_pixel(self.person(), {'bodyPart': 'left_thigh'})
        np.testing.assert_allclose(point['pixel'], [40, 60])
        self.assertEqual(selection['estimate'], 'inferred-limb-region')

    def test_face_surface_follows_triangle_not_head_top(self):
        person = self.person()
        point, selection = body_pixel(person, {'pixel': [120, 45]})
        np.testing.assert_allclose(point['pixel'], [120, 45])
        self.assertEqual(selection['estimate'], 'inferred-visible-face-surface')
        for p in person['keypoints'][2:]:
            p['pixel'][0] += 5
        np.testing.assert_allclose(follow_selection(person, selection)['pixel'], [125, 45])
        with self.assertRaises(ValueError):
            body_pixel(person, {'pixel': [120, 15]})

    def test_occluded_face_anchor_rejected(self):
        person = self.person(); _, selection = body_pixel(person, {'pixel': [120, 45]})
        person['keypoints'][2]['confidence'] = .2
        with self.assertRaisesRegex(ValueError, '遮挡'):
            follow_selection(person, selection)


class ContactKinematicsTests(unittest.TestCase):
    def test_all_five_effectors_and_legacy_hand_options(self):
        for name in EFFECTORS:
            self.assertEqual(options({'effector': name})['effector'], name)
        self.assertEqual(options({'hand': 'left'})['effector'], 'left_hand')
        with self.assertRaises(ValueError):
            options({'effector': 'tentacle'})

    def test_center_clearance_not_double_counted_by_body_mapping(self):
        history = json.loads((ROOT/'public/locomotion/bank.json').read_text())['clips'][0]
        posed = joints(history)[-1]
        own = {key: posed[index].tolist() for key, index in [('base', 8), ('middle', 9), ('end', 10)]}
        visual = target((posed[10]+[0, .02, .04]).tolist())
        plan = build_plan(history, visual, options({}), 'test', own)
        begin = plan['touch']['contactFrames'][0]
        expected = np.asarray(visual['position'])+[0, 0, -.035]
        np.testing.assert_allclose(plan['contactTrack'][begin], expected)
        np.testing.assert_allclose(plan['handTrack'][begin], expected)
        np.testing.assert_allclose(plan['rootTrack'][0], plan['rootTrack'][-1])

    def test_foot_requires_supported_pelvis_projection(self):
        posed = np.zeros((27, 3)); posed[0] = [0, .92, 0]; posed[26] = [0, 0, .14]
        check_support(posed, EFFECTORS['right_foot'])
        posed[0, 0] = .2
        with self.assertRaisesRegex(ValueError, '支撑范围'):
            check_support(posed, EFFECTORS['right_foot'])

    def test_head_solver_bends_only_chest_and_neck_with_bounds(self):
        from constrained import PARENTS
        rest = np.zeros((27, 3)); rest[3] = [0, .2, 0]; rest[4] = [0, .3, 0]
        rest[5] = [0, .4, 0]; rest[6] = [0, .5, 0]
        local = np.zeros(81)
        solve_axial(local, np.zeros(3), np.array([0, .495, .04]), rest, PARENTS, EFFECTORS['head'], np.zeros(81))
        posed, _ = forward(local, np.zeros(3), rest, PARENTS)
        self.assertLess(np.linalg.norm(posed[6]-[0, .495, .04]), .015)
        np.testing.assert_array_equal(np.delete(local.reshape(27, 3), [3, 4, 5], axis=0), 0)
        self.assertLessEqual(Rotation.from_rotvec(local[9:12]).magnitude(), .221)

    def test_concept_names_cannot_bypass_missing_grasp_or_load_constraints(self):
        report = capabilities()
        for concept in ('head_pat', 'lap_pillow', 'ear_tug', 'clap'):
            self.assertTrue(report['concepts'][concept]['missing'])
            with self.assertRaises(ValueError):
                options({'operation': concept})


if __name__ == '__main__':
    unittest.main(verbosity=2)
