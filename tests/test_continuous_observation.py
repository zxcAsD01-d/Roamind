"""Observation is an upper-body channel; valid visible routes start as one clip."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import runtime
from exploration_memory import ExplorationMemory
from social_navigation import SocialNavigator


class ContinuousObservationTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        runtime.sync({'clientId': 'head-test', 'state': {'heading': 0}})

    def test_head_controls_validate_range_and_duration(self):
        self.assertEqual(runtime.look({'clientId': 'head-test', 'relativeBearing': -15})['arguments']['action'], 'look')
        self.assertEqual(runtime.look({'clientId': 'head-test', 'relativeBearing': 75})['arguments']['relativeBearing'], 75)
        self.assertEqual(runtime.look({'clientId': 'head-test', 'relativeBearing': 7})['arguments']['relativeBearing'], 5)
        for value in (76, float('nan'), True, 'left'):
            with self.assertRaises(ValueError):
                runtime.look({'clientId': 'head-test', 'relativeBearing': value})
        with self.assertRaises(ValueError):
            runtime.look({'clientId': 'head-test', 'duration': 9})
        runtime.SESSIONS['head-test']['state'].update(playing=True, locomotion={'phase': 'walk'})
        with self.assertRaises(ValueError):
            runtime.look({'clientId': 'head-test', 'relativeBearing': 35})

    def test_small_novel_view_uses_head_while_wide_view_uses_body(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        memory.coverage.choose_view = lambda views, rotation, body_bearing=0, excluded=(): -25
        self.assertEqual(controller.survey(memory)['action'], 'look')
        self.assertEqual(controller.survey(memory, 35)['action'], 'look')
        self.assertTrue(controller.survey(memory, 65)['action'].startswith('turn_'))
        memory.coverage.choose_view = lambda views, rotation, body_bearing=0, excluded=(): 80
        self.assertTrue(controller.survey(memory)['action'].startswith('turn_'))

    def test_visible_off_axis_route_contains_turn_and_walk_in_one_command(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        navigation = {'tracking': True, 'bodyBearing': 0, 'maxDistance': 3, 'safetyMargin': .5,
                      'sectors': [{'angle': 30, 'clearance': 4, 'confidence': 1}]}
        route = {'goal': [4, 8], 'bearing': 30, 'distance': 2, 'intent': 'explore', 'relativePath': [[30, 2]]}
        result = controller.follow_guidance(navigation, memory, route, 100)
        self.assertEqual(result['action'], 'forward')
        self.assertEqual(result['relativePath'], [[30, 2]])

    def test_out_of_view_route_still_requires_body_observation(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        navigation = {'tracking': True, 'bodyBearing': 0, 'maxDistance': 3, 'safetyMargin': .5,
                      'sectors': [{'angle': 0, 'clearance': 4, 'confidence': 1}]}
        result = controller.follow_guidance(navigation, memory,
                 {'goal': [4, 8], 'bearing': 70, 'distance': 2, 'intent': 'explore'}, 100)
        self.assertEqual(result['action'], 'turn_right')

    def test_missing_floor_does_not_cause_repeated_body_alignment(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        memory.goal = (4, 8)
        result = controller.follow_guidance({'valid': False, 'tracking': True, 'sectors': []}, memory,
                 {'goal': [4, 8], 'bearing': 70, 'distance': 2, 'intent': 'explore'}, 100)
        self.assertTrue(result['wait'])
        self.assertEqual(result['intent'], 'inspect_route_floor')
        self.assertEqual(controller.alignment_turns, 0)
        self.assertEqual(memory.goal, (4, 8))

    def handoff(self, action, data):
        command = {'id': 'turn', 'kind': 'visual', 'status': 'playing', 'arguments': {'action': action}}
        session = {'state': {'playing': True, 'generating': False, 'bufferedSeconds': .5,
                            'locomotion': {'phase': 'turn', 'handoff': data}}}
        return runtime._visual_continuation({'afterCommand': 'turn'}, session, [command])

    def test_turn_can_prepare_successor_only_near_safe_end_pose(self):
        self.assertEqual(self.handoff('turn_left', {'headingError': 5, 'tailDistance': .04}), 'turn')
        for data in ({'headingError': 15, 'tailDistance': .04}, {'headingError': 5, 'tailDistance': .2}, {}):
            self.assertIsNone(self.handoff('turn_left', data))

    def test_head_prefetch_requires_new_semantic_exposure_after_settling(self):
        self.assertEqual(self.handoff('look', {'gazeReady': True}), 'turn')
        self.assertIsNone(self.handoff('look', {'gazeReady': False}))

    def test_recovery_returns_to_last_visual_heading_without_moving_body(self):
        controller = SocialNavigator()
        controller.last_tracking_view = 25
        result = controller.recover_view({'viewHeading': 35, 'bodyHeading': 0})
        self.assertEqual(result['action'], 'look')
        self.assertEqual(result['relativeBearing'], 10)
        self.assertEqual(result['distance'], 0)
        self.assertEqual(result['intent'], 'recovering_gaze')

    def test_recovery_turn_never_prefetches_another_recovery(self):
        command = {'id':'recovery','kind':'visual','status':'playing',
                   'arguments':{'action':'turn_left','intent':'recovering'}}
        session = {'state':{'playing':True,'generating':False,'bufferedSeconds':.2,
                  'locomotion':{'phase':'turn','handoff':{'headingError':1,'tailDistance':0}}}}
        self.assertIsNone(runtime._visual_continuation({'afterCommand':'recovery'}, session, [command]))

    def test_walk_prefetch_reserves_network_and_render_budget(self):
        command = {'id': 'walk', 'kind': 'visual', 'status': 'playing',
                   'arguments': {'action': 'forward', 'continuous': True}}
        session = {'state': {'playing': True, 'generating': False, 'bufferedSeconds': 3,
                            'locomotion': {'phase': 'walk', 'brakeIn': 1.5}}}
        self.assertEqual(runtime._visual_continuation({'afterCommand': 'walk'}, session, [command]), 'walk')
        session['state']['locomotion']['brakeIn'] = 2
        self.assertIsNone(runtime._visual_continuation({'afterCommand': 'walk'}, session, [command]))
        session['state']['locomotion']['brakeIn'] = True
        self.assertIsNone(runtime._visual_continuation({'afterCommand': 'walk'}, session, [command]))


if __name__ == '__main__':
    unittest.main()
