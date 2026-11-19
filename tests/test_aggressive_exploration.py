import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from behavior import BehaviorPolicy
from exploration_memory import ExplorationMemory
from social_navigation import SocialNavigator
from visual_navigation import exploratory_navigation


def observation(stamp, heading=0):
    return {'observedAt': stamp, 'viewHeading': heading, 'bodyHeading': heading,
            'signature': [.2]*64, 'objects': [], 'semanticReady': False,
            'stereo': {'method': 'stereo-sgbm-lr', 'observedAt': stamp, 'mapping': False, 'sectors': []}}


class AggressiveExplorationTests(unittest.TestCase):
    def setUp(self):
        self.memory, self.navigator = ExplorationMemory(), SocialNavigator()
        self.nav = {**exploratory_navigation(None), 'bodyBearing': 0}

    def step(self, stamp=100, heading=0, frame=None, route=None):
        return self.navigator.step(frame or observation(stamp, heading), self.nav, self.memory, route or {}, stamp)

    def test_unknown_stale_floorless_depth_does_not_limit_travel(self):
        policy = BehaviorPolicy()
        for stereo in ({}, {'method': 'stereo-sgbm-lr', 'observedAt': time.time()-10, 'mapping': False}):
            frame = {**observation(time.time()), 'stereo': stereo}
            result = policy.decide({'clientId': 'aggressive', 'vision': frame})
            self.assertEqual(result['action'], 'forward')
            self.assertEqual(result['distance'], 3)
            self.assertFalse(result['navigation']['visualAvoidance'])

    def test_short_measured_clearance_is_not_a_ten_centimetre_command(self):
        frame = observation(time.time())
        frame['stereo'].update(mapping=True, sectors=[{'angle': 0, 'clearance': .8, 'confidence': .9}])
        result = BehaviorPolicy().decide({'clientId': 'door', 'vision': frame})
        self.assertEqual(result['action'], 'forward')
        self.assertEqual(result['distance'], 3)

    def test_image_door_is_preferred_even_without_depth(self):
        frame = observation(100)
        frame['objects'] = [{'label': 'door', 'box': [50, 0, 190, 350], 'confidence': .8}]
        result = self.step(frame=frame)
        self.assertEqual(result['intent'], 'seek_possible_exit')
        self.assertLess(result['relativeBearing'], -20)
        self.assertEqual(result['distance'], 3)
        self.assertIsNone(self.memory.position)

    def test_recovery_finishes_half_turn_then_leaves_without_observation_barrier(self):
        self.step()
        self.navigator.feedback('forward', 'failed', self.memory, 101, '播放碰撞：墙')
        result = self.step(102)
        self.assertEqual(result['intent'], 'escape_deadlock')
        self.assertEqual(result['action'], 'turn_right')
        self.assertAlmostEqual(result['relativeBearing'], 180)
        result = self.step(105, -180)
        self.assertEqual(result['intent'], 'leave_deadlock')
        self.assertEqual(result['action'], 'forward')
        self.assertEqual(result['distance'], 1.5)
        self.assertLess(abs(result['relativeBearing']), 1)
        self.assertTrue(self.navigator.drive.blocked(0, self.memory, 105))
        self.assertFalse(self.navigator.drive.blocked(0, self.memory, 162))

    def test_progress_watchdog_failure_triggers_same_turnaround(self):
        self.step()
        self.navigator.feedback('forward', 'failed', self.memory, 101, '移动死锁：无位移')
        self.assertEqual(self.step(102)['intent'], 'escape_deadlock')

    def test_blocked_turn_reverses_turn_direction(self):
        self.step()
        self.memory.scan_sign = 1
        self.navigator.feedback('turn_right', 'failed', self.memory, 101, '预设根运动无法安全通过：侧墙')
        result = self.step(102)
        self.assertEqual(result['action'], 'turn_left')
        self.assertEqual(result['intent'], 'escape_deadlock')

    def test_user_cancel_and_quality_failure_do_not_invent_physical_blockage(self):
        self.step()
        self.navigator.feedback('forward', 'interrupted', self.memory, 101, '用户取消')
        self.navigator.feedback('forward', 'failed', self.memory, 101, '基础步态质量未通过：')
        self.assertIsNone(self.navigator.drive.back_heading)
        self.assertEqual(self.navigator.drive.rejected, [])

    def test_ten_centimetre_first_leg_keeps_complete_two_metre_route(self):
        self.memory.position, self.memory.rotation = np.zeros(3), np.eye(3)
        route = {'available': True, 'goal': [0, 6], 'bearing': 0, 'distance': .1,
                 'intent': 'navigate_region_exit', 'relativePath': [[0, .1], [0, 2]]}
        self.assertTrue(self.navigator.travel_ready(self.nav, route))
        result = self.step(route=route)
        self.assertEqual(result['relativePath'], route['relativePath'])
        self.assertAlmostEqual(result['routeLength'], 2)
        self.assertTrue(result['continuous'])

    def test_confirmed_route_takes_precedence_over_a_speculative_door(self):
        self.memory.position, self.memory.rotation = np.zeros(3), np.eye(3)
        frame = observation(100.)
        frame['objects'] = [{'label': 'door', 'box': [50, 0, 190, 350], 'confidence': .8}]
        route = {'available': True, 'goal': [0, 6], 'bearing': 0., 'distance': 2.,
                 'intent': 'explore_frontier', 'relativePath': [[0., 2.]]}
        result = self.step(frame=frame, route=route)
        self.assertEqual(result['intent'], 'explore_frontier')
        self.assertEqual(result['relativePath'], route['relativePath'])

    def test_fallback_probe_does_not_follow_its_previous_heading_back_along_the_trail(self):
        self.memory.position, self.memory.rotation = np.zeros(3), np.eye(3)
        for i in range(7):
            self.memory.exit_memory.observe(np.array([0., 0., 3.-i*.5]), 90.+i)
        frame = observation(100.)
        frame['stereo']['mapping'] = True
        self.navigator.drive.commit_heading, self.navigator.drive.commit_until = 0., 110.
        result = self.step(frame=frame)
        self.assertEqual(result['action'], 'forward')
        self.assertGreaterEqual(abs(result['relativeBearing']), 60.)

    def test_head_body_difference_does_not_rotate_unknown_forward_motion(self):
        frame = {**observation(100), 'viewHeading': 55, 'bodyHeading': 20}
        self.nav['bodyBearing'] = 35
        result = self.step(frame=frame)
        self.assertEqual(result['relativeBearing'], 35)
        self.assertTrue(math.isclose(self.navigator.drive.requested_heading, 20))
