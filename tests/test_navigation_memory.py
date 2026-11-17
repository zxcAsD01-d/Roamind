"""Persistent directional search and frontier candidates use visual memory only."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from search_coverage import SearchCoverage
from exploration_memory import ExplorationMemory
from visual_navigation import depth_navigation, safe_travel, body_bearing
from social_navigation import SocialNavigator, person_observation
from visual_step import visual_step
from unittest.mock import patch


class SearchCoverageTests(unittest.TestCase):
    def setUp(self):
        self.memory = SearchCoverage()
        self.position, self.rotation = np.zeros(3), np.eye(3)
        self.sectors = [{'angle': angle, 'clearance': 3, 'confidence': .9} for angle in range(-45, 46, 15)]

    def observe(self, yaw, stamp=100):
        angle = math.radians(yaw)
        rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        self.memory.observe(self.position, rotation, self.sectors, stamp)

    def test_same_view_does_not_increase_coverage_and_next_view_avoids_it(self):
        for stamp in range(100, 120):
            self.observe(0, stamp)
        self.assertEqual(len(self.memory.views_at(self.position)), 7)
        self.assertGreaterEqual(abs(self.memory.next_view(self.position, self.rotation)), 45)

    def test_confirmation_survives_return_and_stops_repeat_scans(self):
        for yaw in range(0, 360, 45):
            self.observe(yaw)
        self.assertIsNone(self.memory.next_view(self.position, self.rotation))
        self.assertTrue(self.memory.snapshot()['regions'][0]['confirmed'])
        self.assertEqual(self.memory.views_at(np.array([.05, 0, -.05])), set(range(24)))
        self.assertIsNotNone(self.memory.next_view(np.array([3., 0, 0]), self.rotation))

    def test_unknown_depth_does_not_confirm_but_observed_wall_does(self):
        self.memory.observe(self.position, self.rotation, [{'angle': 0, 'clearance': None, 'confidence': 1}], 100)
        self.assertEqual(len(self.memory.views_at(self.position)), 0)
        self.memory.observe(self.position, self.rotation, [{'angle': 0, 'clearance': 0, 'confidence': 1}], 101)
        self.assertEqual(self.memory.views_at(self.position), {0})

    def test_near_exit_records_view_without_removing_unverified_exit(self):
        self.memory.mark_exit([.1, .1], 2, 100)
        self.observe(180)
        self.assertEqual(next(iter(self.memory.exits.values()))['status'], 'unvisited')
        self.observe(0, 101)
        exit = next(iter(self.memory.exits.values()))
        self.assertEqual(exit['status'], 'unvisited')
        self.assertEqual(exit['lookedAt'], 101)

    def test_long_route_looks_ahead_but_never_crosses_unobserved_cells(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        memory.cells = {(x, z): (-2., 100., 1) for x in range(-2, 3) for z in range(-2, 12)}
        route = memory.route_to((0, 10), 'explore_frontier')
        self.assertGreater(route['distance'], 2.5)
        for x in range(-2, 3):
            del memory.cells[(x, 5)]
        memory.map_revision += 1
        route = memory.route_to((0, 10), 'explore_frontier')
        self.assertIsNone(route)

    def test_direction_and_exit_memory_are_bounded(self):
        for index in range(2200):
            position = np.array([index*2., 0, 0])
            self.memory.observe(position, self.rotation, self.sectors, 100+index)
            self.memory.mark_exit([index*2., 1], 2, 100+index)
        self.assertLessEqual(len(self.memory.regions), 2048)
        self.assertLessEqual(len(self.memory.exits), 512)

    def test_turned_head_checks_travel_sector_instead_of_camera_center(self):
        stereo = {'method': 'stereo-sgbm-lr', 'observedAt': 100, 'mapping': True,
                  'sectors': [{'angle': 0, 'clearance': 4., 'confidence': 1},
                              {'angle': 30, 'clearance': .6, 'confidence': 1}]}
        with patch('time.time', return_value=100):
            navigation = depth_navigation(stereo)
        self.assertGreater(safe_travel(navigation, 0), 2.)
        self.assertEqual(safe_travel(navigation, 30), 0)
        self.assertEqual(safe_travel(navigation, 70), 0)
        self.assertAlmostEqual(body_bearing({'viewHeading': -170, 'bodyHeading': 170}), 20)

    def test_remembered_corridor_extends_floor_horizon_but_never_obstacle_or_stale_depth(self):
        stereo = {'method': 'stereo-sgbm-lr', 'observedAt': 100, 'mapping': True,
                  'sectors': [{'angle': 0, 'clearance': 1.9, 'confidence': 1, 'obstacleDistance': None}]}
        with patch('time.time', return_value=100):
            navigation = depth_navigation(stereo)
        self.assertLess(safe_travel(navigation, 0), 1.2)
        self.assertEqual(safe_travel(navigation, 0, 2.8), 2.8)
        navigation['sectors'][0]['obstacleDistance'] = 1.5
        self.assertLess(safe_travel(navigation, 0, 2.8), .8)
        with patch('time.time', return_value=102):
            self.assertEqual(safe_travel(depth_navigation(stereo), 0, 2.8), 0)

    def test_head_rotation_does_not_create_a_spurious_body_turn(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        result = controller.follow_guidance({'maxDistance': 2., 'bodyBearing': 30}, memory,
                     {'goal': [0, 8], 'bearing': 30, 'distance': 2., 'intent': 'explore_frontier'}, 100)
        self.assertEqual(result['action'], 'forward')
        turn = controller.turn(30, 'test', 'test', 50)
        self.assertEqual(turn['action'], 'turn_left')
        plan = visual_step({'start': [0, .92, 0], 'viewHeading': 50, **turn})
        self.assertAlmostEqual(math.degrees(plan['headingTrack'][-1]), 20, places=4)

    def test_delayed_person_bearing_uses_its_exposure_heading(self):
        frame = {'observedAt': 100, 'viewHeading': 40, 'semanticViewHeading': 10,
                 'people': [{'id': 'a', 'confidence': 1, 'opticalPosition': [0, 0, 3]}]}
        self.assertAlmostEqual(person_observation(frame, 100)['bearing'], 30)


if __name__ == '__main__':
    unittest.main()
