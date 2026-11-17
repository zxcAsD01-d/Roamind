"""A persistent exit intent must use the observed route around furniture."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from exploration_memory import ExplorationMemory
from intent_route import route_to_target
from social_navigation import SocialNavigator


class IntentRouteTests(unittest.TestCase):
    def setUp(self):
        self.memory = ExplorationMemory()
        self.memory.position, self.memory.rotation = np.array([.175, 0., .175]), np.eye(3)
        self.memory.cells = {(x, z): (-2., 100., 3) for x in range(-6, 9) for z in range(-3, 16)}
        for x in range(-1, 2):
            for z in range(3, 7):
                self.memory.cells[x, z] = (3., 100., 3)
        self.target = np.array([.175, 0., 4.375])

    def test_exit_behind_obstacle_retains_complete_curved_route(self):
        route = route_to_target(self.memory, self.target, 'seek_possible_exit')
        self.assertFalse(route['partial'])
        self.assertGreater(len(route['relativePath']), 1)
        self.assertGreater(abs(route['bearing']), 10)
        navigator = SocialNavigator()
        navigator.drive.commit_target = self.target.copy()
        command = {'action': 'forward', 'bearing': 0, 'distance': 4.2, 'intent': 'seek_possible_exit', 'reason': ''}
        result = navigator.routed_exploration(command, {'tracking': True, 'visualAvoidance': False}, self.memory, 100)
        self.assertEqual(result['relativePath'], route['relativePath'])
        np.testing.assert_array_equal(navigator.drive.commit_target, self.target)

    def test_unknown_target_uses_reachable_prefix_without_fake_final_leg(self):
        target = np.array([.175, 0., 12.])
        route = route_to_target(self.memory, target, 'seek_possible_exit')
        self.assertTrue(route['partial'])
        self.assertLess(route['relativePath'][-1][1], 8)
        self.assertIn(tuple(route['goal']), self.memory.reachable())
        self.assertEqual(route['targetPosition'], target.tolist())

    def test_invalid_target_cannot_enter_grid_search(self):
        for target in ([0., 0.], [0., 0., float('nan')]):
            with self.assertRaises(ValueError):
                route_to_target(self.memory, target, 'explore')


if __name__ == '__main__':
    unittest.main()
