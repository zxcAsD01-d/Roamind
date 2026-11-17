import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
import embodied
import runtime
import behavior
import goals
from geometry_clearance import box_distance_squared, segment_box_squared
from reach import envelope
from world import get_world
from test_social_behavior import observation


class GeometryTests(unittest.TestCase):
    def test_circle_corner_is_not_a_square_exclusion_zone(self):
        self.assertGreater(box_distance_squared(.8, .8, [0, 0, 0], [1, 1, 1]), .4**2)
        self.assertLess(box_distance_squared(.7, .7, [0, 0, 0], [1, 1, 1]), .4**2)

    def test_sweep_catches_a_thin_crossing_and_allows_corner_clearance(self):
        self.assertEqual(segment_box_squared([-1, 0], [1, 0], [0, 0], [.001, .5]), 0)
        self.assertAlmostEqual(segment_box_squared([.8, .8], [1, 1], [0, 0], [.5, .5]), .18)
        self.assertAlmostEqual(segment_box_squared([.8, 0], [.8, 0], [0, 0], [.5, .5]), .09)

    def test_table_has_open_space_below_surface(self):
        parts = [o for o in get_world()['objects'] if o.get('sourceId') == 'dining-table']
        self.assertEqual(len(parts), 5)
        point = np.array([0, .3, 5.6])
        self.assertFalse(any(np.all(np.abs(point-o['center']) < np.asarray(o['size'])/2) for o in parts))
        self.assertTrue(any(np.all(np.abs([0, .72, 5.6]-np.array(o['center'])) < np.asarray(o['size'])/2) for o in parts))

    def test_reach_changes_with_avatar_proportions_and_height(self):
        profile = lambda reach: {'effectors': {'right_hand': {'reach': reach, 'height': 1.2, 'lateral': .15}}}
        short, tall = envelope(profile(.5), 1.2, 0), envelope(profile(.8), 1.2, 0)
        self.assertLess(short['maximum'], tall['maximum'])
        self.assertFalse(envelope(profile(.5), 2.5, 0)['reachableHeight'])
        self.assertIsNone(envelope({}, 1., 0))
        with self.assertRaises(ValueError):
            envelope(profile(float('nan')), 1., 0)

    def test_precise_contact_goal_is_not_completed_at_grid_tolerance(self):
        from exploration_memory import ExplorationMemory
        from behavior import precise_endpoint
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.05, 0., .05]), np.eye(3)
        position = [.25, 0., .2]
        route = precise_endpoint(memory, position, memory.key(position), {'available': False})
        self.assertTrue(route['available'])
        self.assertAlmostEqual(route['distance'], .25)
        self.assertEqual(len(route['relativePath']), 1)

    def test_contact_can_choose_subcell_free_position_for_short_arm(self):
        from exploration_memory import ExplorationMemory, Reachability
        from affordances import approach_point
        memory = ExplorationMemory(); memory.position = np.zeros(3)
        memory.cells[(0, 0)] = (-2, 0, 3)
        paths = Reachability({(0, 0): None}, {(0, 0): 0}, {})
        with patch.object(memory, 'reachable', return_value=paths):
            point = approach_point(memory, [.4, 0., .15], .2, .23)
        self.assertIsNotNone(point)
        self.assertLessEqual(np.linalg.norm((np.array(point)-[.4, 0., .15])[[0, 2]]), .23)


class KernelTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear(); runtime.COMMANDS.clear(); runtime.INTERACTIONS.clear()
        goals.TASKS.clear(); embodied.KERNELS.clear(); behavior.POLICY.reset()
        self.now = time.time()
        self.sync()

    def sync(self, vision=None, reports=None):
        return runtime.sync({'clientId': 'kernel', 'vision': vision, 'reports': reports or [],
                             'state': {'playing': False, 'generating': False}})

    def test_context_does_not_expose_house_truth(self):
        result = embodied.context({'clientId': 'kernel'})
        self.assertEqual(result['coordinateFrame'], 'visual-odometry')
        self.assertNotIn('rooms', result['world'])
        self.assertNotIn('bounds', result['world'])
        self.assertIsNone(result['world']['self']['position'])

    def test_retried_intent_never_interrupts_or_duplicates_motion(self):
        request = {'clientId': 'kernel', 'requestId': 'same', 'intent': {'kind': 'action', 'action': 'wave'}}
        original = embodied.submit(request)
        count = len(runtime.COMMANDS)
        second = embodied.submit(request)
        self.assertEqual(original['goal']['id'], second['goal']['id'])
        self.assertEqual(count, len(runtime.COMMANDS))
        self.assertTrue(second['replayed'])
        with self.assertRaises(ValueError):
            embodied.submit({**request, 'intent': {'kind': 'face', 'heading': 10}})

    def test_invalid_intent_cannot_cancel_active_goal(self):
        initial = embodied.submit({'requestId': 'one', 'intent': {'kind': 'face', 'heading': 30}})
        with self.assertRaises(ValueError):
            embodied.submit({'requestId': 'two', 'intent': {'kind': 'teleport'}})
        self.assertEqual(goals.TASKS['kernel']['id'], initial['goal']['id'])

    def test_autonomy_requires_fresh_vision_and_preserves_active_goal(self):
        embodied.configure({'enabled': True})
        self.sync(observation(self.now-10))
        self.assertFalse(goals.TASKS)
        self.sync(observation(time.time()))
        first = goals.TASKS['kernel']['id']
        for _ in range(5):
            self.sync(observation(time.time()))
        self.assertEqual(goals.TASKS['kernel']['id'], first)
        self.assertTrue(embodied.KERNELS['kernel'].enabled)
        runtime.enqueue('interrupt', {'clientId': 'kernel', 'mode': 'hold'})
        self.assertFalse(embodied.KERNELS['kernel'].enabled)
        self.assertEqual(goals.TASKS['kernel']['status'], 'cancelled')

    def test_completion_comes_from_actual_execution_reports(self):
        task = embodied.submit({'requestId': 'gesture', 'intent': {'kind': 'action', 'action': 'wave'}})['goal']
        stop = next(reversed(runtime.COMMANDS))
        self.sync(reports=[{'id': stop, 'status': 'completed'}])
        command = goals.TASKS['kernel']['commandId']
        self.sync(reports=[{'id': command, 'status': 'playing'}])
        self.assertEqual(goals.TASKS['kernel']['status'], 'running')
        self.sync(reports=[{'id': command, 'status': 'completed'}])
        self.assertEqual(goals.TASKS['kernel']['status'], 'completed')
        replay = embodied.submit({'requestId': 'gesture', 'intent': {'kind': 'action', 'action': 'wave'}})
        self.assertEqual(replay['goal']['id'], task['id'])
        self.assertEqual(replay['goal']['status'], 'completed')

    def test_failures_back_off_and_do_not_create_unbounded_tasks(self):
        embodied.configure({'enabled': True})
        with patch('embodied.choose', side_effect=ValueError('no path')):
            self.sync(observation(time.time()))
            self.sync(observation(time.time()))
        state = embodied.KERNELS['kernel']
        self.assertEqual(state.failures, 1)
        self.assertGreater(state.next_at, time.time())
        self.assertFalse(goals.TASKS)

    def test_social_completion_cools_down_same_person_across_new_goals(self):
        state = embodied.kernel('kernel')
        engine = behavior.POLICY._state('kernel').engine
        engine.arbiter.target_id = 'person-a'
        task = {'id': 'done', 'status': 'completed', 'steps': [{'kind': 'social'}], 'detail': 'waved'}
        embodied.finish('kernel', state, task, self.now)
        behavior.POLICY.begin_social('kernel')
        engine = behavior.POLICY._state('kernel').engine
        engine.world.update(observation(self.now, distance=1.2), self.now)
        self.assertIsNone(engine.arbiter.select(engine.world, {'groups': []}, self.now))
        self.assertEqual(engine.arbiter.deferred['person-a'], self.now+90)


if __name__ == '__main__':
    unittest.main()
