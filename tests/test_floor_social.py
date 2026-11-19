"""Floor topology, complete visual routes and social evidence regressions."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from exploration_memory import ExplorationMemory, Reachability
from floor_topology import FloorTopology, neighbors
from semantic_map import SemanticMap
from social_navigation import SocialNavigator, attach_route, person_observation
from visual_step import visual_step
from visual_attention import person_attention
from visual_navigation import depth_navigation
from stereo_service import visual_body_position
from test_social_behavior import observation
from behavior import BehaviorPolicy
import test_stereo


class FloorSocialTests(unittest.TestCase):
    def two_rooms(self):
        free = {(x, z) for x in [*range(10), *range(13, 23)] for z in range(10)}
        free.update((x, z) for x in range(10, 13) for z in range(3, 6))
        walls = {other for key in free for other in neighbors(key) if other not in free}
        return {**{key: (-2., 100, 1) for key in free}, **{key: (2., 100, 1) for key in walls}}

    def test_floor_partitions_rooms_at_narrow_passage_and_keeps_identity(self):
        topology = FloorTopology()
        cells = self.two_rooms()
        topology.update(cells, 1, 100)
        first = topology.region_at([1, 0, 1])
        second = topology.region_at([6, 0, 1])
        self.assertNotEqual(first, second)
        self.assertTrue(topology.passages)
        self.assertTrue(all(item['boundary'] == 'uncertain-boundary' for item in topology.regions.values()))
        topology.update(cells, 2, 102)
        self.assertEqual(first, topology.region_at([1, 0, 1]))
        del cells[(-1, 4)]
        topology.update(cells, 3, 104)
        self.assertEqual(topology.regions[first]['boundary'], 'open-unobserved')

    def test_depth_coverage_does_not_claim_person_search_and_unknown_is_not_a_door(self):
        memory = ExplorationMemory()
        for index, yaw in enumerate(range(0, 360, 45)):
            frame = observation(100+index, yaw=yaw)
            memory.update(frame['stereo'], [], 100+index)
        self.assertTrue(all(not item['personChecked'] for item in memory.snapshot()['topology']['regions']))
        self.assertEqual(memory.snapshot()['topology']['doors'], [])
        frame = observation(100)
        frame['semanticReady'] = False
        policy = BehaviorPolicy()
        with patch('time.time', return_value=100):
            result = policy.decide({'clientId': 'warming', 'vision': frame})
        self.assertEqual(result['action'], 'forward')
        self.assertGreaterEqual(result.get('routeLength', result['distance']), .65)
        self.assertTrue(all(not region['personChecked'] for region in
                            policy.states['warming'].memory.snapshot()['topology']['regions']))
        policy.observe('warming', frame)
        self.assertTrue(all(not region['views'] for region in policy.states['warming'].memory.topology.regions.values()))

    def test_semantic_survey_only_counts_unique_fresh_exposures(self):
        memory = ExplorationMemory()
        for index, yaw in enumerate(range(0, 360, 45)):
            frame = observation(100+index, yaw=yaw)
            memory.update(frame['stereo'], [], 100+index)
            memory.observe_semantics(frame, 100+index)
        region = memory.topology.regions[memory.topology.region_at(memory.position)]
        self.assertEqual(len(region['views']), 24)
        self.assertIsNotNone(region['searchedAt'])
        before = len(memory.topology.observations)
        memory.observe_semantics(frame, 108)
        self.assertEqual(len(memory.topology.observations), before)

    def test_door_requires_semantics_and_occupied_door_is_not_traversable(self):
        topology = FloorTopology()
        topology.update(self.two_rooms(), 1, 100)
        topology.remember_doors([{'id': 'door', 'label': 'door', 'confidence': .9,
                                  'worldPosition': [-.175, 1., 1.4]}], 100)
        topology.apply_observations()
        self.assertFalse(topology.doors['door']['traversable'])

    def test_searched_region_selects_one_exit_route_to_unsearched_region(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([1.75, 0, 1.75]), np.eye(3)
        memory.cells = self.two_rooms()
        memory.topology.update(memory.cells, 1, 100)
        current = memory.topology.region_at(memory.position)
        memory.topology.regions[current]['searchedAt'] = 100
        route = memory.guidance(101)
        self.assertEqual(route['intent'], 'navigate_region_exit')
        self.assertNotEqual(memory.topology.assignments[tuple(route['goal'])], current)
        self.assertTrue(route['relativePath'])
        chosen = route['goal']
        self.assertEqual(memory.guidance(102)['goal'], chosen)

    def test_complete_route_exceeds_one_segment_without_crossing_unknown_floor(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        memory.cells = {(x, z): (-2., 100, 1) for x in range(-3, 4) for z in range(-2, 29)}
        route = memory.route_to((0, 25), 'user_waypoint')
        self.assertGreater(len(route['relativePath']), 2)
        result = attach_route({'action': 'forward', 'distance': route['distance'], 'duration': 6}, route, route['distance'])
        plan = visual_step({'start': [10, .92, 20], 'viewHeading': 90, **result})
        self.assertGreater(len(plan['path']), 3)
        self.assertAlmostEqual(plan['path'][-1][0], 18.75, places=3)
        self.assertAlmostEqual(plan['path'][-1][2], 20, places=3)
        self.assertEqual(len(plan['timeline']), 1)
        self.assertNotIn('relativePath', attach_route({}, route, .3))
        for x in range(-3, 4):
            del memory.cells[(x, 15)]
        memory.map_revision += 1
        self.assertIsNone(memory.route_to((0, 25), 'user_waypoint'))

    def test_body_footprint_cannot_plan_along_unknown_floor_edge(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        memory.cells = {(x, z): (-2., 100, 1) for x in range(-3, 4) for z in range(-2, 12)}
        self.assertTrue(memory.floor_supported((0, 8)))
        self.assertFalse(memory.floor_supported((3, 8)))
        self.assertNotIn((3, 8), memory.reachable())
        self.assertIn((0, 8), memory.reachable())

    def test_frontier_exit_direction_looks_beyond_the_body_floor_margin(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        memory.cells = {(x, z): (-2., 100, 1) for x in range(-3, 4) for z in range(-3, 7)}
        memory.choose_goal(Reachability({}, {(0, 5): 6}, {}), 100)
        exit = next(iter(memory.coverage.exits.values()))
        self.assertEqual(exit['direction'], 0, '未知地板在北向，不能默认记录为东向出口')

    def test_floor_support_and_slope_gate_translation(self):
        stereo = observation(100)['stereo']
        with patch('time.time', return_value=100):
            self.assertGreater(depth_navigation(stereo)['maxDistance'], 0)
            for ground in (None, {'frame': 'gravity-aligned-y-up', 'support': 400, 'tiltDeg': 20, 'residual': .01}):
                self.assertEqual(depth_navigation({**stereo, 'groundInfo': ground})['maxDistance'], 0)

    def test_navigation_uses_visual_body_center_not_bobbing_camera(self):
        memory = ExplorationMemory()
        for index, x in enumerate((0., .08, -.07)):
            frame = observation(100+index)
            pose = np.eye(4)
            pose[:3, 3] = [x, 1., 0.]
            own = {'bodyInOptical': [-x, -1., 0.], 'opticalToControl': [999.]*16}
            center = visual_body_position(own, pose)
            np.testing.assert_allclose(center, [0, 0, 0])
            frame['stereo']['pose'] = {'cameraToWorld': pose.ravel().tolist(), 'bodyPosition': center}
            memory.update(frame['stereo'], [], 100+index)
            np.testing.assert_allclose(memory.position, [0, 0, 0])
        with self.assertRaises(ValueError):
            visual_body_position({'bodyInOptical': [float('nan'), 0, 0]}, np.eye(4))

    def test_unstable_view_waits_without_switching_exit_target(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        memory.goal = (0, 5)
        route = {'goal': [0, 5], 'bearing': 0, 'distance': 2, 'intent': 'navigate_region_exit'}
        navigation = {'tracking': True, 'maxDistance': 0, 'bodyBearing': 0, 'sectors': []}
        for stamp in (100, 101, 102):
            result = controller.follow_guidance(navigation, memory, route, stamp)
            self.assertEqual(result['intent'], 'inspect_route_floor')
            self.assertEqual(memory.goal, (0, 5))

    def test_survey_waits_for_semantics_after_the_turn_instead_of_reversing(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        controller.phase = 'survey_surroundings'
        controller.feedback('turn_left', 'completed', memory, 100)
        result = controller.step({'semanticObservedAt': 99.9}, {}, memory, {}, 100.1)
        self.assertTrue(result['wait'])

    def test_wide_walkable_doorway_still_separates_broad_floor_regions(self):
        free = {(x, z) for x in [*range(12), *range(15, 27)] for z in range(12)}
        free.update((x, z) for x in range(12, 15) for z in range(4, 8))
        topology = FloorTopology()
        topology.update({key: (-2., 100, 1) for key in free}, 1, 100)
        self.assertNotEqual(topology.region_at([1.75, 0, 1.75]), topology.region_at([7.35, 0, 1.75]))
        self.assertTrue(topology.passages)

    def test_near_person_without_facing_evidence_does_not_trigger_wave(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        person = {'id': 'a', 'distance': 1.3, 'bearing': 0, 'confidence': .9, 'attention': {'facingCamera': None}}
        for stamp in (100, 100.5, 101):
            result = controller.approach({**person, 'observedAt': stamp}, {}, memory, stamp)
        self.assertFalse(controller.ready)
        self.assertEqual(result['intent'], 'await_attention')

    def test_person_range_uses_visual_body_center_despite_pitched_camera(self):
        frame = observation(100, distance=2)
        angle = math.radians(30)
        matrix = np.eye(4)
        matrix[:3, :3] = [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                              [0, math.sin(angle), math.cos(angle)]]
        frame['semanticPose'] = matrix.ravel().tolist()
        frame['stereo']['pose'] = {'cameraToWorld': matrix.ravel().tolist(), 'bodyPosition': [0, -1.2, 0]}
        frame['viewHeading'], frame['semanticViewHeading'] = 30, 0
        person = person_observation(frame, 100)
        self.assertAlmostEqual(person['distance'], 2*math.cos(angle), places=5)
        self.assertAlmostEqual(person['bearing'], 0., places=5)

    def test_visible_face_front_and_side_are_distinguished(self):
        person = observation(100, distance=1.3)['people'][0]
        self.assertTrue(person_attention(person, [])['facingCamera'])
        person['keypoints'][2]['pixel'][0] += 40
        self.assertFalse(person_attention(person, [])['facingCamera'])

    def test_occluded_ears_do_not_overrule_confident_eyes_and_nose(self):
        person = observation(100, distance=1.3)['people'][0]
        person['keypoints'] = [point for point in person['keypoints'] if 'ear' not in point['name']]
        self.assertTrue(person_attention(person, [])['facingCamera'])
        person['keypoints'][0]['confidence'] = .4
        self.assertIsNone(person_attention(person, [])['facingCamera'])

    def test_head_facing_estimate_is_invariant_to_camera_roll(self):
        person = observation(100, distance=1.3)['people'][0]
        angle = math.radians(35)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        for point in person['keypoints']:
            point['pixel'] = ((np.asarray(point['pixel'])-[300, 95])@rotation.T+[300, 95]).tolist()
        self.assertTrue(person_attention(person, [])['facingCamera'])

    def test_person_approach_uses_a_complete_persistent_standoff_route(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        memory.cells = {(x, z): (-2., 100, 1) for x in range(-5, 6) for z in range(-3, 24)}
        person = {'id': 'a', 'worldPosition': [.175, 1, 6.175], 'distance': 6.,
                  'bearing': 0., 'observedAt': 100, 'confidence': .9}
        navigation = {'bodyBearing': 0, 'tracking': True, 'safetyMargin': .5,
                      'sectors': [{'angle': 0, 'clearance': 3, 'confidence': 1, 'obstacleDistance': None}]}
        result = controller.approach(person, navigation, memory, 100)
        self.assertEqual(result['intent'], 'approach_person')
        self.assertGreater(result['routeLength'], 4.)
        goal = controller.person_route_goal
        self.assertEqual(memory.approach_goal(person, 101, goal), goal)
        self.assertEqual(result['targetPosition'], person['worldPosition'])
        controller.feedback('forward', 'interrupted', memory, 101, '人物目标已移动，停步后更新完整路线')
        self.assertEqual(controller.failed_actions, 0)
        self.assertIsNone(controller.person_route_goal)

    def test_follow_replans_on_target_motion_without_chasing_frame_noise(self):
        controller = SocialNavigator()
        args = {'action': 'forward', 'intent': 'approach_person', 'targetPosition': [0, 0, 3]}
        for stamp in (100, 100.5):
            frame = observation(stamp, distance=4)
            frame['semanticPose'] = np.eye(4).ravel().tolist()
            reason = controller.monitor(frame, args, stamp)
        self.assertIsNotNone(reason)
        self.assertIsNone(controller.monitor(frame, args, 100.6))

    def test_moving_person_body_id_change_can_update_navigation_route(self):
        controller = SocialNavigator()
        args = {'action': 'forward', 'intent': 'approach_person', 'targetPosition': [0, 0, 3]}
        for stamp, distance, identity in [(100, 3., 'a'), (100.6, 3.8, 'b')]:
            frame = observation(stamp, distance=distance, identity=identity)
            frame['semanticPose'] = np.eye(4).ravel().tolist()
            reason = controller.monitor(frame, args, stamp)
        self.assertIsNotNone(reason)
        self.assertEqual(controller.last_interrupt, 100.6)

    def test_repeated_person_frame_cannot_confirm_route_interruption(self):
        controller = SocialNavigator()
        args = {'action': 'forward', 'intent': 'approach_person', 'targetPosition': [0, 0, 3]}
        frame = observation(100, distance=4)
        frame['semanticPose'] = np.eye(4).ravel().tolist()
        for now in (100, 100.5, 101):
            self.assertIsNone(controller.monitor(frame, args, now))

    def test_person_loss_checks_last_visual_position_before_resuming_survey(self):
        controller = SocialNavigator()
        controller.target = {'id': 'a', 'worldPosition': [2., 0., 3.], 'observedAt': 100}
        frame = observation(103)
        frame['stereo']['pose']['bodyPosition'] = [0., 0., 0.]
        result = controller.reacquire(None, frame, {'bodyBearing': 0}, 103)
        self.assertEqual(result['intent'], 'retain_person')
        self.assertTrue(result['action'].startswith('turn_'))
        self.assertIsNone(controller.reacquire(None, frame, {'bodyBearing': 0}, 109))

    def test_transient_localization_loss_waits_before_recovery_rotation(self):
        controller, memory = SocialNavigator(), ExplorationMemory()
        frame = observation(100)
        frame['stereo'].update(mapping=False, validRatio=.3)
        frame['stereo']['pose']['status'] = 'lost'
        self.assertTrue(controller.recover(frame, memory, 100)['wait'])
        self.assertTrue(controller.recover(frame, memory, 100.5)['wait'])
        frame['stereo']['observedAt'] = 101
        self.assertEqual(controller.recover(frame, memory, 101)['intent'], 'recovering')

    def test_person_points_expire_but_environment_points_remain_in_memory(self):
        world, frame = SemanticMap(), test_stereo.SemanticMapTests().frame()
        world.integrate(frame, np.eye(4), 100)
        world.annotate(frame, np.eye(4), [{'box': [80, 45, 200, 135], 'label': 'person', 'confidence': .9}], 100)
        self.assertTrue(world.snapshot(100)['personCloud']['positions'])
        later = world.snapshot(110)
        self.assertFalse(later['personCloud']['positions'])
        self.assertGreater(later['totalPoints'], 0)


if __name__ == '__main__':
    unittest.main()
