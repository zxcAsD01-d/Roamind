"""Mission outcomes, visual evidence gates, route memory and movement feedback."""
import math
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from behavior import BehaviorPolicy
from exploration_memory import ExplorationMemory
from social_navigation import SocialNavigator
from visual_step import visual_step
import runtime


def observation(stamp, distance=None, bearing=0, identity='person-a', clearance=3, yaw=0):
    radians = math.radians(yaw)
    matrix = np.array([[math.cos(radians), 0, math.sin(radians), 0], [0, 1, 0, 0],
                       [-math.sin(radians), 0, math.cos(radians), 0], [0, 0, 0, 1]])
    angle = math.radians(bearing)
    people = [] if distance is None else [{'id': identity, 'confidence': .9, 'box': [200, 40, 400, 350],
              'keypoints': [{'name': name, 'pixel': pixel, 'confidence': .9} for name, pixel in
                            [('left_eye', [315, 80]), ('right_eye', [285, 80]), ('nose', [300, 95]),
                             ('left_ear', [335, 90]), ('right_ear', [265, 90])]],
              'opticalPosition': [distance*math.sin(angle), 0, distance*math.cos(angle)]}]
    return {'signature': [.2]*64, 'objects': [], 'faces': [], 'width': 640, 'height': 360,
            'observedAt': stamp, 'semanticObservedAt': stamp, 'people': people,
            'stereo': {'method': 'stereo-sgbm-lr', 'observedAt': stamp, 'mapping': True,
                       'pose': {'cameraToWorld': matrix.ravel().tolist()},
                       'sectors': [{'angle': angle, 'clearance': clearance, 'confidence': .9}
                                   for angle in range(-45, 46, 15)]}}


class ScriptedSocialTests(unittest.TestCase):
    def setUp(self):
        self.policy = BehaviorPolicy()

    def decide(self, stamp=100, **kwargs):
        with patch('time.time', return_value=stamp):
            return self.policy.decide({'clientId': 'mission', 'vision': observation(stamp, **kwargs)})

    def test_far_person_is_approached_without_overshooting_social_distance(self):
        result = self.decide(distance=1.9)
        self.assertEqual(result['action'], 'forward')
        self.assertLessEqual(result['distance'], .6)
        self.assertFalse(result['mission']['ready'])

    def test_confirmed_long_corridor_is_one_continuous_plan(self):
        result = self.decide(distance=5, clearance=5)
        self.assertEqual(result['action'], 'forward')
        self.assertGreater(result['distance'], 2.)
        self.assertTrue(result['continuous'])
        plan = visual_step({'start': [0, .92, 0], **result})
        self.assertTrue(plan['timeline'][0]['continuous'])
        self.assertGreater(plan['duration'], 4)

    def test_brief_person_occlusion_does_not_switch_to_exploration_or_another_person(self):
        self.decide(distance=3)
        for stamp, args in [(100.5, {}), (101, {'distance': 2, 'identity': 'person-b'})]:
            result = self.decide(stamp, **args)
            self.assertEqual(result['intent'], 'retain_person')
            self.assertEqual(result['distance'], 0)
            self.assertEqual(result['mission']['target']['id'], 'person-a')

    def test_known_angular_coverage_survives_starting_a_new_social_goal(self):
        memory = self.policy._state('mission').memory
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        memory.coverage.regions[memory.coverage.region(memory.position)] = {'views': set(range(24)), 'seen': 100}
        self.policy.begin_social('mission')
        self.assertIsNone(self.policy.states['mission'].navigator.survey(memory))
        self.assertTrue(self.policy.states['mission'].navigator.survey_complete)

    def test_off_axis_person_is_aligned_before_any_approach(self):
        result = self.decide(distance=3, bearing=-22)
        self.assertEqual(result['action'], 'turn_left')
        self.assertEqual(result['distance'], 0)
        self.assertAlmostEqual(result['relativeBearing'], -22)

    def test_social_ready_requires_unique_metric_observations_over_time(self):
        for stamp in (100, 100.4):
            self.assertFalse(self.decide(stamp, distance=1.3)['mission']['ready'])
        final = self.decide(100.9, distance=1.3)
        self.assertTrue(final['mission']['ready'])
        self.assertEqual(final['action'], 'observe')
        self.assertEqual(final['distance'], 0)

    def test_completed_mission_never_resumes_movement_during_visual_loss(self):
        for stamp in (100, 100.4, 100.9):
            self.decide(stamp, distance=1.3)
        frame = observation(102)
        frame['stereo'].update({'mapping': False, 'validRatio': .5})
        with patch('time.time', return_value=102):
            result = self.policy.decide({'clientId': 'mission', 'vision': frame})
        self.assertEqual(result['action'], 'observe')
        self.assertEqual(result['distance'], 0)
        self.assertTrue(self.policy.snapshot('mission')['mission']['ready'])

    def test_fresh_depth_cannot_recount_one_semantic_frame(self):
        for stamp in (100, 100.4, 100.9):
            frame = observation(stamp, distance=1.3)
            frame['semanticObservedAt'] = 100
            with patch('time.time', return_value=stamp):
                result = self.policy.decide({'clientId': 'mission', 'vision': frame})
            self.assertFalse(result['mission']['ready'])
        self.assertEqual(result['mission']['confirmationFrames'], 1)

    def test_identity_switch_resets_confirmation(self):
        self.decide(100, distance=1.3)
        self.decide(100.5, distance=1.3)
        result = self.decide(101, distance=1.3, identity='person-b')
        self.assertFalse(result['mission']['ready'])
        self.assertEqual(result['mission']['confirmationFrames'], 1)

    def test_missing_or_stale_person_distance_never_means_success(self):
        for stamp in (100, 101, 102):
            frame = observation(stamp)
            frame['objects'] = [{'label': 'person', 'confidence': .95, 'box': [100, 0, 600, 360]}]
            with patch('time.time', return_value=stamp):
                result = self.policy.decide({'clientId': 'mission', 'vision': frame})
            self.assertFalse(result['mission']['ready'])
            self.assertEqual(result['action'], 'forward')
            self.assertLessEqual(result['distance'], 1.5)
        frame = observation(103, distance=1.3)
        frame['semanticObservedAt'] = 99
        with patch('time.time', return_value=103):
            result = self.policy.decide({'clientId': 'mission', 'vision': frame})
        self.assertFalse(result['mission']['ready'])

    def test_policy_weights_do_not_randomly_override_script(self):
        first = self.decide(100, distance=3, bearing=25)
        self.policy.states['mission'].weights[:] = -400
        second = self.decide(101, distance=3, bearing=25)
        self.assertEqual(first['action'], second['action'])
        self.assertEqual(first['relativeBearing'], second['relativeBearing'])

    def test_safe_known_route_moves_without_faking_a_complete_survey(self):
        result = self.decide()
        self.assertEqual(result['action'], 'forward')
        self.assertGreaterEqual(result.get('routeLength', result['distance']), .65)
        self.assertTrue(result['continuous'])
        self.assertFalse(self.policy.states['mission'].navigator.survey_complete)
        self.assertTrue(all(not region['personChecked'] for region in
                            self.policy.states['mission'].memory.snapshot()['topology']['regions']))

    def test_insufficient_travel_keeps_surveying_unseen_headings(self):
        memory, controller = ExplorationMemory(), SocialNavigator()
        for index, yaw in enumerate((0, 65, 130, 195)):
            frame = observation(100+index, yaw=yaw)
            memory.update(frame['stereo'], [], 100+index)
            memory.observe_semantics(frame, 100+index)
            result = controller.step(frame, {'maxDistance': .3}, memory, memory.guidance(100+index), 100+index)
            self.assertEqual(result['intent'], 'survey_surroundings')
            self.assertEqual(result['distance'], 0)
        frame = observation(105, yaw=260)
        memory.update(frame['stereo'], [], 105)
        memory.observe_semantics(frame, 105)
        controller.step(frame, {'maxDistance': .3}, memory, memory.guidance(105), 105)
        self.assertTrue(controller.survey_complete)

    def test_repeated_views_do_not_fake_a_complete_survey(self):
        memory, controller = ExplorationMemory(), SocialNavigator()
        for index in range(12):
            frame = observation(100+index, yaw=0 if index % 2 else 65)
            memory.update(frame['stereo'], [], 100+index)
            controller.survey(memory)
        self.assertFalse(controller.survey_complete)

    def test_visual_pose_route_and_snapshot_are_json_serializable(self):
        result = self.decide()
        json.dumps(result)
        json.dumps(self.policy.snapshot('mission'))

    def test_lost_localization_explores_without_using_simulation_position(self):
        frame = observation(100)
        frame['stereo']['mapping'] = False
        with patch('time.time', return_value=100):
            result = self.policy.decide({'clientId': 'mission', 'vision': frame, 'position': [10, 0, 20]})
        self.assertEqual(result['action'], 'forward')
        self.assertGreaterEqual(result['distance'], 1.5)
        self.assertIsNone(self.policy.states['mission'].memory.position)

    def test_visual_loss_does_not_force_repeated_stationary_scanning(self):
        for stamp in range(100, 110):
            frame = observation(stamp)
            frame['stereo'].update({'mapping': False, 'validRatio': .5})
            with patch('time.time', return_value=stamp):
                result = self.policy.decide({'clientId': 'mission', 'vision': frame})
            self.assertEqual(result['action'], 'forward')
            self.assertGreaterEqual(result['distance'], 1.5)
            self.assertFalse(result['navigation']['tracking'])

    def test_recovery_keeps_returning_towards_last_known_view(self):
        memory, controller = ExplorationMemory(), SocialNavigator()
        memory.scan_sign = 1
        for index, angle in enumerate((15, 30, 45, 60, 60, 60, 60, 30)):
            stamp = 100+index
            frame = observation(stamp)
            frame['stereo'].update({'mapping': False, 'validRatio': 0})
            result = controller.recover(frame, memory, stamp)
            self.assertEqual(result['relativeBearing'], -angle)
            memory.scan_sign = -1
        self.assertIsNone(controller.recover(frame, memory, 107))

    def test_waypoint_relocalizes_without_abandoning_its_visual_target(self):
        self.decide()
        memory = self.policy.states['mission'].memory
        origin = memory.position.copy()
        frame = observation(101)
        frame['stereo'].update(mapping=False, validRatio=0)
        with patch('time.time', return_value=101):
            result = self.policy.waypoint('mission', frame, [0, 0, 2])
        self.assertEqual(result['intent'], 'recovering')
        self.assertEqual(result['distance'], 0)
        np.testing.assert_array_equal(memory.position, origin)

    def test_visible_person_can_be_confirmed_without_global_localization(self):
        for stamp in (100, 100.4, 100.9):
            frame = observation(stamp, distance=1.3)
            frame['stereo']['mapping'] = False
            with patch('time.time', return_value=stamp):
                result = self.policy.decide({'clientId': 'mission', 'vision': frame})
            self.assertEqual(result['action'], 'observe')
            self.assertEqual(result['distance'], 0)
        self.assertTrue(result['mission']['ready'])

    def test_lost_global_pose_still_allows_ego_relative_person_approach(self):
        frame = observation(100, distance=3)
        frame['stereo'].update({'mapping': False, 'validRatio': .5})
        with patch('time.time', return_value=100):
            result = self.policy.decide({'clientId': 'mission', 'vision': frame})
        self.assertEqual(result['action'], 'forward')
        self.assertAlmostEqual(result['distance'], 1.7)
        self.assertFalse(result['mission']['ready'])

    def test_social_turn_waits_for_semantics_captured_after_completion(self):
        first = self.decide(100, distance=1.3, bearing=30)
        with patch('time.time', return_value=104):
            self.policy.feedback_for('mission', first['decisionId'], 'completed')
        frame = observation(104.2)
        frame['semanticObservedAt'] = 103.9
        with patch('time.time', return_value=104.2):
            result = self.policy.decide({'clientId': 'mission', 'vision': frame})
        self.assertTrue(result['wait'])
        self.assertNotIn('action', result)
        fresh = self.decide(104.5, distance=1.3)
        self.assertEqual(fresh['intent'], 'confirming_social')

    def test_visual_map_reset_does_not_reuse_old_region_coordinates(self):
        self.decide()
        self.policy.states['mission'].memory.people['old-person'] = {'position': [99, 0, 99], 'seen': 100, 'confidence': .9}
        frame = observation(101)
        frame['stereo']['pose']['status'] = 'initialized'
        with patch('time.time', return_value=101):
            self.policy.decide({'clientId': 'mission', 'vision': frame})
        self.assertEqual(self.policy.states['mission'].memory.people, {})

    def test_complete_scan_with_no_passage_waits_without_spinning_forever(self):
        memory, controller = ExplorationMemory(), SocialNavigator()
        nav = {'maxDistance': 0}
        for index, yaw in enumerate(range(0, 361, 45)):
            frame = observation(100+index, clearance=0, yaw=yaw)
            memory.update(frame['stereo'], [], 100+index)
            result = controller.step(frame, nav, memory, memory.guidance(100+index), 100+index)
        self.assertEqual(result['intent'], 'waiting_for_change')
        self.assertEqual(result['action'], 'observe')
        self.assertFalse(result['mission']['ready'])

    def test_small_turn_compiler_preserves_social_alignment_angle(self):
        for action, bearing, expected in [('turn_left', -17, 17), ('turn_right', 19, -19)]:
            plan = visual_step({'start': [0, .92, 0], 'action': action, 'relativeBearing': bearing})
            self.assertAlmostEqual(math.degrees(plan['headingTrack'][-1]), expected, places=4)
            self.assertEqual(plan['rootTrack'][0], plan['rootTrack'][-1])

    def test_visual_bearings_are_resolved_from_camera_heading(self):
        plan = visual_step({'start': [0, .92, 0], 'startHeading': 80, 'viewHeading': 30,
                            'action': 'turn_right', 'relativeBearing': 20})
        self.assertAlmostEqual(math.degrees(plan['headingTrack'][-1]), 10, places=4)
        self.assertAlmostEqual(math.degrees(plan['headingTrack'][0]), 80, places=4)
        forward = visual_step({'start': [0, .92, 0], 'startHeading': 80, 'viewHeading': 30,
                               'action': 'forward', 'relativeBearing': 10, 'distance': .6})
        self.assertAlmostEqual(math.degrees(math.atan2(forward['rootTrack'][-1][0],
                                                      forward['rootTrack'][-1][2])), 20, places=4)


class RegionMemoryTests(unittest.TestCase):
    def test_measured_side_wall_does_not_become_an_obstacle_in_corridor_center(self):
        memory = ExplorationMemory()
        stereo = observation(100)['stereo']
        stereo['sectors'] = [{'angle': 0, 'clearance': 1., 'confidence': 1., 'obstacleDistance': 1.05}]
        stereo['navigationGrid'] = {'frame': 'visual-odometry', 'free': [[.05, 1.2]], 'occupied': [[.85, 1.2]]}
        memory.update(stereo, [], 100)
        self.assertFalse(memory.near_obstacle((0, 3)))
        self.assertTrue(memory.near_obstacle((2, 3)))

    def test_measured_corridor_width_is_retained_without_filling_unseen_sides(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0., .175]), np.eye(3)
        memory.ray({'angle': 0, 'clearance': 3, 'confidence': 1}, 100)
        self.assertLess(memory.cells[(-1, 3)][0], -.15)
        self.assertLess(memory.cells[(1, 3)][0], -.15)
        self.assertNotIn((2, 3), memory.cells)

    def test_corridor_route_keeps_clearance_and_smoothing_does_not_cut_corner(self):
        memory = self.memory()
        for x in range(-4, 5):
            for z in range(5, 16):
                memory.cells[(x, z)] = (-2., 100., 1)
        for z in range(1, 16):
            memory.cells[(-3, z)] = (3., 100., 1)
            memory.cells[(4, z)] = (3., 100., 1)
        memory.position = np.array([-.525, 0, -.175])
        path = memory.reachable().path((-1, 12))
        self.assertTrue(any(key[0] in (0, 1) for key in path[3:-3]))
        route = memory.route_to((-1, 12), 'explore_frontier')
        self.assertLess(route['bearing'], 0, '先向通道中心靠拢，不能直线切回墙边')
        self.assertTrue(memory.near_obstacle((-2, 0)), '对角障碍也必须保留身体间距')

    def test_failed_local_motion_changes_route_without_inventing_obstacle(self):
        memory, controller = self.memory(), SocialNavigator()
        controller.survey_complete = True
        cells = dict(memory.cells)
        motion = ([.175, .175], [.175, .825])
        controller.feedback('forward', 'failed', memory, 100, '预设根运动无法安全通过：门框', motion)
        result = controller.step(observation(101), {'maxDistance': .65}, memory, memory.guidance(101), 101)
        self.assertEqual(result['intent'], 'inspect_blockage')
        self.assertNotEqual(result['action'], 'forward')
        self.assertFalse(memory.clear_segment((0, 2)))
        self.assertEqual(memory.cells, cells, '执行失败不能捏造视觉占据证据')
        self.assertNotEqual(memory.reachable().path((0, 3))[1], (0, 1))
        memory.expire_motions(131)
        self.assertTrue(memory.clear_segment((0, 2)))

    def test_cancelled_or_short_motion_does_not_poison_local_route(self):
        memory, controller = self.memory(), SocialNavigator()
        motion = ([.175, .175], [.175, .825])
        controller.feedback('forward', 'interrupted', memory, 100, '用户取消', motion)
        controller.feedback('forward', 'failed', memory, 101, '移动距离不足一个安全落脚步', motion)
        self.assertEqual(memory.rejected_motions, [])
        self.assertFalse(controller.blocked_turn)

    def test_partial_step_failure_allows_leaving_the_rejected_direction(self):
        memory = self.memory()
        memory.reject_motion(([.175, -.2], [.175, .8]), 100)
        self.assertTrue(memory.motion_blocked(np.array([.175, .175]), np.array([.175, .6])))
        self.assertFalse(memory.motion_blocked(np.array([.175, .175]), np.array([.175, -.2])))
        self.assertFalse(memory.motion_blocked(np.array([.175, .175]), np.array([.6, .175])))
        paths = memory.reachable()
        self.assertIn((1, 0), paths)
        self.assertIn((0, -1), paths)

    def test_start_inside_inflation_can_only_escape_towards_increasing_clearance(self):
        memory = self.memory()
        memory.cells[(1, 0)] = (3., 100., 1)
        self.assertIn((-2, 0), memory.reachable())
        self.assertTrue(memory.clear_segment((-2, 0)))
        self.assertFalse(memory.clear_segment((2, 0)))

    def test_failed_turn_tries_opposite_direction_before_reusing_goal(self):
        memory, controller = self.memory(), SocialNavigator()
        controller.survey_complete = True
        controller.feedback('turn_right', 'failed', memory, 100, '预设根运动无法安全通过：侧墙')
        result = controller.step(observation(101), {'maxDistance': .65}, memory,
                                 {'goal': [4, 0], 'bearing': 70, 'intent': 'explore_frontier'}, 101)
        self.assertEqual(result['action'], 'turn_left')
        self.assertEqual(result['intent'], 'inspect_blockage')

    def test_frontier_prefers_new_area_over_nearby_grid_edges(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0., .175]), np.eye(3)
        memory.visits[(0, 0)] = 1
        for x in range(-1, 2):
            for z in range(13):
                memory.cells[(x, z)] = (-2., 100., 1)
        guidance = memory.guidance(100)
        self.assertGreaterEqual(guidance['goal'][1], 4)
        self.assertLess(abs(guidance['bearing']), 18)

    def test_new_area_outranks_walking_back_along_old_footprints(self):
        memory = self.memory()
        for z in range(5):
            for x in range(-4, 5):
                memory.visits[(x, z)] = 2
        guidance = memory.guidance(100)
        self.assertLess(guidance['goal'][1], 0)

    def memory(self):
        memory = ExplorationMemory()
        memory.position, memory.rotation = np.array([.175, 0, .175]), np.eye(3)
        for x in range(-4, 5):
            for z in range(-4, 5):
                memory.cells[(x, z)] = (-2, 100, 1)
        return memory

    def test_target_is_persistent_across_new_frontier_scores(self):
        memory = self.memory()
        first = memory.guidance(100)
        memory.visits[tuple(first['goal'])] = 1000
        second = memory.guidance(101)
        self.assertEqual(first['goal'], second['goal'])
        self.assertGreater(len(second['path']), 1)

    def test_rejected_route_is_not_immediately_reselected(self):
        memory = self.memory()
        first = memory.guidance(100)
        memory.reject_goal(101)
        second = memory.guidance(102)
        self.assertNotEqual(first['goal'], second['goal'])
        self.assertEqual(memory.snapshot()['rejectedGoals'], 1)

    def test_obstacle_blocks_body_width_and_disconnected_regions(self):
        memory = self.memory()
        for z in range(-4, 5):
            memory.cells[(2, z)] = (3, 100, 1)
        paths = memory.reachable()
        self.assertNotIn((1, 0), paths)
        self.assertNotIn((3, 0), paths)

    def test_cache_reuses_evidence_but_obstacles_invalidate_immediately(self):
        memory = self.memory()
        first = memory.reachable()
        memory.mark((2, 0), -.1, 101)
        self.assertIs(memory.reachable(), first)
        memory.mark((2, 0), 4, 102)
        updated = memory.reachable()
        self.assertIsNot(updated, first)
        self.assertNotIn((1, 0), updated)
        self.assertEqual(memory.snapshot()['planner']['cacheHits'], 1)

    def test_weighted_search_never_exceeds_node_budget(self):
        memory = self.memory()
        memory.cells = {(x, z): (-2., 100., 1) for x in range(-50, 50) for z in range(-50, 50)}
        self.assertLessEqual(len(memory.reachable().parents), 6000)

    def test_parent_tree_reconstructs_adjacent_path_and_tracks_position(self):
        memory = self.memory()
        first = memory.reachable()
        path = first.path((3, 3))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(len(path), first.lengths[(3, 3)])
        self.assertTrue(all(b in memory.neighbors(a) for a, b in zip(path, path[1:])))
        memory.position[0] += memory.resolution
        self.assertIsNot(memory.reachable(), first)
        self.assertEqual(memory.reachable().path((3, 3))[0], (1, 0))

    def test_expired_evidence_invalidates_cached_routes(self):
        memory = self.memory()
        first = memory.reachable()
        memory.cleanup(1901)
        updated = memory.reachable()
        self.assertIsNot(updated, first)
        self.assertNotIn((4, 4), updated)

    def test_equal_frontiers_prefer_current_view_instead_of_turning_around(self):
        memory = self.memory()
        guidance = memory.guidance(100)
        self.assertGreater(guidance['goal'][1], 0)
        self.assertLess(abs(guidance['bearing']), 18)

    def test_diagonal_route_looks_ahead_instead_of_chasing_grid_corners(self):
        memory = self.memory()
        route = memory.route_to((3, 2), 'explore_frontier')
        self.assertAlmostEqual(route['bearing'], -math.degrees(math.atan2(3, 2)))
        self.assertGreater(route['distance'], 1.)
        # Small camera sway must not flip the first grid step into a 90-degree correction.
        memory.position[0] += .05
        updated = memory.route_to((3, 2), 'explore_frontier')
        self.assertLess(abs(updated['bearing']-route['bearing']), 5)

    def test_route_smoothing_never_crosses_unknown_or_blocked_corners(self):
        memory = self.memory()
        self.assertTrue(memory.clear_segment((2, 2)))
        del memory.cells[(1, 0)]
        self.assertFalse(memory.clear_segment((2, 2)))
        memory.cells[(1, 0)] = (3, 100, 1)
        self.assertFalse(memory.clear_segment((2, 2)))

    def test_repeated_alignment_replans_instead_of_turning_forever(self):
        memory, controller = self.memory(), SocialNavigator()
        memory.goal = (4, 0)
        guidance = {'goal': [4, 0], 'bearing': 90, 'intent': 'explore_frontier'}
        for stamp in range(100, 104):
            result = controller.explore({'maxDistance': 0}, memory, guidance, stamp)
            self.assertEqual(result['action'], 'turn_right')
        result = controller.explore({'maxDistance': 0}, memory, guidance, 104)
        self.assertEqual(result['intent'], 'replan_frontier')
        self.assertEqual(result['action'], 'observe')
        self.assertGreater(memory.rejected[(4, 0)], 104)

    def test_turning_in_place_is_not_counted_as_failed_translation(self):
        memory = self.memory()
        for _ in range(8):
            memory.record_action('turn_left')
        self.assertEqual(memory.stalls, 0)
        memory.record_action('forward')
        memory.record_action('turn_left')
        self.assertEqual(memory.stalls, 1)

    def test_known_free_retreat_preserves_personal_space(self):
        memory = self.memory()
        goal = memory.retreat_goal({'distance': .5, 'bearing': 0}, 100)
        self.assertIsNotNone(goal)
        self.assertIn(goal, memory.reachable())
        self.assertLess(goal[1], 0)

    def test_blocked_social_route_is_committed_before_facing_person_again(self):
        memory = self.memory()
        memory.cells[(0, 2)] = (3, 100, 1)
        controller = SocialNavigator()
        person = {'id': 'person', 'distance': 3, 'bearing': 0, 'observedAt': 100, 'confidence': .9}
        nav = {'maxDistance': 0}
        result = controller.approach(person, nav, memory, 100)
        self.assertIsNotNone(controller.detour)
        self.assertEqual(result['intent'], 'social_detour')
        frame = observation(101, distance=3)
        result = controller.step(frame, nav, memory, memory.guidance(101), 101)
        self.assertEqual(result['intent'], 'social_detour')


class MissionRuntimeTests(unittest.TestCase):
    def test_memory_reset_during_playback_does_not_break_sync_feedback(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        runtime.sync({'clientId': 'reset', 'state': {}})
        command = runtime.enqueue('visual', {'clientId': 'reset', 'behaviorDecisionId': 'old-decision'})
        runtime.behavior.POLICY.reset('reset')
        runtime.sync({'clientId': 'reset', 'state': {}, 'reports': [{'id': command['id'], 'status': 'interrupted'}]})
        self.assertEqual(runtime.COMMANDS[command['id']]['status'], 'interrupted')

    def test_success_enqueues_one_greeting_instead_of_more_navigation(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        runtime.behavior.POLICY.reset('ready')
        runtime.sync({'clientId': 'ready', 'state': {'position': [0, .92, 0], 'heading': 0}})
        decision = {'decisionId': 'confirmed', 'mission': {'ready': True}, 'reason': '社交条件已达成'}
        with patch.object(runtime.behavior.POLICY, 'decide', return_value=decision), patch.object(runtime.behavior.POLICY, 'feedback_for'):
            result = runtime.autonomy_decide({'clientId': 'ready'})
        self.assertTrue(result['completed'])
        self.assertEqual(len(runtime.COMMANDS), 1)
        command = next(iter(runtime.COMMANDS.values()))
        self.assertEqual(command['kind'], 'gesture')
        self.assertEqual(command['arguments']['action'], 'wave')


if __name__ == '__main__':
    unittest.main(verbosity=2)
