"""Noisy openings, crossing hysteresis and continued exploration regressions."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from exit_memory import ExitMemory
from exploration_drive import ExplorationDrive, wrap
from exploration_memory import ExplorationMemory
from floor_topology import FloorTopology
from pointcloud_exits import PointCloudExits
from search_coverage import SearchCoverage
from semantic_map import SemanticMap
from social_navigation import SocialNavigator


def portal(z=2., confidence=.9):
    return {'position': [0., 0., z], 'normal': [0., 1.], 'width': .9,
            'confidence': confidence, 'uncertaintyMeters': .04, 'kind': 'pointcloud-wall-gap'}


def evidence(candidates, stamp=100):
    return {'frame': 'visual-odometry', 'observedAt': stamp, 'candidates': candidates}


def cloud(with_floor=True, gap=True, frames=4, rotation=0, noise=0):
    world, points = SemanticMap(), []
    for x in np.arange(-2., 2.01, .08):
        for y in np.arange(.24, 1.81, .16):
            if not gap or abs(x) > .45:
                points.append([x, y, 2.])
    if with_floor:
        points.extend([x, 0., z] for x in np.arange(-2., 2.01, .12) for z in np.arange(.5, 4., .12))
    points = np.asarray(points)
    rng = np.random.default_rng(12)
    points += rng.normal(0., noise, points.shape)
    angle = math.radians(rotation)
    matrix = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                       [-math.sin(angle), 0., math.cos(angle)]])
    points = points@matrix.T
    for index, point in enumerate(points):
        world.voxels[index] = {'point': point, 'label': 0, 'frames': frames, 'lastSeen': 100.}
    return world


class PointCloudExitTests(unittest.TestCase):
    def analyze(self, world):
        return PointCloudExits().analyze(world, SimpleNamespace(normal=np.array([0., 1., 0.]), offset=0.),
                                        np.eye(4), 100, SimpleNamespace(fx=360, baseline=.12), {'residual': .01})

    def test_measured_gap_remains_open_without_free_grid_at_threshold(self):
        result = self.analyze(cloud())
        self.assertEqual(len(result['candidates']), 1)
        opening = result['candidates'][0]
        self.assertAlmostEqual(opening['width'], .8, delta=.15)
        self.assertGreater(opening['confidence'], .85)
        self.assertGreater(opening['throughSupport'], 4)
        self.assertEqual(opening['observedFrames'], 4)
        self.assertFalse(result['sealed'])

    def test_sparse_single_frame_or_missing_floor_is_uncertain_not_sealed(self):
        strong = self.analyze(cloud())['candidates'][0]
        result = self.analyze(cloud(with_floor=False, frames=1))
        self.assertTrue(result['candidates'])
        weak = result['candidates'][0]
        self.assertLess(weak['confidence'], strong['confidence'])
        self.assertEqual(weak['state'], 'unconfirmed-opening')
        self.assertFalse(result['sealed'])

    def test_continuous_wall_does_not_invent_a_door(self):
        result = self.analyze(cloud(gap=False))
        self.assertEqual(result['candidates'], [])
        self.assertFalse(result['sealed'], '一面墙不能证明整个房间封闭')

    def test_rotated_noisy_cloud_has_consistent_world_opening(self):
        opening = self.analyze(cloud(rotation=35, noise=.012))['candidates'][0]
        expected = [2*math.sin(math.radians(35)), 0., 2*math.cos(math.radians(35))]
        np.testing.assert_allclose(opening['position'], expected, atol=.15)
        self.assertGreater(opening['width'], .6)

    def test_dynamic_person_surface_does_not_close_gap(self):
        world = cloud()
        world.labels.append('person')
        for index, x in enumerate(np.arange(-.4, .41, .08), start=len(world.voxels)):
            world.voxels[index] = {'point': np.array([x, 1., 2.]), 'label': 1, 'frames': 4, 'lastSeen': 100.}
        self.assertTrue(self.analyze(world)['candidates'])

    def test_low_furniture_edges_are_not_wall_openings(self):
        world = cloud(with_floor=False)
        for voxel in world.voxels.values():
            voxel['point'][1] *= .5
        result = self.analyze(world)
        self.assertEqual(result['candidates'], [])
        self.assertFalse(result['sealed'])

    def test_low_evidence_grid_boundary_cannot_claim_enclosure(self):
        topology = FloorTopology()
        cells = {(x, z): (-2., 100., 1) for x in range(8) for z in range(8)}
        cells.update({(x, z): (.9, 100., 1) for x in range(-1, 9) for z in range(-1, 9) if (x, z) not in cells})
        topology.update(cells, 1, 100)
        self.assertTrue(all(row['boundary'] == 'uncertain-boundary' for row in topology.regions.values()))
        self.assertTrue(all(not row['boundaryBlocksTravel'] for row in topology.regions.values()))

    def test_large_metric_uncertainty_reduces_confidence_even_with_many_points(self):
        analyzer = PointCloudExits()
        points = np.array([[-.8, 2.], [-.6, 2.], [.6, 2.], [.8, 2.]])
        floor = np.array([[0., 2.8], [.1, 3.], [0., 3.3], [-.1, 3.5]])
        gap = (np.array([0., 2.]), np.array([1., 0.]), np.array([0., 1.]), .9, 1.)
        near = analyzer.evidence(gap, points, floor, np.full(4, 10), np.zeros(3), .03)
        far = analyzer.evidence(gap, points, floor, np.full(4, 10), np.zeros(3), .5)
        self.assertLess(far['confidence'], .25)
        self.assertGreater(near['confidence'], far['confidence']*3)

    def test_multiple_rays_in_one_frame_do_not_fake_multiframe_boundary_support(self):
        memory = ExplorationMemory()
        for _ in range(10):
            memory.mark((0, 0), .9, 100)
        self.assertEqual(memory.cells[(0, 0)][2], 1)
        memory.mark((0, 0), .9, 101)
        revision = memory.map_revision
        memory.mark((0, 0), .9, 102)
        self.assertEqual(memory.cells[(0, 0)][2], 3)
        self.assertGreater(memory.map_revision, revision)

    def test_voxel_frame_support_does_not_count_multiple_pixels_as_frames(self):
        from test_stereo import SemanticMapTests
        world, frame = SemanticMap(), SemanticMapTests().frame()
        world.integrate(frame, np.eye(4), 100)
        world.integrate(frame, np.eye(4), 100)
        self.assertEqual({voxel['frames'] for voxel in world.voxels.values()}, {1})
        world.integrate(frame, np.eye(4), 101)
        self.assertEqual({voxel['frames'] for voxel in world.voxels.values()}, {2})


class ExitMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = ExitMemory()
        self.memory.integrate(evidence([portal()]), 100)

    def cross(self):
        for index, z in enumerate((.8, 1.4, 1.9, 2., 2.1, 2.7, 3.3)):
            self.memory.observe(np.array([0., 0., z]), 100+index)

    def test_crossing_is_recorded_once_with_plane_zero_and_hysteresis(self):
        self.cross()
        self.assertEqual(len(self.memory.crossings), 1)
        for index, z in enumerate((2.2, 1.9, 2.1, 2.2)):
            self.memory.observe(np.array([0., 0., z]), 108+index)
        self.assertEqual(len(self.memory.crossings), 1)

    def test_recent_return_is_penalized_but_new_direction_and_late_return_allowed(self):
        self.cross()
        self.assertGreater(self.memory.return_penalty([0., 0., 3.], [0., 0., .8], 108), 10)
        self.assertEqual(self.memory.return_penalty([0., 0., 3.], [0., 0., 6.], 108), 0)
        late = self.memory.return_penalty([0., 0., 3.], [0., 0., .8], 230)
        self.assertGreater(late, 0, '冷却结束仍应记得这是走过的门，但只降权、不封闭')
        self.assertLess(late, self.memory.return_penalty([0., 0., 3.], [0., 0., .8], 108))

    def test_walking_around_wall_end_is_not_portal_crossing(self):
        for index, z in enumerate((1., 1.5, 2.1, 2.7)):
            self.memory.observe(np.array([2., 0., z]), 100+index)
        self.assertFalse(self.memory.crossings)

    def test_repartition_and_noisy_reobservations_do_not_erase_crossing(self):
        self.cross()
        candidate = {**portal(), 'position': [.1, 0., 2.1], 'normal': [0., -1.]}
        self.memory.integrate(evidence([candidate], 108), 108)
        self.assertEqual(len(self.memory.portals), 1)
        self.assertGreater(self.memory.return_penalty([0., 0., 3.], [0., 0., 1.], 108), 10)

    def test_invalid_and_stale_inputs_cannot_create_portals(self):
        self.memory.integrate(evidence([portal(8)], 90), 100)
        self.assertEqual(len(self.memory.portals), 1)
        with self.assertRaises(ValueError):
            self.memory.integrate(evidence([{**portal(), 'normal': [0., float('nan')]}]), 100)

    def test_recent_travel_also_discourages_return_without_portal_detection(self):
        memory = ExitMemory()
        for index, z in enumerate(np.arange(0., 4., .4)):
            memory.observe(np.array([0., 0., z]), 100+index)
        backward = memory.travel_penalty([0., 0., 4.], [0., 0., 1.], 112)
        forward = memory.travel_penalty([0., 0., 4.], [0., 0., 7.], 112)
        self.assertGreater(backward-forward, 7)

    def test_same_visible_door_does_not_pull_character_back_after_crossing(self):
        self.cross()
        memory, drive = ExplorationMemory(), ExplorationDrive()
        memory.exit_memory = self.memory
        memory.position, memory.rotation = np.array([0., 0., 3.3]), np.diag([-1., 1., -1.])
        frame = {'observedAt': 108, 'viewHeading': 180, 'bodyHeading': 180,
                 'stereo': {'mapping': True}, 'objects': [{'label': 'door', 'confidence': .9, 'box': [250, 10, 390, 350]}]}
        drive.observe_heading(frame)
        result = drive.plan(frame, {'bodyBearing': 0, 'sectors': []}, memory, 108)
        target = drive.world_target(result['bearing'], result['distance'], memory)
        self.assertGreater(target[2], memory.position[2])

    def test_exit_target_is_kept_across_candidate_loss_and_collision_clears_it(self):
        memory, drive = ExplorationMemory(), ExplorationDrive()
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        memory.exit_memory = self.memory
        frame = {'observedAt': 100, 'stereo': {'mapping': True}}
        drive.plan(frame, {'bodyBearing': 0}, memory, 100)
        goal = drive.commit_target.copy()
        memory.position = np.array([0., 0., 1.])
        memory.exit_memory.portals.clear()
        result = drive.plan(frame, {'bodyBearing': 0}, memory, 105)
        self.assertEqual(result['intent'], 'seek_possible_exit')
        np.testing.assert_array_equal(goal, drive.commit_target)
        drive.failure(memory, 106)
        self.assertIsNone(drive.commit_target)
        self.assertEqual(drive.recovery()['intent'], 'escape_deadlock')

    def test_look_and_execution_failure_do_not_close_frontier(self):
        coverage = SearchCoverage()
        item = coverage.mark_exit([.1, .1], 2, 100)
        coverage.observe(np.zeros(3), np.eye(3), [], 101)
        self.assertEqual(item['status'], 'unvisited')
        self.assertEqual(item['lookedAt'], 101)
        memory = ExplorationMemory()
        memory.coverage, memory.goal = coverage, (0, 0)
        memory.reject_goal(102)
        self.assertEqual(item['status'], 'attempted')
        self.assertNotEqual(item['status'], 'blocked')

    def test_known_route_does_not_bypass_return_penalty(self):
        memory, navigator = ExplorationMemory(), SocialNavigator()
        memory.position, memory.rotation = np.array([0., 0., 4.]), np.eye(3)
        for index, z in enumerate(np.arange(0., 4.01, .4)):
            memory.exit_memory.observe(np.array([0., 0., z]), 100+index)
        frame = {'observedAt': 115, 'viewHeading': 0, 'bodyHeading': 0, 'stereo': {'mapping': True}}
        route = {'available': True, 'goal': [0, 3], 'distance': 3., 'bearing': 180, 'intent': 'explore_frontier'}
        navigator.drive.observe_heading(frame)
        result = navigator.explore_aggressively(frame, {'bodyBearing': 0, 'visualAvoidance': False}, memory, route, 115)
        self.assertLess(abs(result['relativeBearing']), 90)

    def test_failed_exit_target_is_temporarily_demoted_not_closed(self):
        memory, drive = ExplorationMemory(), ExplorationDrive()
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        drive.commit_target = np.array([0., 0., 3.])
        drive.failure(memory, 100)
        self.assertGreater(drive.attempt_penalty(np.array([0., 0., 3.]), 101), 10)
        self.assertEqual(drive.attempt_penalty(np.array([3., 0., 0.]), 101), 0)
        self.assertEqual(drive.attempt_penalty(np.array([0., 0., 3.]), 191), 0)

    def test_grid_frontier_is_not_promoted_to_a_direct_doorway(self):
        memory, drive = ExplorationMemory(), ExplorationDrive()
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        memory.coverage.mark_exit([0., 2.], 2, 100)
        frame = {'observedAt': 100, 'stereo': {'mapping': True}}
        self.assertEqual(drive.exit_candidates(frame, memory), [])

    def test_failed_retreat_uses_side_direction_instead_of_repeating_half_turns(self):
        memory, drive = ExplorationMemory(), ExplorationDrive()
        memory.position, memory.rotation = np.zeros(3), np.eye(3)
        drive.failure(memory, 100)
        self.assertEqual(abs(drive.back_heading), 180)
        drive.heading = drive.view = 180
        self.assertEqual(drive.recovery()['intent'], 'leave_deadlock')
        drive.failure(memory, 101)
        self.assertEqual(abs(wrap(drive.back_heading-drive.heading)), 90)
        self.assertFalse(drive.blocked(drive.back_heading, memory, 101))

    def test_uncertain_portal_crossing_does_not_get_strong_return_penalty(self):
        self.cross()
        start, target = [0., 0., 3.], [0., 0., 1.]
        strong = self.memory.return_penalty(start, target, 108)
        self.memory.portals[1]['confidence'] = .1
        weak = self.memory.return_penalty(start, target, 108)
        self.assertLess(weak, strong/5)
        self.assertGreater(weak, 0)


if __name__ == '__main__':
    unittest.main()
