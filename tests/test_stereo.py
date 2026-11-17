import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from semantic_map import SemanticMap
from stereo_depth import Calibration, DepthFrame, StereoDepth, clearance_sectors, ground_estimate, _horizontal_view, epipolar_texture
from visual_navigation import depth_navigation
from visual_odometry import VisualOdometry
from stereo_service import StereoSession, StereoSessions
from navigation_grid import navigation_grid
from floor_tracking import FloorTracker


class StereoDepthTests(unittest.TestCase):
    def test_recorded_floor_stripe_does_not_create_a_near_obstacle(self):
        import json
        fixture = Path(__file__).parent/'fixtures'/'stereo-floor'
        record = json.loads((fixture/'calibration.json').read_text())
        left, right = [cv2.imread(str(fixture/f'{eye}.jpg')) for eye in ('left', 'right')]
        calibration = Calibration.read(record['calibration'], left.shape[:2])
        frame = StereoDepth().infer(left, right, calibration)
        rig = np.asarray(record['rig']['opticalToControl']).reshape(4, 4)
        sectors, plane = clearance_sectors(frame, rig)
        self.assertIsNotNone(plane)
        self.assertGreater(sectors[3]['clearance'], 2.)
        self.assertIsNone(sectors[3]['obstacleDistance'])

    def test_horizontal_stripes_alone_cannot_support_horizontal_stereo(self):
        rows = np.arange(180)[:, None]
        stripes = np.repeat(np.uint8(100+80*np.sin(rows*.5)), 320, axis=1)
        self.assertFalse(epipolar_texture(stripes).any())
        stripes[:, 155:165] = 240
        self.assertGreater(np.sum(epipolar_texture(stripes)), 500)

    def test_visual_floor_memory_rejects_dominant_tabletop_and_never_fills_missing_floor(self):
        tracker = FloorTracker()
        rig = np.diag([1., -1., -1., 1.])
        rig[1, 3] = 1.6
        pose = rig.copy()
        tracker.update({'normal': np.array([0., 1., 0.]), 'offset': 0.,
                        'support': 400, 'tiltDeg': 0., 'residual': .01}, pose, rig)
        x, z = np.meshgrid(np.linspace(-2, 2, 45), np.linspace(-4, -.5, 45))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        table = np.repeat(floor, 3, axis=0)+[0., .7, 0.]
        optical = lambda cloud: (cloud-rig[:3, 3])@rig[:3, :3]
        prior = tracker.control_prior(pose, rig)
        ground = ground_estimate(optical(np.vstack((floor, table))), rig, prior)
        self.assertAlmostEqual(ground['coefficients'][2], 0., places=5)
        self.assertIsNone(ground_estimate(optical(table), rig, prior))
        moved = rig.copy()
        moved[:3, 3] += [12, 20, 30]
        normal, offset = tracker.control_prior(pose, moved)
        self.assertAlmostEqual(float(normal@np.array([12., 20., 30.])+offset), 0., places=5)

    def test_obstacles_keep_their_measured_lateral_position_in_visual_grid(self):
        x, z = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-4, -.5, 90))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        y, z = np.meshgrid(np.linspace(.3, 1.8, 40), np.linspace(-2.2, -1.8, 20))
        wall = np.column_stack((np.full(y.size, .85), y.ravel(), z.ravel()))
        rig = np.diag([1., -1., -1., 1.])
        rig[1, 3] = 1.6
        optical = (np.vstack((floor, wall))-rig[:3, 3])@rig[:3, :3]
        frame = SimpleNamespace(points=lambda: (optical, None, None, None), calibration=self.calibration)
        ground = {'coefficients': np.array([0., 0., 0.])}
        grid = navigation_grid(frame, np.diag([1., -1., -1., 1.]), rig, ground)
        self.assertTrue(grid['occupied'])
        self.assertTrue(all(point[0] > .7 for point in grid['occupied']))
        shifted = rig.copy()
        shifted[:3, 3] += [30, 40, 50]
        same = navigation_grid(frame, np.diag([1., -1., -1., 1.]), shifted,
                               {'coefficients': np.array([0., 0., 40.])})
        self.assertEqual(grid, same)

    def test_current_floor_can_recover_from_vo_height_drift_without_accepting_table(self):
        tracker = FloorTracker()
        rig = np.diag([1., -1., -1., 1.]); rig[1, 3] = 1.6
        foot = np.array([0., 1.6, 0.])
        tracker.update({'normal': np.array([0., 1., 0.]), 'offset': 0.,
                        'support': 400, 'tiltDeg': 0., 'residual': .01}, rig, rig, foot)
        drifted = rig.copy(); drifted[1, 3] += .35
        x, z = np.meshgrid(np.linspace(-2, 2, 35), np.linspace(-4, -.5, 35))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        optical = lambda points: (points-rig[:3, 3])@rig[:3, :3]
        self.assertIsNone(ground_estimate(optical(floor), rig, tracker.control_prior(drifted, rig)))
        prior = tracker.body_prior(rig, foot)
        current = ground_estimate(optical(floor), rig, prior)
        self.assertGreater(current['support'], 80)
        self.assertIsNone(ground_estimate(optical(floor+[0., .7, 0.]), rig, prior))
        self.assertIsNone(ground_estimate(np.empty((0,3)), rig, prior))
        shifted = rig.copy(); shifted[:3, 3] += [10,20,30]
        normal, offset = tracker.body_prior(shifted, foot)
        self.assertAlmostEqual(float(normal@np.array([10.,20.,30.])+offset), 0., places=6)

    def test_page_release_frees_capacity_without_deleting_other_sessions(self):
        sessions = StereoSessions()
        for i in range(4):
            sessions.session({'clientId': str(i)})
        with self.assertRaises(BlockingIOError):
            sessions.session({'clientId': 'new-page'})
        self.assertTrue(sessions.release({'clientId': '1'})['released'])
        self.assertFalse(sessions.release({'clientId': '1'})['released'])
        sessions.session({'clientId': 'new-page'})
        self.assertEqual(set(sessions.sessions), {'0', '2', '3', 'new-page'})

    def setUp(self):
        self.width, self.height = 320, 180
        self.calibration = Calibration(self.width, self.height, 240., 240., 160., 90., .12)

    def test_calibration_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            Calibration.read({'width': 320, 'height': 180, 'fx': 240, 'fy': 240,
                              'cx': 160, 'cy': 90, 'baseline': .12}, (90, 320))

    def test_textured_plane_recovers_metric_depth(self):
        rng = np.random.default_rng(4)
        left_gray = rng.integers(20, 235, (self.height, self.width), dtype=np.uint8)
        left_gray = cv2.GaussianBlur(left_gray, (3, 3), 0)
        disparity = 24
        right_gray = np.zeros_like(left_gray)
        right_gray[:, :-disparity] = left_gray[:, disparity:]
        left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
        frame = StereoDepth().infer(left, right, self.calibration)
        expected = self.calibration.fx*self.calibration.baseline/disparity
        measured = frame.depth[frame.valid]
        self.assertGreater(len(measured), 1000)
        self.assertAlmostEqual(float(np.median(measured)), expected, delta=.12)

    def test_blank_pair_is_unknown(self):
        blank = np.zeros((self.height, self.width, 3), np.uint8)
        frame = StereoDepth().infer(blank, blank, self.calibration)
        self.assertEqual(int(frame.valid.sum()), 0)
        sectors, plane = clearance_sectors(frame)
        self.assertIsNone(plane)
        self.assertTrue(all(item['confidence'] == 0 for item in sectors))
        nav = depth_navigation({'method': 'stereo-sgbm-lr', 'observedAt': time.time(),
                                'mapping': True, 'sectors': sectors})
        self.assertFalse(nav['valid'])
        self.assertEqual(nav['maxDistance'], 0.)

    def test_tilted_optical_view_is_levelled_by_rig(self):
        angle = np.deg2rad(30.)
        pitch = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                          [0., np.sin(angle), np.cos(angle)]])
        rig = np.eye(4)
        rig[:3, :3] = pitch@np.diag([1., -1., -1.])
        rig[:3, 3] = [0., 1.6, 0.]
        x, z = np.meshgrid(np.linspace(-2., 2., 80), np.linspace(-5., -.5, 80))
        world = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        optical = (world-rig[:3, 3])@rig[:3, :3]
        estimate = ground_estimate(optical, rig)
        self.assertIsNotNone(estimate)
        self.assertLess(estimate['tiltDeg'], .1)
        np.testing.assert_allclose(estimate['coefficients'], [0., 0., 0.], atol=1e-3)

    def test_clearance_is_invariant_to_heading_and_world_translation(self):
        x, z = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-6, -.5, 100))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        x, y = np.meshgrid(np.linspace(-.5, .5, 30), np.linspace(.1, 1.8, 30))
        wall = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -2.)))
        points = np.vstack((floor, wall))-[0., 1.6, 0.]
        optical = points@np.diag([1., -1., -1.])
        frame = SimpleNamespace(points=lambda: (optical, None, None, None))
        baseline = None
        for degrees in (0, 45, 90, 180, 270):
            yaw = cv2.Rodrigues(np.array([0., np.deg2rad(degrees), 0.]))[0]
            rig = np.eye(4)
            rig[:3, :3] = yaw@np.diag([1., -1., -1.])
            rig[:3, 3] = [30., 1.6, 40.]
            sectors, plane = clearance_sectors(frame, rig)
            self.assertIsNotNone(plane)
            values = np.array([item['clearance'] for item in sectors], dtype=float)
            self.assertAlmostEqual(values[3], 1.85, places=3)
            if baseline is None:
                baseline = values
            np.testing.assert_allclose(values, baseline, atol=.001)

    def test_vertical_camera_cannot_claim_horizontal_clearance(self):
        rig = np.eye(4)
        rig[:3, 2] = [0., 1., 0.]
        self.assertIsNone(_horizontal_view(np.zeros((3, 3)), rig))

    def test_disparity_noise_on_distant_floor_does_not_seal_the_passage(self):
        x, z = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-6, -.6, 120))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        noisy = floor[(floor[:, 2] < -3) & (floor[:, 2] > -4)].copy()
        noisy[:, 1] = .15
        obstacle_x, obstacle_y = np.meshgrid(np.linspace(-.3, .3, 20), np.linspace(.3, .8, 20))
        obstacle = np.column_stack((obstacle_x.ravel(), obstacle_y.ravel(), np.full(obstacle_x.size, -1.5)))
        rig = np.diag([1., -1., -1., 1.])
        rig[1, 3] = 1.6

        def frame(points):
            optical = (points-rig[:3, 3])@rig[:3, :3]
            return SimpleNamespace(points=lambda: (optical, None, None, None), calibration=self.calibration)

        clear, _ = clearance_sectors(frame(np.vstack((floor, noisy))), rig)
        self.assertIsNone(clear[3]['obstacleDistance'])
        self.assertGreater(clear[3]['clearance'], 4)
        blocked, _ = clearance_sectors(frame(np.vstack((floor, noisy, obstacle))), rig)
        self.assertLess(blocked[3]['clearance'], 1.5)


class SemanticMapTests(unittest.TestCase):
    def frame(self):
        calibration = Calibration(320, 180, 240., 240., 160., 90., .12)
        depth = np.full((180, 320), 5., np.float32)
        depth[60:120, 110:170] = 2.
        rgb = np.full((180, 320, 3), 120, np.uint8)
        return DepthFrame(rgb, rgb[:, :, 0], depth, depth > 0, np.zeros_like(depth, np.uint8), calibration)

    def test_surface_annotation_excludes_majority_background(self):
        frame, world, pose = self.frame(), SemanticMap(), np.eye(4)
        world.integrate(frame, pose, 1.)
        detection = {'box': [80, 45, 200, 135], 'label': 'chair', 'confidence': .9}
        result = world.annotate(frame, pose, [detection], 1.)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['depth'], 2.)
        self.assertGreater(len(result[0]['boundary']), 2)
        labeled = [v for v in world.voxels.values() if v['label']]
        self.assertGreater(len(labeled), 10)
        self.assertTrue(all(abs(v['point'][2]-2) < .01 for v in labeled))
        repeated = world.annotate(frame, pose, [detection], 1.1)
        self.assertEqual(result[0]['id'], repeated[0]['id'])

    def test_voxel_limit_and_expiration(self):
        world = SemanticMap(limit=100)
        world.integrate(self.frame(), np.eye(4), 1.)
        self.assertLessEqual(len(world.voxels), 100)
        self.assertGreater(world.snapshot(47.)['totalPoints'], 0)
        self.assertTrue(world.snapshot(47.)['persistent'])
        self.assertEqual(world.snapshot(1802.)['totalPoints'], 0)

    def test_point_memory_cleanup_skips_duplicate_frame_and_rechecks_movement(self):
        world = SemanticMap()
        world.integrate(self.frame(), np.eye(4), 1.)
        original = len(world.voxels)
        world._expire(1.1, np.zeros(3))
        self.assertEqual(len(world.voxels), original)
        world._expire(1.2, np.array([30., 0., 0.]))
        self.assertEqual(len(world.voxels), 0)

    def test_same_frame_detections_keep_distinct_instances(self):
        world = SemanticMap()
        points = np.random.default_rng(4).uniform(-.1, .1, (40, 3))
        first = world._object('chair', points, .9, 1., set())
        second = world._object('chair', points+.1, .9, 1., {first['id']})
        self.assertNotEqual(first['id'], second['id'])


class NavigationTests(unittest.TestCase):
    def observation(self, front=3.):
        return {'method': 'stereo-sgbm-lr', 'observedAt': time.time(), 'mapping': True,
                'sectors': [{'angle': a, 'clearance': front, 'confidence': .9} for a in [-30, 0, 30]]}

    def test_obstacle_stale_and_lost_cannot_move_forward(self):
        close = self.observation(.5)
        stale = self.observation()
        stale['observedAt'] -= 2
        lost = {**self.observation(), 'mapping': False}
        for raw in (close, stale, lost):
            self.assertEqual(depth_navigation(raw)['maxDistance'], 0.)
        self.assertGreater(depth_navigation(self.observation())['maxDistance'], .5)


def render_boards(calibration, camera_x=0., camera_z=0.):
    """Known motion is only a test oracle. The runtime receives RGB alone."""
    output = np.zeros((calibration.height, calibration.width, 3), np.uint8)
    rng = np.random.default_rng(81)
    boards = [(-7, -4, 7, 4, 6), (-1.8, -.8, -.25, .9, 2.8), (.25, -.7, 1.7, .65, 3.7)]
    source = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], np.float32)
    for x0, y0, x1, y1, z in boards:
        noise = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        texture = cv2.cvtColor(cv2.GaussianBlur(noise, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        projected = (corners-[camera_x, 0])*[calibration.fx, calibration.fy]/(z-camera_z)
        projected += [calibration.cx, calibration.cy]
        transform = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
        warped = cv2.warpPerspective(texture, transform, (calibration.width, calibration.height))
        mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), transform,
                                   (calibration.width, calibration.height))
        output[mask > 0] = warped[mask > 0]
    return output


class VisualOdometryTests(unittest.TestCase):
    def test_failed_pose_keeps_pixel_tracks_without_publishing_unverified_motion(self):
        calibration = Calibration(640, 360, 400., 400., 320., 180., .12)
        stereo, odometry = StereoDepth(), VisualOdometry()
        for index in range(5):
            left = render_boards(calibration, index*.025, index*.01)
            right = render_boards(calibration, index*.025+.12, index*.01)
            frame = stereo.infer(left, right, calibration)
            if 0 < index < 4:
                with patch.object(odometry, '_solve', return_value=None):
                    result = odometry.update(frame, 100+index*.2)
                self.assertEqual(result['status'], 'lost')
                np.testing.assert_allclose(odometry.pose[:3, 3], [0, 0, 0])
                np.testing.assert_array_equal(odometry.reference.gray, frame.gray)
                self.assertEqual(odometry.reference.timestamp, 100.)
            else:
                result = odometry.update(frame, 100+index*.2)
        self.assertEqual(result['status'], 'tracking')
        np.testing.assert_allclose(odometry.pose[:3, 3], [.1, 0, -.04], atol=.025)

    def test_real_roaming_keeps_metric_landmarks_through_weak_texture_turn(self):
        import json
        fixture = Path(__file__).parent/'fixtures'/'weak-texture-roaming'
        records = json.loads((fixture/'frames.json').read_text())
        odometry, stereo, origin = VisualOdometry(), StereoDepth(), None
        for index, record in enumerate(records):
            left, right = [cv2.imread(str(fixture/f'{index}-{eye}.jpg')) for eye in ('left', 'right')]
            calibration = Calibration.read(record['calibration'], left.shape[:2])
            rig = np.asarray(record['rig']['opticalToControl']).reshape(4, 4)
            # World translation is an oracle only. Localization gets arbitrary translation.
            calibration_rig = rig.copy()
            calibration_rig[:3, 3] = [index*17, -index*3, index*29]
            result = odometry.update(stereo.infer(left, right, calibration), record['observedAt'], calibration_rig)
            if origin is None:
                origin = odometry.pose@np.linalg.inv(rig)
            self.assertIn(result['status'], ('initialized', 'tracking', 'relocalized'))
            np.testing.assert_allclose(odometry.pose[:3, 3], (origin@rig)[:3, 3], atol=.04)

    def test_temporal_landmarks_survive_repeated_depth_dropouts(self):
        calibration = Calibration(640, 360, 400., 400., 320., 180., .12)
        stereo, odometry = StereoDepth(), VisualOdometry()
        for index in range(7):
            left = render_boards(calibration, index*.02, index*.01)
            right = render_boards(calibration, index*.02+.12, index*.01)
            frame = stereo.infer(left, right, calibration)
            if index:
                frame.valid[:] = False
                frame.depth[:] = 0
            result = odometry.update(frame, 100+index*.2)
            self.assertIn(result['status'], ('initialized', 'tracking'))
        np.testing.assert_allclose(odometry.pose[:3, 3], [.12, 0., -.06], atol=.025)

    def test_real_low_texture_turn_retains_seventeen_valid_landmarks(self):
        import json
        fixture = Path(__file__).parent/'fixtures'/'low-texture-turn'
        records = json.loads((fixture/'frames.json').read_text())
        odometry, stereo, origin = VisualOdometry(), StereoDepth(), None
        for index, record in enumerate(records):
            left, right = [cv2.imread(str(fixture/f'{index}-{eye}.jpg')) for eye in ('left', 'right')]
            calibration = Calibration.read(record['calibration'], left.shape[:2])
            rig = np.asarray(record['rig']['opticalToControl']).reshape(4, 4)
            result = odometry.update(stereo.infer(left, right, calibration), record['observedAt'], rig)
            if origin is None:
                origin = odometry.pose@np.linalg.inv(rig)
        self.assertEqual(result['status'], 'tracking')
        self.assertGreaterEqual(result['inliers'], 15)
        # Rig positions are an independent test oracle, never a localization input.
        np.testing.assert_allclose(odometry.pose[:3, 3], (origin@rig)[:3, 3], atol=.05)

    def test_relocalization_search_is_bounded_and_retrieves_older_views(self):
        odometry = VisualOdometry()
        target = np.full((30, 32), 255, np.uint8)
        older = SimpleNamespace(signature=odometry._signature(target))
        recent = [SimpleNamespace(signature=np.zeros(256)) for _ in range(95)]
        odometry.keys = [older, *recent]
        candidates = odometry._candidates(target)
        self.assertLessEqual(len(candidates), 16)
        self.assertTrue(any(key is older for key in candidates))

    def test_optical_flow_keeps_pose_when_current_depth_features_drop_out(self):
        calibration = Calibration(640, 360, 400., 400., 320., 180., .12)
        stereo, odometry = StereoDepth(), VisualOdometry()
        left = render_boards(calibration)
        right = render_boards(calibration, .12)
        odometry.update(stereo.infer(left, right, calibration), 100.)
        left = render_boards(calibration, .02, .01)
        right = render_boards(calibration, .14, .01)
        frame = stereo.infer(left, right, calibration)
        frame.valid[:] = False
        frame.depth[:] = 0
        result = odometry.update(frame, 100.2)
        self.assertEqual(result['status'], 'tracking')
        self.assertGreaterEqual(result['inliers'], 15)
        np.testing.assert_allclose(odometry.pose[:3, 3], [.02, 0., -.01], atol=.015)

    def test_metric_motion_and_tracking_loss(self):
        calibration = Calibration(640, 360, 400., 400., 320., 180., .12)
        stereo, odometry = StereoDepth(), VisualOdometry()
        for index in range(6):
            x, z = index*.02, index*.01
            left = render_boards(calibration, x, z)
            right = render_boards(calibration, x+.12, z)
            result = odometry.update(stereo.infer(left, right, calibration), 100+index*.2)
            self.assertIn(result['status'], ('initialized', 'tracking', 'relocalized'))
        error = np.linalg.norm(odometry.pose[:3, 3]-[.1, 0., -.05])
        self.assertLess(error, .035)
        blank = np.zeros_like(left)
        result = odometry.update(stereo.infer(blank, blank, calibration), 102.)
        self.assertEqual(result['status'], 'lost')
        recovered = odometry.update(stereo.infer(left, right, calibration), 102.2)
        self.assertIn(recovered['status'], ('tracking', 'relocalized'))

    def test_stale_and_duplicate_session_frames_rejected(self):
        session = StereoSession()
        with self.assertRaisesRegex(ValueError, '过期'):
            session.infer({'observedAt': time.time()-3, 'frameId': 1})
        session.last_frame = 2
        with self.assertRaisesRegex(ValueError, '递增'):
            session.infer({'observedAt': time.time(), 'frameId': 2})


if __name__ == '__main__':
    unittest.main(verbosity=2)
