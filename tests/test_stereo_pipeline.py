"""Regression cases for planar reconstruction and asynchronous semantic fusion."""
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from pointcloud_exits import PointCloudExits
from semantic_map import SemanticMap
from stereo_depth import Calibration, DepthFrame, StereoDepth
from stereo_service import StereoSession
from surface_completion import complete_surfaces
from touch_vision import surface


def supported_wall(slanted=False):
    height, width = 180, 320
    calibration = Calibration(width, height, 240., 240., 160., 90., .12)
    rgb = np.full((height, width, 3), 160, np.uint8)
    valid = np.zeros((height, width), bool)
    valid[20:160, 40:55] = valid[20:160, 265:280] = True
    valid[20:35, 40:280] = valid[145:160, 40:280] = True
    rows, cols = np.indices(valid.shape)
    disparity = 12.+(cols-160)*.015+(rows-90)*.008 if slanted else np.full(valid.shape, 12.)
    expected = (calibration.fx*calibration.baseline/disparity).astype(np.float32)
    frame = DepthFrame(rgb, rgb[:, :, 0], np.where(valid, expected, 0), valid,
                       np.zeros(valid.shape, np.uint8), calibration, valid.copy())
    return frame, expected


class SurfaceCompletionTests(unittest.TestCase):
    def test_rgb_stereo_completes_blank_center_from_peripheral_texture(self):
        frame, expected = supported_wall()
        left = frame.rgb.copy()
        noise = np.random.default_rng(2).integers(100, 220, frame.valid.shape, dtype=np.uint8)
        left[frame.valid] = noise[frame.valid, None]
        right = np.full_like(left, 160)
        right[:, :-12] = left[:, 12:]
        result = StereoDepth().infer(left, right, frame.calibration)
        self.assertLess(result.valid[45:135, 65:255].mean(), .01)
        self.assertGreater((result.reconstruction_depth[45:135, 65:255] > 0).mean(), .95)
        np.testing.assert_allclose(result.reconstruction_depth[45:135, 65:255], expected[45:135, 65:255], atol=.03)

    def test_uniform_interior_recovers_metric_plane_without_promoting_measurements(self):
        frame, expected = supported_wall()
        original_depth, original_valid = frame.depth.copy(), frame.valid.copy()
        original_points = frame.points()[0].copy()
        complete_surfaces(frame, frame.rgb)
        np.testing.assert_allclose(frame.reconstruction_depth[45:135, 65:255], expected[45:135, 65:255], atol=.001)
        np.testing.assert_array_equal(frame.depth, original_depth)
        np.testing.assert_array_equal(frame.valid, original_valid)
        np.testing.assert_array_equal(frame.points()[0], original_points)
        self.assertGreater(len(frame.mapping_points()[0]), len(frame.points(stride=4)[0])*2)
        self.assertEqual(frame.completion_info['usage'], 'reconstruction-only')
        self.assertEqual(int(frame.reconstruction_depth[:15].sum()), 0)

    def test_slanted_wall_uses_inverse_depth_plane(self):
        frame, expected = supported_wall(slanted=True)
        complete_surfaces(frame, frame.rgb)
        np.testing.assert_allclose(frame.reconstruction_depth[45:135, 65:255], expected[45:135, 65:255], atol=.001)

    def test_blank_pair_has_no_plane_or_inferred_depth(self):
        frame, _ = supported_wall()
        blank = np.full_like(frame.rgb, 160)
        result = StereoDepth().infer(blank, blank, frame.calibration)
        self.assertFalse(result.valid.any())
        self.assertFalse(result.reconstruction_depth.any())

    def test_inferred_interior_cannot_authorize_contact(self):
        frame, _ = supported_wall()
        complete_surfaces(frame, frame.rgb)
        self.assertGreater(frame.reconstruction_depth[90, 160], 0)
        with self.assertRaisesRegex(ValueError, '双目深度不足'):
            surface(frame, {'pixel': [160, 90], 'confidence': .9}, np.eye(4))

    def test_doorway_color_boundary_stays_unknown(self):
        frame, _ = supported_wall()
        frame.rgb[65:140, 130:195] = 40
        complete_surfaces(frame, frame.rgb)
        self.assertFalse(frame.reconstruction_depth[70:135, 135:190].any())
        self.assertGreater(frame.completion_info['inferredPixels'], 5000)

    def test_two_competing_depth_planes_cannot_fill_uniform_gap(self):
        frame, _ = supported_wall()
        frame.depth[:, 160:][frame.valid[:, 160:]] *= 2
        complete_surfaces(frame, frame.rgb)
        self.assertEqual(frame.completion_info['inferredPixels'], 0)

    def test_collinear_support_cannot_define_wall_interior(self):
        frame, _ = supported_wall()
        frame.valid[:] = frame.surface_valid[:] = False
        frame.valid[88:92, 40:280] = frame.surface_valid[88:92, 40:280] = True
        frame.depth[:] = np.where(frame.valid, 2.4, 0)
        complete_surfaces(frame, frame.rgb)
        self.assertEqual(frame.completion_info['inferredPixels'], 0)

    def test_conflicting_weak_depth_and_right_eye_color_reject_completion(self):
        frame, _ = supported_wall()
        frame.valid[80:100, 145:175] = True
        frame.depth[80:100, 145:175] = 1.
        other = frame.rgb.copy()
        other[45:65, 100:140] = 40
        complete_surfaces(frame, other)
        self.assertFalse(frame.reconstruction_depth[80:100, 145:175].any())
        self.assertFalse(frame.reconstruction_depth[45:65, 112:152].any())

    def test_inferred_wall_does_not_supply_exit_or_navigation_evidence(self):
        frame, _ = supported_wall()
        complete_surfaces(frame, frame.rgb)
        world = SemanticMap()
        world.integrate(frame, np.eye(4), 100.)
        for voxel in world.voxels.values():
            if voxel.get('measured', True):
                voxel['lastSeen'] = -2000.
        floor = SimpleNamespace(normal=np.array([0., 1., 0.]), offset=1.6)
        walls, floor_points, frames = PointCloudExits().surfaces(world, floor, np.zeros(3), 100.)
        self.assertEqual(len(walls)+len(floor_points)+len(frames), 0)


class SemanticPipelineTests(unittest.TestCase):
    def test_connected_cloud_builds_continuous_surface_and_stable_color(self):
        world = SemanticMap(voxel_size=.08)
        x, y = np.meshgrid(np.arange(-.8, .8, .08), np.arange(.4, 1.8, .08))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2.)))
        colors = np.tile(np.array([120., 150., 180.]), (len(points), 1))
        colors[::17] = [245., 5., 5.]
        world._merge(points, colors, np.ones(len(points), bool), 100.)
        snapshot = world.snapshot(100.)
        self.assertEqual(len(snapshot['volumes']), 1)
        self.assertEqual(len(snapshot['surfaces']), 1)
        surface = snapshot['surfaces'][0]
        self.assertTrue(surface['continuous'])
        self.assertEqual(surface['color'], (120 << 16) | (150 << 8) | 180)
        self.assertGreaterEqual(len(surface['vertices']), 9)

    def test_disconnected_voxels_are_separate_volumes(self):
        world = SemanticMap(voxel_size=.08)
        first = np.array([[0., 0., 2.], [.08, 0., 2.], [0., .08, 2.]])
        second = first + [1.2, 0., 0.]
        points = np.vstack((first, second))
        world._merge(points, np.full((6, 3), 100.), np.ones(6, bool), 100.)
        snapshot = world.snapshot(100.)
        self.assertEqual(len(snapshot['volumes']), 2)

    def test_yolo_semantics_stay_on_volume_after_later_geometry_merge(self):
        frame, _ = supported_wall()
        complete_surfaces(frame, frame.rgb)
        world = SemanticMap()
        world.integrate(frame, np.eye(4), 100.)
        result = world.annotate(frame, np.eye(4),
                                [{'box': [80, 45, 200, 135], 'label': 'chair', 'confidence': .9}], 100.)
        self.assertTrue(result[0]['volumeIds'])
        world.integrate(frame, np.eye(4), 101.)
        snapshot = world.snapshot(101.)
        semantic = [item for item in snapshot['volumes'] if item['id'] in result[0]['volumeIds']]
        self.assertTrue(semantic)
        self.assertTrue(all(item['label'] == 'chair' for item in semantic))
        self.assertTrue(any(item['semanticConfidence'] > 0 for item in semantic))

    def test_semantic_surface_adds_points_and_labels_to_shared_cloud_immediately(self):
        frame, _ = supported_wall()
        complete_surfaces(frame, frame.rgb)
        session, timestamp = StereoSession(), time.time()
        session.world.integrate(frame, np.eye(4), timestamp)
        session.frames[7] = frame, np.eye(4), timestamp
        detection = {'box': [40, 20, 280, 160], 'label': 'wall', 'confidence': .9}
        result = session.annotate({'frameId': 7}, [detection])
        self.assertTrue(result['fused'])
        self.assertGreater(result['map']['inferredPoints'], 100)
        self.assertGreater(sum(bool(label) for label in result['map']['semantics']), 100)
        self.assertGreater(result['associations'][0]['inferredPoints'], 100)
        x, y = result['associations'][0]['surfacePixel']
        self.assertTrue(frame.surface_valid[y, x])
        revision = result['mapRevision']
        repeated = session.annotate({'frameId': 7}, [detection])
        self.assertEqual(repeated['mapRevision'], revision)
        self.assertEqual(repeated['map']['objects'][0]['observations'], 1)

    def test_observation_density_and_semantic_refresh_do_not_count_as_extra_frames(self):
        frame, _ = supported_wall()
        world = SemanticMap()
        world.integrate(frame, np.eye(4), 100.)
        world.integrate(frame, np.eye(4), 100.)
        self.assertTrue(all(voxel['count'] == 1 and voxel['frames'] == 1 for voxel in world.voxels.values()))
        world.annotate(frame, np.eye(4), [{'box': [35, 15, 285, 165], 'label': 'wall', 'confidence': .9}], 100.)
        self.assertTrue(all(voxel['count'] == 1 and voxel['frames'] == 1 for voxel in world.voxels.values()))
        world.integrate(frame, np.eye(4), 101.)
        self.assertTrue(all(voxel['frames'] <= 2 for voxel in world.voxels.values()))

    def test_measured_geometry_replaces_inference_and_cannot_be_refreshed_by_inference(self):
        world = SemanticMap()
        colors = np.array([[120., 120., 120.]])
        points = np.array([[.01, .01, 2.01]])
        world._merge(points, colors, np.array([False]), 100.)
        world._merge(points+.01, colors, np.array([True]), 101.)
        voxel = next(iter(world.voxels.values()))
        self.assertTrue(voxel['measured'])
        np.testing.assert_allclose(voxel['point'], points[0]+.01)
        world._merge(points, colors, np.array([False]), 102.)
        self.assertEqual(voxel['lastSeen'], 101.)
        self.assertEqual(voxel['frames'], 1)

    def test_invalid_semantic_confidence_has_no_label_or_object_side_effects(self):
        frame, _ = supported_wall()
        world = SemanticMap()
        world.integrate(frame, np.eye(4), 100.)
        before = len(world.voxels)
        for confidence in (float('nan'), 'bad', -1, 2):
            self.assertEqual(world.annotate(frame, np.eye(4), [
                {'box': [30, 10, 290, 170], 'label': 'wall', 'confidence': confidence}], 100.), [])
        self.assertEqual(world.labels, ['unlabeled'])
        self.assertFalse(world.objects)
        self.assertEqual(len(world.voxels), before)

    def test_late_semantics_cannot_rewind_geometry_or_newer_labels(self):
        world = SemanticMap()
        world.labels.extend(['wall', 'chair'])
        points, colors = np.array([[.01, .01, 2.01]]), np.full((1, 3), 120.)
        world._merge(points, colors, np.array([True]), 102., 2, .9)
        world._merge(points+.01, colors, np.array([True]), 101., 1, 1.)
        voxel = next(iter(world.voxels.values()))
        self.assertEqual(voxel['label'], 2)
        self.assertEqual(voxel['lastSeen'], 102.)
        np.testing.assert_allclose(voxel['point'], points[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
