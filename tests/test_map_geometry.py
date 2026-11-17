"""Supported meshing, instance separation and persistent bounded map regression."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from semantic_map import SemanticMap


def wall(width=24, height=24, offset=0):
    x, y = np.meshgrid(np.arange(offset, width+offset), np.arange(height))
    return np.column_stack(((x.ravel()+.5)*.08, (y.ravel()+.5)*.08, np.full(x.size, 2.04)))


def merge(world, points, timestamp=100., label=0, instance='', color=(120., 150., 180.)):
    world._merge(points, np.tile(color, (len(points), 1)), np.ones(len(points), bool),
                 timestamp, label, .9 if label else 0., instance)
    world._expire(timestamp, np.zeros(3))
    world._ensure_geometry(force=True)
    return world.snapshot(timestamp)


class GeometryTests(unittest.TestCase):
    def test_doorway_is_not_closed_by_surface_convex_hull(self):
        world = SemanticMap()
        points = wall(30, 30)
        hole = (points[:, 0] >= .8) & (points[:, 0] < 1.6) & (points[:, 1] < 1.6)
        snapshot = merge(world, points[~hole])
        area = 0.
        for surface in snapshot['surfaces']:
            vertices = np.asarray(surface['vertices']).reshape(-1, 3)
            for quad in vertices.reshape(-1, 4, 3):
                low, high = quad.min(axis=0), quad.max(axis=0)
                self.assertTrue(high[0] <= .8+.0001 or low[0] >= 1.6-.0001 or low[1] >= 1.6-.0001)
                area += np.linalg.norm(np.cross(quad[1]-quad[0], quad[3]-quad[0]))
        self.assertAlmostEqual(area, len(points[~hole])*.08**2, places=4)

    def test_touching_same_label_instances_remain_separate(self):
        world = SemanticMap()
        world.labels.append('chair')
        points = wall(24, 20)
        left = points[:, 0] < .96
        merge(world, points[left], label=1, instance='chair-a')
        result = merge(world, points[~left], label=1, instance='chair-b')
        self.assertEqual({v['instanceId'] for v in result['volumes']}, {'chair-a', 'chair-b'})
        self.assertEqual(len(result['volumes']), 2)
        self.assertEqual(len(result['surfaces']), 2)

    def test_floor_contact_cannot_inherit_wall_semantics(self):
        world = SemanticMap()
        world.labels.append('wall')
        points = wall(24, 24)
        x, z = np.meshgrid((np.arange(24)+.5)*.08, (np.arange(26)+.5)*.08)
        floor = np.column_stack((x.ravel(), np.full(x.size, .04), z.ravel()))
        merge(world, points, label=1, instance='wall-a')
        result = merge(world, floor)
        floor_voxels = [v for v in world.voxels.values() if v['point'][1] < .08 and v['point'][2] < 1.6]
        self.assertTrue(floor_voxels)
        self.assertTrue(all(v['label'] == 0 for v in floor_voxels))
        self.assertFalse(any(v['bounds'][1][1]-v['bounds'][0][1] > .5
                             and v['bounds'][1][2]-v['bounds'][0][2] > .5 for v in result['volumes']))

    def test_new_connected_points_inherit_semantics_and_identity(self):
        world = SemanticMap()
        world.labels.append('wall')
        first = merge(world, wall(16, 20), label=1, instance='wall-a')
        identity = first['volumes'][0]['id']
        result = merge(world, wall(4, 20, offset=16), 101.)
        extended = [v for v in world.voxels.values() if v['point'][0] > 1.28]
        self.assertTrue(all(v['label'] == 1 and v['instanceId'] == 'wall-a' for v in extended))
        self.assertEqual(result['volumes'][0]['id'], identity)
        self.assertEqual(result['volumes'][0]['label'], 'wall')

    def test_surface_color_changes_gradually_across_exposure_change(self):
        world = SemanticMap()
        first = merge(world, wall(), color=(100., 120., 140.))
        second = merge(world, wall(), 101., color=(180., 200., 220.))
        self.assertEqual(first['surfaces'][0]['id'], second['surfaces'][0]['id'])
        change = np.asarray(second['surfaces'][0]['meanColor'])-first['surfaces'][0]['meanColor']
        self.assertTrue(np.all(change > 0))
        self.assertTrue(np.all(change <= 4))
        self.assertEqual(len(set(second['colors'])), 1)

    def test_split_components_cannot_reuse_the_same_identity(self):
        world = SemanticMap()
        merge(world, wall(30, 20))
        for voxel in world.voxels.values():
            voxel['lastSeen'] = -2000. if .8 < voxel['point'][0] < 1.6 else 101.
        result = world.snapshot(102.)
        self.assertEqual(len(result['volumes']), 2)
        self.assertEqual(len({v['id'] for v in result['volumes']}), 2)
        self.assertEqual(len(result['surfaces']), 2)

    def test_expiration_invalidates_geometry_revision_and_cached_payload(self):
        world = SemanticMap()
        first = merge(world, wall())
        again = world.snapshot(100.1)
        self.assertIs(first['positions'], again['positions'])
        empty = world.snapshot(1901.)
        self.assertEqual(empty['totalPoints'], 0)
        self.assertEqual(empty['surfaces'], [])
        self.assertEqual(empty['volumes'], [])
        self.assertGreater(empty['revision'], first['revision'])
        self.assertGreater(empty['geometryRevision'], first['geometryRevision'])
        self.assertFalse(world.geometry.blocks)

    def test_slanted_surface_has_finite_coplanar_vertices(self):
        world = SemanticMap()
        points = wall()
        points[:, 2] = points[:, 0] + points[:, 1] + 2.
        result = merge(world, points)
        self.assertTrue(result['surfaces'])
        for surface in result['surfaces']:
            vertices = np.asarray(surface['vertices']).reshape(-1, 3)
            self.assertTrue(np.isfinite(vertices).all())
            np.testing.assert_allclose(vertices[:, 2]-vertices[:, 0]-vertices[:, 1], 2., atol=.03)

    def test_stationary_observation_reuses_local_normals(self):
        world = SemanticMap()
        points = wall()
        merge(world, points)
        with patch.object(world.geometry, '_local_normals', wraps=world.geometry._local_normals) as local:
            merge(world, points, 101.)
        self.assertEqual(local.call_count, 0)

    def test_sparse_points_gain_normals_when_neighboring_observations_arrive(self):
        world = SemanticMap()
        points = wall(20, 20)
        first = points[::25]
        merge(world, first)
        result = merge(world, points, 101.)
        old = [world.voxels[tuple(np.floor(point/.08).astype(int))] for point in first]
        self.assertTrue(all(v['normalValid'] for v in old))
        self.assertTrue(result['surfaces'])

    def test_many_frames_keep_voxels_bounded_and_semantics_stable(self):
        world = SemanticMap(limit=500)
        world.labels.append('wall')
        points = wall(20, 20)
        first = merge(world, points, label=1, instance='wall-a')
        identity = first['volumes'][0]['id']
        rng = np.random.default_rng(8)
        for frame in range(80):
            jitter = rng.normal(0, .002, points.shape)
            world._merge(points+jitter, np.full(points.shape, 140.), np.ones(len(points), bool), 101.+frame)
            world._expire(101.+frame, np.zeros(3))
        world._ensure_geometry(force=True)
        result = world.snapshot(180.)
        self.assertLessEqual(result['totalPoints'], 500)
        self.assertEqual(result['volumes'][0]['id'], identity)
        self.assertTrue(all(v['label'] == 'wall' for v in result['volumes']))
        self.assertEqual(sum(map(len, world.geometry.blocks.values())), result['totalPoints'])

    def test_weak_single_label_and_duplicate_evidence_cannot_replace_votes(self):
        world = SemanticMap()
        world.labels.extend(['wall', 'chair'])
        points = wall(8, 8)
        for frame in range(4):
            world._merge(points, np.full(points.shape, 140.), np.ones(len(points), bool),
                         100.+frame, 1, .9, 'wall-a')
        for _ in range(3):
            world._merge(points, np.full(points.shape, 140.), np.ones(len(points), bool),
                         104., 2, .3, 'chair-b')
        self.assertTrue(all(v['label'] == 1 for v in world.voxels.values()))
        self.assertTrue(all(v['labelVotes'][2] <= .3 for v in world.voxels.values()))


if __name__ == '__main__':
    unittest.main()
