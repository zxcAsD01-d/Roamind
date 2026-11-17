"""Observed enclosure, stable exits and retractable floor connectivity."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from exit_memory import ExitMemory
from exploration_memory import ExplorationMemory
from floor_topology import FloorTopology, neighbors
from exploration_drive import ExplorationDrive


def enclosed_floor(free, stamp=100):
    walls = {other for key in free for other in neighbors(key) if other not in free}
    return {**{key: (-2., stamp, 4) for key in free}, **{key: (3., stamp, 4) for key in walls}}


def rooms(width=3):
    free = {(x, z) for x in [*range(10), *range(13, 23)] for z in range(10)}
    free.update((x, z) for x in range(10, 13) for z in range(3, 3+width))
    return enclosed_floor(free)


def portal(point=(4.2, 0., 1.575), normal=(1., 0.), stamp=100):
    memory = ExitMemory()
    memory.integrate({'frame': 'visual-odometry', 'observedAt': stamp, 'candidates': [
        {'position': list(point), 'normal': list(normal), 'width': 1.05,
         'confidence': .9, 'uncertaintyMeters': .04}]}, stamp)
    return memory


class SpatialMemoryTests(unittest.TestCase):
    def test_complete_measured_loop_has_scoped_enclosure(self):
        topology = FloorTopology()
        topology.update(enclosed_floor({(x, z) for x in range(10) for z in range(10)}), 1, 100)
        state = topology.snapshot([1, 0, 1])
        region = state['regions'][0]
        self.assertTrue(region['enclosure']['closedBoundary'])
        self.assertEqual(region['enclosure']['scope'], 'observed-floor-boundary')
        self.assertEqual(len(state['boundarySegments']), 4)
        self.assertAlmostEqual(region['perimeterMeters'], 14.)
        self.assertFalse(state['sealedRoomClaim'])
        self.assertFalse(region['boundaryBlocksTravel'])

    def test_one_unknown_edge_prevents_enclosure_even_above_95_percent_support(self):
        cells = enclosed_floor({(x, z) for x in range(20) for z in range(20)})
        del cells[(-1, 10)]
        topology = FloorTopology()
        topology.update(cells, 1, 100)
        region = next(iter(topology.regions.values()))
        self.assertGreater(region['enclosureConfidence'], .95)
        self.assertFalse(region['enclosure']['closedBoundary'])
        self.assertEqual(region['boundary'], 'open-unobserved')
        self.assertTrue(any(edge['state'] == 'unknown' for edge in topology.snapshot(None)['boundarySegments']))

    def test_opening_between_regions_is_never_counted_as_enclosing_wall(self):
        topology = FloorTopology()
        topology.update(rooms(), 1, 100)
        self.assertEqual(len(topology.regions), 2)
        self.assertEqual(topology.snapshot([1, 0, 1])['components'], 1)
        space = topology.snapshot([1, 0, 1])['spaces'][0]
        self.assertTrue(space['enclosure']['closedBoundary'])
        self.assertEqual(space['regionIds'], [1, 2])
        self.assertNotIn('opening', space['boundaryEvidence'])
        for region in topology.regions.values():
            self.assertFalse(region['enclosure']['closedBoundary'])
            self.assertEqual(region['enclosure']['status'], 'connected')
            self.assertGreater(region['boundaryEvidence']['opening'], 0)
            self.assertLess(region['enclosureConfidence'], 1.)

    def test_three_sources_share_one_stable_exit(self):
        topology, exits = FloorTopology(), portal()
        cells = rooms()
        topology.remember_doors([{'id': 'door-1', 'label': 'door', 'confidence': .9,
                                  'worldPosition': [4.2, 1., 1.575]}], 100)
        topology.update(cells, 1, 100, exits.portals)
        state = topology.snapshot([1, 0, 1])
        self.assertEqual(len(state['exits']), 1)
        item = state['exits'][0]
        self.assertEqual(len(item['sources']), 3)
        self.assertTrue(item['traversable'])
        self.assertAlmostEqual(item['width'], 1.05)
        self.assertEqual(state['doors'][0]['exitId'], state['passages'][0]['exitId'])
        topology.update(cells, 2, 102, exits.portals)
        self.assertEqual(topology.snapshot(None)['exits'][0]['id'], item['id'])

    def test_close_and_reopen_retracts_and_restores_connection_without_new_identity(self):
        topology, cells = FloorTopology(), rooms()
        topology.update(cells, 1, 100)
        identity = next(iter(topology.connections.items))
        original_regions = set(topology.regions)
        cells.update({(12, z): (3., 102, 4) for z in range(3, 6)})
        topology.update(cells, 2, 102)
        item = topology.connections.items[identity]
        self.assertEqual(item['state'], 'blocked')
        self.assertEqual(topology.snapshot([1, 0, 1])['components'], 2)
        self.assertTrue(all(not region['connectedRegions'] for region in topology.regions.values()))
        cells.update({(12, z): (-2., 104, 4) for z in range(3, 6)})
        topology.update(cells, 3, 104)
        self.assertEqual(set(topology.regions), original_regions)
        self.assertEqual(len(topology.connections.items), 1)
        self.assertTrue(topology.connections.items[identity]['connectsRegions'])

    def test_narrow_free_link_is_observed_but_cannot_authorize_connectivity(self):
        topology = FloorTopology()
        topology.update(rooms(width=2), 1, 100)
        item = topology.snapshot(None)['exits'][0]
        self.assertEqual(item['state'], 'narrow')
        self.assertFalse(item['traversable'])
        self.assertEqual(topology.snapshot(None)['components'], 2)
        self.assertTrue(all(not region['enclosure']['closedBoundary'] for region in topology.regions.values()))

    def test_missing_floor_at_portal_does_not_create_graph_edge(self):
        topology, exits = FloorTopology(), portal()
        cells = rooms()
        del cells[(12, 4)]
        topology.update(cells, 1, 100, exits.portals)
        self.assertFalse(any(item['traversable'] for item in topology.snapshot(None)['exits']))
        self.assertEqual(topology.snapshot(None)['components'], 2)

    def test_semantic_door_on_free_floor_cannot_create_a_passage(self):
        topology = FloorTopology()
        cells = enclosed_floor({(x, z) for x in range(10) for z in range(10)})
        topology.remember_doors([{'id': 'door', 'label': 'door', 'confidence': .9,
                                  'worldPosition': [1.75, 1., 1.75]}], 100)
        topology.update(cells, 1, 100)
        self.assertFalse(topology.doors['door']['traversable'])
        self.assertEqual(topology.snapshot(None)['exits'][0]['state'], 'unconfirmed')

    def test_enclosure_evidence_ages_and_can_be_reconfirmed_without_revision_change(self):
        topology = FloorTopology()
        cells = enclosed_floor({(x, z) for x in range(10) for z in range(10)})
        topology.update(cells, 1, 100)
        topology.update(cells, 1, 221)
        self.assertFalse(next(iter(topology.regions.values()))['enclosure']['closedBoundary'])
        cells = {key: (value[0], 225, value[2]) for key, value in cells.items()}
        topology.update(cells, 1, 226)
        self.assertTrue(next(iter(topology.regions.values()))['enclosure']['closedBoundary'])

    def test_expired_floor_clears_assignments_graph_and_exits(self):
        topology = FloorTopology()
        topology.update(rooms(), 1, 100)
        topology.update(rooms(), 1, 1901)
        state = topology.snapshot([1, 0, 1])
        self.assertIsNone(state['currentRegion'])
        self.assertEqual(state['regions'], [])
        self.assertEqual(state['exits'], [])
        self.assertEqual(state['passages'], [])

    def test_split_does_not_duplicate_region_identity_and_merged_search_is_replayed(self):
        topology = FloorTopology()
        cells = enclosed_floor({(x, z) for x in range(20) for z in range(10)})
        topology.update(cells, 1, 100)
        identity = topology.region_at([1, 0, 1])
        topology.observations.append((np.array([1., 0., 1.]), 0., True, 100))
        for z in range(10):
            cells[(10, z)] = (3., 102, 4)
        topology.update(cells, 2, 102)
        self.assertEqual(len(topology.regions), 2)
        self.assertIn(identity, topology.regions)
        left = topology.region_at([1, 0, 1])
        right = topology.region_at([5, 0, 1])
        self.assertEqual(topology.regions[left]['lastPersonAt'], 100)
        self.assertIsNone(topology.regions[right]['lastPersonAt'])
        cells.update({(10, z): (-2., 104, 4) for z in range(10)})
        topology.update(cells, 3, 104)
        self.assertEqual(len(topology.regions), 1)
        self.assertEqual(next(iter(topology.regions.values()))['lastPersonAt'], 100)

    def test_geometric_crossing_is_deduplicated_with_cloud_crossing(self):
        topology, exits = FloorTopology(), portal()
        topology.update(rooms(), 1, 100, exits.portals)
        for index, x in enumerate((3.2, 3.7, 4.1, 4.3, 4.8)):
            point = np.array([x, 0., 1.575])
            exits.observe(point, 100+index*.4)
            topology.connections.observe(point, 100+index*.4)
        topology.update(rooms(), 1, 103, exits.portals)
        item = topology.snapshot(None)['exits'][0]
        self.assertEqual(item['crossings'], 1)
        self.assertEqual(item['state'], 'crossed')

    def test_crossing_cannot_be_manufactured_by_tracking_gap_or_delayed_position(self):
        exits = portal(point=(0., 0., 2.), normal=(0., 1.))
        exits.observe(np.array([0., 0., 1.5]), 100)
        exits.observe(np.array([0., 0., 2.6]), 105)
        exits.observe(np.array([0., 0., 1.5]), 101)
        self.assertEqual(exits.snapshot()['crossings'], [])
        self.assertEqual(exits.last_observed_at, 105)

    def test_stale_portal_and_door_evidence_cannot_overwrite_new_geometry(self):
        exits = portal(stamp=102)
        candidate = {'position': [4.4, 0., 1.575], 'normal': [1., 0.], 'width': .7,
                     'confidence': .2, 'uncertaintyMeters': .4}
        exits.remember(candidate, 101)
        self.assertAlmostEqual(exits.portals[1]['confidence'], .9)
        topology = FloorTopology()
        door = {'id': 'door', 'label': 'door', 'confidence': .9, 'worldPosition': [1., 1., 1.]}
        topology.remember_doors([door], 102)
        topology.remember_doors([{**door, 'worldPosition': [2., 1., 1.]}], 101)
        self.assertEqual(topology.doors['door']['position'], [1., 1., 1.])

    def test_snapshot_is_json_serializable_and_unknown_position_has_no_reachable_regions(self):
        memory = ExplorationMemory()
        memory.cells, memory.position = rooms(), np.array([1., 0., 1.])
        memory.topology.update(memory.cells, 1, 100)
        state = memory.snapshot()
        json.dumps(state, allow_nan=False)
        self.assertTrue(any(len(cell) == 4 and cell[3] is not None for cell in state['knownCells']))
        self.assertEqual(memory.topology.snapshot(None)['reachableRegions'], [])

    def test_separate_parallel_openings_keep_distinct_stable_ids(self):
        free = {(x, z) for x in [*range(12), *range(15, 27)] for z in range(20)}
        free.update((x, z) for x in range(12, 15) for z in [*range(3, 6), *range(13, 16)])
        cells, topology = enclosed_floor(free), FloorTopology()
        topology.update(cells, 1, 100)
        before = {(tuple(item['position']), item['id']) for item in topology.snapshot(None)['exits']}
        self.assertEqual(len(before), 2)
        topology.update(cells, 2, 102)
        self.assertEqual(before, {(tuple(item['position']), item['id']) for item in topology.snapshot(None)['exits']})

    def test_graph_can_find_unsearched_region_through_searched_intermediate_region(self):
        free = {(x, z) for x in [*range(10), *range(13, 23), *range(26, 36)] for z in range(10)}
        free.update((x, z) for x in [*range(10, 13), *range(23, 26)] for z in range(3, 6))
        topology = FloorTopology()
        topology.update(enclosed_floor(free), 1, 100)
        a, b, c = (topology.region_at(point) for point in ([1, 0, 1], [6, 0, 1], [10, 0, 1]))
        topology.regions[a]['searchedAt'] = topology.regions[b]['searchedAt'] = 100
        self.assertEqual(topology.unsearched_hop(a), b)
        self.assertEqual(topology.snapshot([1, 0, 1])['reachableRegions'], list(topology.regions))
        self.assertIn(c, topology.connections.adjacency[b])

    def test_blocked_cloud_exit_is_demoted_until_new_free_floor_arrives(self):
        memory = ExplorationMemory()
        memory.cells, memory.exit_memory = rooms(), portal()
        memory.position, memory.rotation = np.array([1., 0., 1.575]), np.eye(3)
        memory.cells.update({(12, z): (3., 102, 4) for z in range(3, 6)})
        memory.topology.update(memory.cells, 1, 102, memory.exit_memory.portals)
        self.assertEqual(ExplorationDrive().cloud_candidates(memory, 102), [])
        memory.cells.update({(12, z): (-2., 104, 4) for z in range(3, 6)})
        memory.topology.update(memory.cells, 2, 104, memory.exit_memory.portals)
        self.assertTrue(ExplorationDrive().cloud_candidates(memory, 104))

    def test_disappearance_without_any_exit_evidence_expires_old_portals(self):
        exits = portal()
        exits.integrate(None, 1901)
        self.assertEqual(exits.portals, {})

    def test_free_history_and_delayed_cells_cannot_fake_multiframe_walls(self):
        memory = ExplorationMemory()
        for stamp in range(100, 120):
            memory.mark((0, 0), -.35, stamp)
        memory.mark((0, 0), 6., 120)
        self.assertEqual(memory.cells[(0, 0)][2], 1)
        before = memory.cells[(0, 0)]
        revision = memory.map_revision
        memory.mark((0, 0), -3., 119)
        self.assertEqual(memory.cells[(0, 0)], before)
        self.assertEqual(memory.map_revision, revision)
        memory.mark((0, 0), .9, 121)
        memory.mark((0, 0), .9, 122)
        self.assertEqual(memory.cells[(0, 0)][2], 3)

    def test_memory_maintenance_expires_topology_even_without_new_camera_frames(self):
        memory = ExplorationMemory()
        memory.cells, memory.exit_memory = rooms(), portal()
        memory.topology.update(memory.cells, 1, 100, memory.exit_memory.portals)
        memory.maintain(1901)
        self.assertEqual(memory.cells, {})
        self.assertEqual(memory.topology.snapshot(None)['regions'], [])
        self.assertEqual(memory.exit_memory.portals, {})

    def test_delayed_stereo_pose_cannot_rewind_spatial_position(self):
        memory = ExplorationMemory()
        pose = np.eye(4)
        frame = {'mapping': True, 'observedAt': 101, 'pose': {'cameraToWorld': pose.ravel().tolist()}}
        memory.update(frame, [], 101)
        pose[0, 3] = 5.
        delayed = {**frame, 'observedAt': 100, 'pose': {'cameraToWorld': pose.ravel().tolist()}}
        self.assertFalse(memory.update(delayed, [], 101))
        self.assertEqual(memory.position.tolist(), [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
