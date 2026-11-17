"""Behavioral contracts for the hierarchical, vision-only navigation engine."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from social_engine.world_state import WorldState
from social_engine.social_graph import social_graph
from social_engine.arbitration import BehaviorArbiter
from behavior import BehaviorPolicy
from social_engine.engine import phase_for
from test_social_behavior import observation


def frame(stamp, people):
    result = observation(stamp)
    result['semanticPose'] = np.eye(4).ravel().tolist()
    result['people'] = [{'id': key, 'confidence': confidence,
                         'opticalPosition': [x, 0, z]} for key, x, z, confidence in people]
    return result


class WorldStateTests(unittest.TestCase):
    def test_unique_semantic_timestamps_not_depth_rate_count_evidence(self):
        world = WorldState()
        world.update(frame(100, [('a', 0, 3, .9)]), 100)
        repeat = frame(100.8, [('a', 0, 1, .9)])
        repeat['semanticObservedAt'] = 100
        world.update(repeat, 100.8)
        self.assertEqual(world.tracks['a']['samples'], 1)
        self.assertEqual(world.tracks['a']['distance'], 3)

    def test_old_and_future_semantics_cannot_create_entities(self):
        for stamp in (90, 101):
            world = WorldState()
            world.update(frame(stamp, [('a', 0, 3, .9)]), 100)
            self.assertEqual(world.tracks, {})

    def test_head_movement_reprojects_same_semantic_frame_without_new_evidence(self):
        world = WorldState()
        first = frame(100, [('a', 0, 3, .9)])
        first.update(viewHeading=0, semanticViewHeading=0)
        first['stereo']['pose']['bodyPosition'] = [0, 0, 0]
        world.update(first, 100)
        moved = frame(100.5, [('a', 0, 3, .9)])
        moved.update(semanticObservedAt=100, viewHeading=30, semanticViewHeading=0)
        moved['stereo']['pose'].update(bodyPosition=[0, 0, 1],
            cameraToWorld=np.array([[.8660254, 0, .5, 0], [0, 1, 0, 0],
                                   [-.5, 0, .8660254, 1], [0, 0, 0, 1]]).ravel().tolist())
        world.update(moved, 100.5)
        person = world.visible(100.5)[0]
        self.assertAlmostEqual(person['distance'], 2)
        self.assertAlmostEqual(person['bearing'], 30, places=4)
        self.assertEqual(person['samples'], 1)

    def test_one_to_one_association_and_bounded_velocity(self):
        world = WorldState()
        world.update(frame(100, [('a', -1, 3, .9), ('b', 1, 3, .9)]), 100)
        world.update(frame(100.5, [('new-a', -.8, 3, .9), ('new-b', 1.2, 3, .9)]), 100.5)
        self.assertEqual(set(world.tracks), {'a', 'b'})
        self.assertEqual(world.tracks['a']['observationId'], 'new-a')
        self.assertGreater(world.tracks['a']['velocity'][0], 0)
        self.assertLessEqual(np.linalg.norm(world.tracks['a']['velocity']), 1.6)

    def test_ambiguous_new_identity_does_not_merge_two_people(self):
        world = WorldState()
        world.update(frame(100, [('a', -.3, 3, .9), ('b', .3, 3, .9)]), 100)
        world.update(frame(100.5, [('new', 0, 3, .9)]), 100.5)
        self.assertEqual(world.tracks['a']['samples'], 1)
        self.assertEqual(world.tracks['b']['samples'], 1)

    def test_reused_detector_id_does_not_teleport(self):
        world = WorldState()
        world.update(frame(100, [('a', 0, 3, .9)]), 100)
        world.update(frame(100.2, [('a', 20, 3, .9)]), 100.2)
        self.assertEqual(world.tracks['a']['worldPosition'], [0, 0, 3])

    def test_occlusion_memory_expires_and_history_is_bounded(self):
        world = WorldState()
        for i in range(100):
            stamp = 100+i*.1
            world.update(frame(stamp, [('a', 0, 3, .9)]), stamp)
        self.assertEqual(len(world.tracks['a']['history']), 24)
        world.update(frame(111, []), 111)
        self.assertFalse(world.snapshot(111)['entities'][0]['visible'])
        world.update(frame(123, []), 123)
        self.assertEqual(world.tracks, {})

    def test_entity_count_is_bounded_and_unready_semantics_adds_no_track(self):
        world = WorldState()
        world.update(frame(100, [(f'p{i}', i*.2, 3, .9) for i in range(80)]), 100)
        self.assertEqual(len(world.tracks), 64)
        unready = frame(101, [('fake', 0, 3, 1.)])
        unready['semanticReady'] = False
        world.update(unready, 101)
        self.assertNotIn('fake', world.tracks)

    def test_simulation_truth_never_populates_visual_world(self):
        world = WorldState()
        image = frame(100, [])
        image.update(position=[99, 0, 99], players=[{'id': 'fake', 'position': [1, 0, 1]}])
        image['stereo']['mapping'] = False
        world.update(image, 100)
        self.assertEqual(world.tracks, {})
        self.assertIsNone(world.self_state['position'])


class ArbitrationTests(unittest.TestCase):
    def test_nearest_person_jitter_does_not_switch_persistent_target(self):
        world, arbiter = WorldState(), BehaviorArbiter()
        for i in range(80):
            stamp = 100+i*.1
            world.update(frame(stamp, [('a', -1, 2.9+.1*(i%2), .9), ('b', 1, 3-.1*(i%2), .9)]), stamp)
            chosen = arbiter.select(world, {'groups': []}, stamp)
            self.assertEqual(chosen['id'], 'a')
        self.assertEqual(arbiter.switches, 0)

    def test_better_candidate_requires_minimum_hold_and_margin(self):
        world, arbiter = WorldState(), BehaviorArbiter()
        world.update(frame(100, [('a', -1, 4, .9)]), 100)
        arbiter.select(world, {'groups': []}, 100)
        world.update(frame(101, [('a', -1, 4, .5), ('b', 1, 1, 1.)]), 101)
        self.assertEqual(arbiter.select(world, {'groups': []}, 101)['id'], 'a')
        world.update(frame(104, [('a', -1, 4, .5), ('b', 1, 1, 1.)]), 104)
        self.assertEqual(arbiter.select(world, {'groups': []}, 104)['id'], 'b')

    def test_lost_target_is_retained_then_expires(self):
        world, arbiter = WorldState(), BehaviorArbiter()
        world.update(frame(100, [('a', -1, 3, .9)]), 100)
        arbiter.select(world, {'groups': []}, 100)
        world.update(frame(102, [('b', 4, 3, .9)]), 102)
        self.assertIsNone(arbiter.select(world, {'groups': []}, 102))
        self.assertEqual(arbiter.target_id, 'a')
        world.update(frame(109, [('b', 4, 3, .9)]), 109)
        self.assertEqual(arbiter.select(world, {'groups': []}, 109)['id'], 'b')

    def test_group_entry_is_outside_shared_space_without_claiming_intent(self):
        world = WorldState()
        world.update(frame(100, [('a', -.5, 4, .9), ('b', .5, 4, .9)]), 100)
        graph = social_graph(world.visible(100), [0, 0, 0])
        group = graph['groups'][0]
        self.assertEqual(group['formation'], 'unverified')
        self.assertLess(group['entry'][2], group['center'][2]-1.)
        self.assertIsNone(social_graph(world.visible(100), None)['groups'][0]['entry'])


class EngineIntegrationTests(unittest.TestCase):
    def test_deadlock_recovery_is_reported_as_recover(self):
        for intent in ('escape_deadlock', 'leave_deadlock', 'inspect_blockage'):
            self.assertEqual(phase_for(intent), 'Recover')

    def test_arbitrated_group_entry_reaches_real_route_controller(self):
        policy = BehaviorPolicy()
        state = policy._state('engine')
        memory = state.memory
        for x in range(-4, 5):
            for z in range(-2, 15):
                memory.cells[(x, z)] = (-2., 100., 3)
        with patch('time.time', return_value=100):
            result = policy.decide({'clientId': 'engine', 'vision': frame(100, [('a', -.5, 4, .9), ('b', .5, 4, .9)])})
        self.assertEqual(result['intent'], 'join_group')
        self.assertEqual(result['action'], 'forward')
        self.assertTrue(result['continuous'])
        self.assertIsNotNone(result['engine']['groupRoute'])
        self.assertLess(result['distance'], 3.2)
        with patch('time.time', return_value=100.1):
            policy.feedback_for('engine', result['decisionId'], 'failed', '播放碰撞：墙')
        self.assertIsNotNone(state.navigator.drive.back_heading)
        with patch('time.time', return_value=101):
            recovered = policy.decide({'clientId': 'engine', 'vision': frame(101, [('a', -.5, 4, .9), ('b', .5, 4, .9)])})
        self.assertNotEqual(recovered['intent'], 'join_group')
        self.assertNotEqual(recovered['action'], 'forward')

    def test_feedback_and_reset_keep_world_and_execution_consistent(self):
        policy = BehaviorPolicy()
        with patch('time.time', return_value=100):
            result = policy.decide({'clientId': 'engine', 'vision': frame(100, [('a', 0, 4, .9)])})
            policy.feedback_for('engine', result['decisionId'], 'failed', '播放碰撞：门框')
            snapshot = policy.snapshot('engine')['engine']
        self.assertEqual(snapshot['phase'], 'Recover')
        self.assertEqual(snapshot['execution']['status'], 'failed')
        self.assertEqual(snapshot['reason'], '播放碰撞：门框')
        self.assertEqual(snapshot['arbitration']['targetId'], 'a')
        reset = frame(101, [])
        reset['stereo']['pose']['status'] = 'initialized'
        with patch('time.time', return_value=101):
            policy.observe('engine', reset)
            snapshot = policy.snapshot('engine')['engine']
        self.assertEqual(snapshot['world']['entities'], [])
        self.assertIsNone(snapshot['arbitration']['targetId'])
        json.dumps(snapshot, allow_nan=False)

    def test_interaction_phase_requires_real_gesture_feedback(self):
        policy = BehaviorPolicy()
        for stamp in (100, 100.4, 100.9):
            with patch('time.time', return_value=stamp):
                policy.decide({'clientId': 'engine', 'vision': observation(stamp, distance=1.3)})
        engine = policy._state('engine').engine
        self.assertEqual(engine.phase, 'Join')
        policy.gesture_feedback('engine', 'wave', 'playing', '')
        self.assertEqual(engine.phase, 'Interact')
        policy.gesture_feedback('engine', 'wave', 'completed', '上身挥手完成')
        self.assertEqual(engine.phase, 'Maintain')


if __name__ == '__main__':
    unittest.main()
