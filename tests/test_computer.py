import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from affordances import ObjectEvidence, select_node, valid_nodes, InteractionOptions
from interaction_nodes import workstation_nodes
from needs import Needs
from world import get_world, route, segment_clear
import embodied
import goals
import node_behavior


def node(kind, position, size):
    return {'kind': kind, 'point': position, 'controlPoint': position, 'axis': [1, 0, 0], 'controlAxis': [1, 0, 0],
            'size': size, 'pixel': [10, 10], 'samples': 80, 'sigma': .005,
            'source': 'measured-stereo-surface', 'frame': 'control-world-y-up'}


class ComputerTests(unittest.TestCase):
    def evidence(self):
        now = time.time()
        keyboard = ObjectEvidence('keys', 'keyboard', [0, .7, 0], [10, 10], [0, 0, 50, 20], 1, now, .9,
                                  nodes=[node('workstation', [0, .7, 0], [.42, 0, .15])])
        chair = ObjectEvidence('seat', 'chair', [0, .45, .65], [20, 30], [0, 20, 50, 80], 1, now, .9,
                               nodes=[node('support', [0, .45, .65], [.5, 0, .5])])
        return keyboard, chair, {'keys': keyboard, 'seat': chair}

    def test_room_replaced_and_reachable_without_using_hidden_nodes(self):
        world = get_world()
        self.assertEqual(next(r for r in world['rooms'] if r['id'] == 'bedroom4')['name'], '电脑房')
        self.assertFalse(any(o.get('sourceId') == 'bed-3' for o in world['objects']))
        path = route(world['spawn'], [8, .92, -3.8])
        self.assertTrue(all(segment_clear((a[0], a[2]), (b[0], b[2])) for a, b in zip(path, path[1:])))
        self.assertEqual(len(world['computers']), 1)

    def test_measured_keyboard_surface_not_vertical_or_sparse(self):
        self.assertEqual(workstation_nodes(np.empty((0, 3)), np.empty((0, 2)), np.eye(4)), [])
        x, z = np.meshgrid(np.linspace(-.22, .22, 30), np.linspace(-.08, .08, 15))
        cloud = np.column_stack((x.ravel(), np.full(x.size, .7), z.ravel()))
        pixels = np.column_stack((np.arange(len(cloud)), np.zeros(len(cloud))))
        nodes = workstation_nodes(cloud, pixels, np.eye(4))
        self.assertEqual(len(valid_nodes(nodes)), 1)
        cloud[:, [1, 2]] = cloud[:, [2, 1]]
        self.assertEqual(workstation_nodes(cloud, pixels, np.eye(4)), [])

    def test_needs_observed_seat_and_rejects_stale_or_wrong_height(self):
        keyboard, chair, objects = self.evidence()
        target = select_node(keyboard, 'play_computer', {'stature': 1.3}, objects)
        self.assertEqual(target['seatObjectId'], 'seat')
        self.assertNotIn('seat', keyboard.nodes[0])
        self.assertIsNone(select_node(keyboard, 'play_computer', {'stature': 1.3}))
        chair.observed -= 10
        self.assertIsNone(select_node(keyboard, 'play_computer', {'stature': 1.3}, objects))
        chair.observed = time.time(); chair.nodes[0]['point'][1] = .85
        self.assertIsNone(select_node(keyboard, 'play_computer', {'stature': 1.3}, objects))

    def test_needs_real_input_to_reduce_boredom_and_does_not_recover_energy_while_playing(self):
        gaming = {'playing': True, 'interaction': {'posture': 'sit', 'supported': True, 'activity': 'computer',
                   'computer': {'active': True, 'inputs': 1}}}
        state = Needs(); state.update(0, gaming, {})
        for t in range(1, 31): state.update(t, gaming, {})
        self.assertLess(state.energy, .8); self.assertLess(state.boredom, .15)
        gaming['interaction']['computer']['inputs'] = 0
        before = state.boredom; state.update(31, gaming, {})
        self.assertGreater(state.boredom, before)

    def test_bored_selects_computer_but_fatigue_still_selects_rest(self):
        row = {'objectId': 'keys', 'label': 'keyboard', 'fresh': True, 'distance': 2,
               'options': [{'operation': 'play_computer', 'enabled': True}, {'operation': 'sit', 'enabled': True}]}
        state = embodied.Kernel(); state.needs.boredom = .8
        self.assertEqual(embodied.choose_node({'objects': [row]}, state, time.time())[0]['operation'], 'play_computer')
        state.needs.resting = True; state.recent.clear()
        self.assertEqual(embodied.choose_node({'objects': [row]}, state, time.time())[0]['operation'], 'sit')

    def test_game_duration_and_timeout_validation_and_high_level_interface(self):
        intent = {'kind': 'interact', 'objectId': 'keys', 'operation': 'play_computer', 'duration': 10}
        self.assertEqual(embodied.intent_plan(intent)['steps'][0]['duration'], 10)
        for duration in (0, float('nan'), 121):
            with self.assertRaises(ValueError): goals.validate_step({**intent, 'duration': duration})
        with self.assertRaises(ValueError): goals.validate_step({**intent, 'duration': 30, 'timeout': 31})

    def test_animation_alone_does_not_prove_computer_use(self):
        task, step, command = {'clientId': 'a'}, {'operation': 'play_computer', 'objectId': 'keys', 'duration': 5}, {'id': 'c'}
        state = {'lastCommand': 'c', 'objectId': 'keys', 'posture': 'sit', 'supported': True,
                 'computer': {'elapsed': 5, 'inputs': 0, 'active': False, 'completed': True}}
        with patch('runtime.active_session', return_value={'state': {'interaction': state}}):
            with self.assertRaises(ValueError): node_behavior.verify(task, step, command)
            state['computer']['inputs'] = 20
            node_behavior.verify(task, step, command)


if __name__ == '__main__': unittest.main()
