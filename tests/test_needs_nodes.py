import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from needs import Needs
from interaction_nodes import support_nodes, grasp_nodes
from affordances import ObjectEvidence, select_node, valid_nodes
import embodied
import goals
import runtime
import node_behavior


class NeedsTests(unittest.TestCase):
    def test_rates_are_time_based_and_disconnect_does_not_drain_hours(self):
        moving = {'playing': True, 'locomotion': {'phase': 'walk'}}
        fast, slow = Needs(), Needs()
        for step in range(601): fast.update(step/10, moving, {})
        for step in range(61): slow.update(step, moving, {})
        self.assertAlmostEqual(fast.energy, slow.energy)
        before = fast.energy; fast.update(100000, moving, {})
        self.assertGreater(fast.energy, before-.003)

    def test_rest_requires_actual_support_and_hysteresis(self):
        state = Needs(energy=.29); state.update(0, {}, {})
        self.assertTrue(state.resting)
        for i in range(1, 61): state.update(i, {'interaction': {'posture': 'lie', 'supported': False}}, {})
        self.assertLess(state.energy, .31)
        for i in range(61, 201): state.update(i, {'interaction': {'posture': 'lie', 'supported': True}}, {})
        self.assertFalse(state.resting); self.assertGreater(state.energy, .68)

    def test_boredom_and_fatigue_change_budget_not_motion_speed(self):
        energetic, tired, bored = Needs(energy=.9), Needs(energy=.1), Needs(energy=.9, boredom=.9)
        self.assertGreater(energetic.snapshot()['initiative'], tired.snapshot()['initiative'])
        self.assertGreater(bored.snapshot()['initiative'], energetic.snapshot()['initiative'])
        self.assertNotIn('speed', bored.snapshot())

    def test_repeated_detection_is_not_repeated_novelty(self):
        state = Needs(); objects = {'x': SimpleNamespace(identity='x', observed=0., confidence=.9)}
        state.update(0, {}, objects); first = state.boredom
        state.update(.1, {}, objects)
        self.assertGreaterEqual(state.boredom, first)
        state.reward('social'); self.assertLess(state.boredom, first)


class NodeTests(unittest.TestCase):
    def plane(self, y=.46):
        x,z = np.meshgrid(np.linspace(-1,1,60), np.linspace(-.3,.3,30))
        points = np.column_stack((x.ravel(), np.full(x.size,y),z.ravel()))
        return points,np.column_stack((np.arange(len(points)),np.zeros(len(points))))

    def test_plane_has_metric_size_and_control_and_vo_frames(self):
        points,pixels=self.plane(); transform=np.eye(4);transform[0,3]=7
        nodes=support_nodes(points,pixels,transform)
        self.assertTrue(nodes)
        self.assertAlmostEqual(nodes[0]['point'][0]-nodes[0]['controlPoint'][0],7)
        self.assertGreater(nodes[0]['size'][0],1.7)
        self.assertEqual(len(valid_nodes(nodes)),len(nodes))

    def test_vertical_wall_and_thin_edge_are_not_support(self):
        points,pixels=self.plane(); points[:,[1,2]]=points[:,[2,1]]
        self.assertEqual(support_nodes(points,pixels,np.eye(4)),[])
        points,pixels=self.plane(); points[:,2]*=.02
        self.assertEqual(support_nodes(points,pixels,np.eye(4)),[])

    def test_grasp_rejects_large_furniture(self):
        points,pixels=self.plane()
        self.assertEqual(grasp_nodes(points,pixels,np.eye(4)),[])
        points*=.08
        self.assertEqual(len(grasp_nodes(points,pixels,np.eye(4))),1)

    def test_lie_needs_actor_length_and_no_inferred_support(self):
        points,pixels=self.plane(); nodes=support_nodes(points,pixels,np.eye(4))
        row=ObjectEvidence('x','sofa',[0,.46,0],[1,1],[0,0,2,2],1,time.time(),.9,nodes=nodes)
        self.assertIsNotNone(select_node(row,'lie',{'stature':1.6}))
        self.assertIsNone(select_node(row,'lie',{'stature':2.5}))
        self.assertIsNone(select_node(row,'sit',None))
        nodes[0]['source']='plane-completion'
        self.assertEqual(valid_nodes(nodes),[])

    def test_completed_animation_is_not_verified_grasp(self):
        with patch.object(runtime,'active_session',return_value={'state':{'interaction':{'lastCommand':'c'}}}):
            with self.assertRaises(ValueError): node_behavior.verify({'clientId':'a'}, {'operation':'pickup','objectId':'x'}, {'id':'c'})

    def test_low_energy_prefers_supported_rest_over_pickup(self):
        state=embodied.Kernel();state.needs.energy=.1;state.needs.resting=True;state.needs.boredom=.9
        row={'objectId':'x','label':'sofa','fresh':True,'distance':2.,'options':[{'operation':op,'enabled':True} for op in ('lie','sit','pickup')]}
        step,_=embodied.choose_node({'objects':[row]},state,time.time())
        self.assertEqual(step['operation'],'lie')

    def test_expired_object_cooldown_cannot_starve_rest(self):
        state=embodied.Kernel();state.needs.energy=.2;state.needs.resting=True;state.recent['x']=1
        row={'objectId':'x','label':'sofa','fresh':True,'distance':1.,'options':[{'operation':'sit','enabled':True}]}
        with patch('affordances.catalog',return_value={'objects':[row]}):
            step,_=embodied.choose('cooldown-test',state,10)
        self.assertEqual(step['operation'],'sit')

    def test_body_commands_are_validated(self):
        self.assertEqual(goals.validate_step({'kind':'body','operation':'stand'})['operation'],'stand')
        with self.assertRaises(ValueError):goals.validate_step({'kind':'body','operation':'teleport'})
        with self.assertRaises(ValueError):goals.validate_step({'kind':'interact','objectId':'x','operation':'release'})


if __name__ == '__main__': unittest.main()
