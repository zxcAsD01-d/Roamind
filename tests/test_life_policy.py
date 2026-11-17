import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
import affordances
import behavior
import embodied
import goals
import node_behavior
import runtime
from exploration_memory import ExplorationMemory
from life_policy import IntentState, demand
from needs import Needs
from test_social_behavior import observation


def receipt(command='consume', operation='eat'):
    return {'commandId': command, 'objectId': 'food', 'operation': operation, 'completed': True,
            'returned': True, 'mouthError': .002, 'graspError': .005, 'remaining': .5}


class MetabolismTests(unittest.TestCase):
    def test_rates_are_fps_independent_and_hysteretic(self):
        a, b = Needs(), Needs()
        moving = {'playing': True, 'locomotion': {'phase': 'walk'}}
        for i in range(3601):
            a.update(i/60, moving, {})
        for i in range(61):
            b.update(i, moving, {})
        self.assertAlmostEqual(a.satiety, b.satiety)
        self.assertAlmostEqual(a.thirst, b.thirst)
        a.satiety, a.thirst = .34, .61
        a.update(60, {}, {})
        self.assertTrue(a.seeking_food and a.seeking_water)
        a.satiety, a.thirst = .5, .4
        a.update(60, {}, {})
        self.assertTrue(a.seeking_food and a.seeking_water)
        a.satiety, a.thirst = .8, .1
        a.update(60, {}, {})
        self.assertFalse(a.seeking_food or a.seeking_water)

    def test_consumption_is_exactly_once_and_animation_reward_is_insufficient(self):
        state = Needs(satiety=.3, thirst=.8)
        state.reward('eat')
        self.assertEqual(state.satiety, .3)
        self.assertTrue(state.consume('consume', 'eat', receipt()))
        self.assertAlmostEqual(state.satiety, .72)
        self.assertFalse(state.consume('consume', 'eat', receipt()))
        self.assertAlmostEqual(state.satiety, .72)
        state.consume('sip', 'drink', receipt('sip', 'drink'))
        self.assertAlmostEqual(state.thirst, .25)
        with self.assertRaises(ValueError):
            state.consume('bad', 'eat', {**receipt('bad'), 'returned': False})

    def test_demands_and_state_history_remain_bounded_long_term(self):
        state, intent = Needs(), IntentState()
        for i in range(20000):
            state.update(i*.5, {}, {})
            intent.enter('explore' if i%2 else 'approach', demand(state), str(i%5), i*.5)
        self.assertTrue(state.seeking_food and state.seeking_water)
        self.assertLessEqual(len(intent.history), 24)
        self.assertTrue(0 <= state.energy <= 1 and 0 <= state.thirst <= 1 and 0 <= state.satiety <= 1)


class LifeExecutionTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear(); runtime.COMMANDS.clear(); runtime.INTERACTIONS.clear()
        goals.TASKS.clear(); embodied.KERNELS.clear(); behavior.POLICY.reset()
        runtime.sync({'clientId': 'life', 'state': {'playing': False, 'generating': False}})
        self.now = time.time()

    def test_idle_autonomous_submission_has_no_interrupt_but_manual_does(self):
        goals.submit({'clientId': 'life', 'steps': [{'kind': 'social'}], '_autonomous': True})
        self.assertFalse(runtime.COMMANDS)
        goals.submit({'clientId': 'life', 'steps': [{'kind': 'social'}]})
        self.assertEqual(next(iter(runtime.COMMANDS.values()))['kind'], 'interrupt')

    def test_urgent_need_keeps_motion_and_rearbitrates_only_at_boundary(self):
        state = embodied.kernel('life'); state.enabled = True
        task = goals.submit({'clientId': 'life', 'steps': [{'kind': 'social'}], '_autonomous': True})
        task = goals.TASKS['life']; task['drive'] = 'curiosity'; state.goal_id = task['id']
        state.needs.thirst, state.needs.seeking_water = .9, True
        self.assertFalse(embodied.reconsider('life', state, task, {'playing': True}, self.now))
        self.assertEqual(task['status'], 'running')
        self.assertTrue(embodied.reconsider('life', state, task, {'playing': False}, self.now))
        self.assertEqual(task['status'], 'superseded')
        task['status'] = 'running'; state.goal_id = 'user-task'
        self.assertFalse(embodied.reconsider('life', state, task, {}, self.now))

    def test_demand_targets_observed_memory_not_scene_truth(self):
        state = embodied.kernel('life'); state.needs.thirst = .8; state.needs.seeking_water = True
        row = {'objectId': 'remembered-cup', 'label': 'cup', 'fresh': False,
               'distance': 8., 'confidence': .9, 'remaining': 1}
        step, _ = embodied.choose_resource({'objects': [row]}, state)
        self.assertEqual(step['objectId'], 'remembered-cup')
        self.assertNotIn('position', step)
        row['remaining'] = 0
        self.assertIsNone(embodied.choose_resource({'objects': [row]}, state))

    def test_consumption_requires_exact_execution_and_contact_feedback(self):
        state = runtime.SESSIONS['life']['state']
        state['interaction'] = {'lastCommand': 'consume', 'objectId': 'food', 'consumption': receipt()}
        step, task = {'operation': 'eat', 'objectId': 'food'}, {'clientId': 'life'}
        needs = embodied.kernel('life').needs; needs.satiety = .2
        node_behavior.verify(task, step, {'id': 'consume'})
        self.assertAlmostEqual(needs.satiety, .62)
        node_behavior.verify(task, step, {'id': 'consume'})
        self.assertAlmostEqual(needs.satiety, .62)
        state['interaction']['lastCommand'] = 'bad'
        state['interaction']['consumption'] = receipt('bad') | {'mouthError': float('nan')}
        with self.assertRaises(ValueError):
            node_behavior.verify(task, step, {'id': 'bad'})
        self.assertAlmostEqual(needs.satiety, .62)

    def test_successful_goal_has_no_artificial_idle_delay(self):
        state = embodied.kernel('life')
        embodied.finish('life', state, {'id': 'done', 'status': 'completed', 'steps': [{'kind': 'body'}], 'detail': 'done'}, self.now)
        self.assertEqual(state.next_at, self.now)

    def test_resource_search_does_not_turn_into_person_pursuit(self):
        from social_engine.engine import SocialNavigationEngine
        from social_navigation import SocialNavigator
        engine, controller = SocialNavigationEngine(), SocialNavigator()
        frame = observation(self.now, distance=2)
        memory = behavior.POLICY._state('life').memory
        navigation = {'tracking': False, 'visualAvoidance': False}
        with patch.object(controller, 'step', return_value={'intent': 'searching', 'reason': 'explore'}) as step:
            engine.step(frame, navigation, memory, {}, controller, self.now, social_enabled=False)
        self.assertIsNone(step.call_args.kwargs['selected_person'])
        self.assertFalse(engine.social_enabled)
        self.assertIsNone(engine.monitor(frame, {}, controller, self.now))


class ApproachMemoryTests(unittest.TestCase):
    def test_landing_point_is_retained_until_target_or_path_changes(self):
        memory = ExplorationMemory(); memory.position, memory.rotation = np.zeros(3), np.eye(3)
        memory.cells = {(x, z): (-2., 100., 1) for x in range(-3, 7) for z in range(-3, 7)}
        plan = {}
        first = affordances.committed_approach(plan, memory, [1., .8, 1.], .45, .55)
        self.assertIsNotNone(first)
        memory.position = np.array([.3, 0., .1])
        with patch('affordances.approach_point', side_effect=AssertionError('unnecessary replan')):
            self.assertEqual(affordances.committed_approach(plan, memory, [1.01, .8, 1.], .45, .55), first)
        with patch('affordances.approach_point', return_value=[1, 0, 1.5]) as changed:
            affordances.committed_approach(plan, memory, [1.3, .8, 1.], .45, .55)
            changed.assert_called_once()

    def test_food_memory_outlives_semantics_but_reset_clears_it(self):
        options = affordances.InteractionOptions(); options.visual_client = 'eyes'
        options.objects = {'food': affordances.ObjectEvidence('food', 'sandwich', [0, .8, 1], None, [0, 0, 2, 2], 1, 100, .9),
                           'other': affordances.ObjectEvidence('other', 'table', [0, .8, 1], None, [0, 0, 2, 2], 1, 100, .9)}
        options.update(None, 'eyes', 200)
        self.assertEqual(set(options.objects), {'food'})
        options.update(None, 'new-eyes', 200)
        self.assertFalse(options.objects)


if __name__ == '__main__':
    unittest.main()
