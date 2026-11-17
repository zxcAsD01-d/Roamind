import copy
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
import affordances
import behavior
import goals
import object_behavior
import runtime
import lumen_actions
from test_social_behavior import observation


def objects(stamp, label='unseen-category', position=None):
    vision = observation(stamp)
    vision['semanticFusion'] = {'fused': True, 'frameId': 12, 'associations': [
        {'id': 'object-one', 'label': label, 'worldPosition': position or [0, 0, .8],
         'surfacePixel': [310, 220], 'confidence': .9, 'points': 150, 'box': [280, 190, 350, 280]}]}
    return vision


class ObjectOptionsTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear(); runtime.COMMANDS.clear(); runtime.INTERACTIONS.clear()
        goals.TASKS.clear(); behavior.POLICY.reset()
        self.now = time.time()
        self.sync(objects(self.now))

    def sync(self, vision=None, reports=None, **state):
        return runtime.sync({'clientId': 'objects', 'vision': vision, 'reports': reports or [],
            'state': {'position': [0, .92, 0], 'heading': 0, 'perceptionClientId': 'eyes',
                      'body': {'effectors': {'right_hand': {'reach': 1., 'height': .4, 'lateral': .1}}},
                      'playing': False, 'generating': False, **state}})

    def test_unknown_category_gets_geometric_options_without_label_table(self):
        row = affordances.catalog({'clientId': 'objects'})['objects'][0]
        self.assertEqual(row['label'], 'unseen-category')
        self.assertEqual({v['operation'] for v in row['options']}, set(affordances.OPERATIONS)-set(affordances.NODE_OPERATIONS))
        self.assertTrue(next(v for v in row['options'] if v['operation'] == 'touch')['enabled'])
        self.assertNotIn('sit', [v['operation'] for v in row['options']])

    def test_low_confidence_surface_only_allows_verification(self):
        row = runtime.INTERACTIONS['objects'].objects['object-one']
        row.confidence = .25
        options = affordances.catalog({'clientId': 'objects'})['objects'][0]['options']
        self.assertEqual([v['operation'] for v in options if v['enabled']], ['inspect'])

    def test_expired_evidence_disables_every_action(self):
        options = runtime.INTERACTIONS['objects']
        row = options.snapshot(behavior.POLICY._state('objects').memory, self.now+2)['objects'][0]
        self.assertFalse(any(v['enabled'] for v in row['options']))

    def test_stale_localization_cannot_execute_fresh_object(self):
        task = {'clientId': 'objects', 'status': 'running'}
        vision = objects(self.now)
        vision['stereo']['observedAt'] = self.now-3
        self.assertIsNone(object_behavior.fresh_object(task, {'objectId': 'object-one', 'operation': 'point'}, vision))
        self.assertEqual(task['status'], 'waiting')

    def test_operation_completion_requires_post_playback_semantics(self):
        task = affordances.submit({'clientId': 'objects', 'objectId': 'object-one', 'operation': 'point'})
        current = goals.TASKS['objects']
        command = runtime.action({'clientId': 'objects', 'goalId': task['id'], 'action': 'point', 'duration': 3})
        current['commandId'] = command['id']
        current['objectPlan'] = {'phase': 'verify', 'awaiting': 'execute', 'observedAfter': 0.}
        with patch('time.time', return_value=self.now+.5):
            self.sync(objects(self.now+.1), [{'id': command['id'], 'status': 'completed'}])
        self.assertEqual(current['status'], 'waiting')
        with patch('time.time', return_value=self.now+.7):
            self.sync(objects(self.now+.6))
        self.assertEqual(current['status'], 'completed')

    def test_missing_surface_pixel_blocks_contact_but_not_observation(self):
        runtime.INTERACTIONS['objects'].objects['object-one'].pixel = None
        row = affordances.catalog({'clientId': 'objects'})['objects'][0]
        enabled = {v['operation']: v['enabled'] for v in row['options']}
        self.assertTrue(enabled['inspect']); self.assertFalse(enabled['touch'])

    def test_raw_box_without_same_frame_depth_cannot_create_object(self):
        options = affordances.InteractionOptions()
        vision = objects(self.now); vision['semanticFusion']['fused'] = False
        options.update(vision, 'eyes', self.now)
        self.assertFalse(options.objects)

    def test_visual_session_change_clears_old_object_identity(self):
        self.sync(None, perceptionClientId='other-eyes')
        self.assertFalse(runtime.INTERACTIONS['objects'].objects)

    def test_repeated_same_frame_does_not_increase_count(self):
        self.sync(objects(self.now))
        self.assertEqual(runtime.INTERACTIONS['objects'].objects['object-one'].observations, 1)

    def test_planning_is_read_only_and_execution_uses_interact_step(self):
        plan = affordances.plan({'clientId': 'objects', 'objectId': 'object-one', 'operation': 'inspect'})
        self.assertEqual(plan['steps'][0]['kind'], 'interact')
        self.assertFalse(runtime.COMMANDS)
        task = affordances.submit({'clientId': 'objects', 'objectId': 'object-one', 'operation': 'inspect'})
        self.assertEqual(task['steps'][0]['objectId'], 'object-one')
        self.assertEqual(len(runtime.COMMANDS), 1)

    def test_inspect_waits_for_actual_head_playback_and_new_semantics(self):
        task = affordances.submit({'clientId': 'objects', 'objectId': 'object-one', 'operation': 'inspect'})
        stop = next(reversed(runtime.COMMANDS))
        self.sync(objects(self.now+.01), [{'id': stop, 'status': 'completed'}])
        current = goals.TASKS['objects']; look = current['commandId']
        self.assertEqual(runtime.COMMANDS[look]['arguments']['action'], 'look')
        self.sync(objects(self.now+.02), [{'id': look, 'status': 'playing'}], playing=True)
        self.assertEqual(goals.TASKS['objects']['index'], 0)
        with patch('time.time', return_value=self.now+.5):
            self.sync(objects(self.now+.03), [{'id': look, 'status': 'completed'}])
        self.assertEqual(goals.TASKS['objects']['status'], 'waiting')
        with patch('time.time', return_value=self.now+.7):
            self.sync(objects(self.now+.6))
        self.assertEqual(goals.TASKS['objects']['status'], 'completed')

    def test_far_unknown_route_is_disabled_and_not_treated_as_free(self):
        runtime.INTERACTIONS['objects'].objects['object-one'].position = [30, 0, 30]
        row = affordances.catalog({'clientId': 'objects'})['objects'][0]
        self.assertFalse(next(v for v in row['options'] if v['operation'] == 'approach')['enabled'])
        self.assertFalse(next(v for v in row['options'] if v['operation'] == 'touch')['enabled'])

    def test_nonfinite_geometry_is_rejected(self):
        options = affordances.InteractionOptions()
        options.update(objects(self.now, position=[float('nan'), 0, 1]), 'eyes', self.now)
        self.assertFalse(options.objects)
        vision = objects(self.now)
        vision['semanticFusion']['associations'][0]['points'] = float('nan')
        options.update(vision, 'eyes', self.now)
        self.assertFalse(options.objects)

    def test_near_approach_needs_no_unnecessary_route(self):
        row = runtime.INTERACTIONS['objects'].objects['object-one']
        option = affordances.InteractionOptions._option(row, 'approach', True, .8, None, self.now)
        self.assertTrue(option['enabled'])

    def test_far_approach_uses_known_visual_path_and_retains_identity(self):
        task = {'id': 'plan', 'clientId': 'objects', 'objectPlan': {}, 'events': []}
        memory = behavior.POLICY._state('objects').memory
        row = runtime.INTERACTIONS['objects'].objects['object-one']
        row.position = [0, 0, 3.]
        result = {'action': 'forward', 'relativePath': [[0, 2]], 'duration': 4, 'distance': 2}
        with patch('affordances.approach_point', return_value=[0, 0, 2]), \
                patch.object(behavior.POLICY, 'waypoint', return_value=result) as planner:
            self.assertTrue(object_behavior.approach(task, {'objectId': row.identity}, row, memory, objects(self.now)))
        planner.assert_called_once()
        command = runtime.command_status(task['commandId'])
        self.assertEqual(command['arguments']['relativePath'], [[0, 2]])
        self.assertEqual(task['objectPlan']['awaiting'], 'approach')

    def test_gesture_does_not_supersede_root_command(self):
        move = runtime.enqueue('visual', {'clientId': 'objects', 'action': 'forward'})
        gesture = runtime.action({'clientId': 'objects', 'action': 'wave', 'duration': 3})
        self.assertEqual(gesture['kind'], 'gesture')
        self.assertEqual(runtime.command_status(move['id'])['status'], 'queued')
        runtime.enqueue('visual', {'clientId': 'objects', 'action': 'turn_left'})
        self.assertEqual(runtime.command_status(gesture['id'])['status'], 'queued')
        runtime.enqueue('interrupt', {'clientId': 'objects', 'mode': 'hold'})
        self.assertEqual(runtime.command_status(gesture['id'])['status'], 'superseded')

    def test_native_catalog_is_discovered_and_opposite_arms_run_independently(self):
        catalog = lumen_actions.capabilities()
        self.assertGreaterEqual(len(catalog['actions']), 20)
        left = next(row['action'] for row in catalog['actions'] if row['action'].startswith('lumen:left-'))
        old = runtime.action({'clientId': 'objects', 'action': left, 'duration': 3})
        right = runtime.action({'clientId': 'objects', 'action': 'wave', 'duration': 3})
        self.assertEqual(runtime.command_status(old['id'])['status'], 'queued')
        replacement = runtime.action({'clientId': 'objects', 'action': 'point', 'duration': 3})
        self.assertEqual(runtime.command_status(old['id'])['status'], 'queued')
        self.assertEqual(runtime.command_status(right['id'])['status'], 'superseded')
        self.assertEqual(replacement['kind'], 'gesture')
        validated = goals.validate_step({'kind': 'action', 'action': left, 'duration': 3})
        self.assertEqual(validated['action'], left)
        self.assertFalse(lumen_actions.supports('lumen:unknown'))

    def test_wide_object_observation_turns_in_visual_bearing_direction(self):
        memory = behavior.POLICY._state('objects').memory
        row = runtime.INTERACTIONS['objects'].objects['object-one']
        task = {'id': 'plan', 'clientId': 'objects', 'objectPlan': {}, 'events': []}
        for bearing, expected in [(70, 'turn_right'), (-70, 'turn_left')]:
            with patch.object(memory, 'bearing_to', return_value=bearing):
                object_behavior.observe(task, row, memory, objects(self.now))
            self.assertEqual(runtime.command_status(task['commandId'])['arguments']['action'], expected)

    def test_point_wraps_direction_at_180_degrees(self):
        memory = behavior.POLICY._state('objects').memory
        row = runtime.INTERACTIONS['objects'].objects['object-one']
        task = {'id': 'plan', 'clientId': 'objects', 'objectPlan': {}, 'events': []}
        with patch.object(memory, 'bearing_to', return_value=-179):
            object_behavior.operate(task, {'operation': 'point'}, row, memory,
                                    {**objects(self.now), 'viewHeading': 179, 'bodyHeading': 0})
        self.assertEqual(runtime.command_status(task['commandId'])['kind'], 'gesture')

    def test_gesture_duration_rejects_nonfinite_or_speedup(self):
        for duration in [True, float('nan'), .5, 10]:
            with self.assertRaises(ValueError):
                runtime.action({'clientId': 'objects', 'action': 'wave', 'duration': duration})


if __name__ == '__main__':
    unittest.main()
