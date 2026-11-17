import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import behavior
import goals
import runtime


class GoalTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        goals.TASKS.clear()
        behavior.POLICY.reset()
        self.client = 'goal-test'
        self.sync()

    def sync(self, reports=None, vision=None, **state):
        return runtime.sync({'clientId': self.client, 'reports': reports or [], 'vision': vision,
                             'state': {'position': [0, .92, 0], 'heading': 0,
                                       'playing': False, 'generating': False, **state}})

    def submit(self, steps):
        task = goals.submit({'clientId': self.client, 'title': '测试', 'steps': steps})
        interrupt = next(reversed(runtime.COMMANDS.values()))
        self.sync([{'id': interrupt['id'], 'status': 'completed'}])
        return task

    def test_wait_requires_real_playback_report_before_next_step(self):
        self.submit([{'kind': 'wait', 'duration': 1}, {'kind': 'face', 'heading': 45}])
        task = goals.snapshot(self.client)
        first = task['commandId']
        self.assertEqual(runtime.command_status(first)['arguments']['goalId'], task['id'])
        self.sync([{'id': first, 'status': 'playing'}], playing=True)
        self.assertEqual(goals.snapshot(self.client)['index'], 0)
        self.sync([{'id': first, 'status': 'completed'}])
        self.assertEqual(goals.snapshot(self.client)['index'], 1)
        self.sync()
        second = goals.snapshot(self.client)['commandId']
        self.assertEqual(runtime.command_status(second)['kind'], 'face')
        self.sync([{'id': second, 'status': 'completed'}])
        self.assertEqual(goals.snapshot(self.client)['status'], 'completed')

    def test_manual_command_cancels_all_later_steps(self):
        self.submit([{'kind': 'wait'}, {'kind': 'action', 'action': 'wave'}])
        runtime.enqueue('face', {'clientId': self.client, 'heading': 30})
        self.assertEqual(goals.snapshot(self.client)['status'], 'cancelled')
        self.sync()
        self.assertFalse(any(command['kind'] == 'execute' for command in runtime.COMMANDS.values()))

    def test_new_person_interrupts_exit_route_only_after_repeated_evidence(self):
        from test_social_behavior import observation
        self.submit([{'kind': 'social'}])
        task = goals.TASKS[self.client]
        command = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'],
                                  'action': 'forward', 'intent': 'navigate_region_exit', 'relativePath': [[0, 3]]})
        task['commandId'] = command['id']
        now = time.time()
        runtime.COMMANDS[command['id']]['created'] = now-2
        with patch('time.time', return_value=now):
            self.sync([{'id': command['id'], 'status': 'playing'}], vision=observation(now, distance=3), playing=True)
        self.assertEqual(goals.snapshot(self.client)['commandId'], command['id'])
        with patch('time.time', return_value=now+.6):
            self.sync(vision=observation(now+.6, distance=3), playing=True)
        task = goals.snapshot(self.client)
        self.assertEqual(task['status'], 'running')
        self.assertNotEqual(task['commandId'], command['id'])
        self.assertEqual(runtime.command_status(task['commandId'])['kind'], 'interrupt')
        self.assertEqual(task['events'][-1]['status'], 'person-route-update')

    def test_failed_child_stops_task_and_never_executes_next_step(self):
        self.submit([{'kind': 'wait'}, {'kind': 'action', 'action': 'wave'}])
        child = goals.snapshot(self.client)['commandId']
        self.sync([{'id': child, 'status': 'failed', 'detail': '碰撞已阻止'}])
        task = goals.snapshot(self.client)
        self.assertEqual(task['status'], 'failed')
        self.assertIn('碰撞', task['detail'])
        self.assertEqual(task['steps'][1]['status'], 'pending')

    def test_failed_handoff_waits_for_retained_motion_before_retrying(self):
        self.submit([{'kind': 'social'}])
        task = goals.TASKS[self.client]
        old = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'], 'action': 'forward'})
        runtime.COMMANDS[old['id']]['status'] = 'playing'
        new = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'],
                                        'action': 'forward', 'continuationOf': old['id']})
        task['commandId'] = new['id']
        self.sync([{'id': new['id'], 'status': 'failed', 'retainedCommandId': old['id']}], playing=True)
        self.assertEqual(task['commandId'], old['id'])
        self.assertTrue(task['handoffFailed'])
        with patch('goals.monitor_social', return_value=False), patch('goals.visual_step') as dispatch:
            for _ in range(4):
                self.sync(playing=True, bufferedSeconds=.1)
            dispatch.assert_not_called()
            self.sync([{'id': old['id'], 'status': 'completed'}])
            dispatch.assert_called_once()
        self.assertNotIn('handoffFailed', task)

    def test_navigation_collision_replans_without_abandoning_social_goal(self):
        self.submit([{'kind': 'social'}, {'kind': 'wait'}])
        task = goals.TASKS[self.client]
        command = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'], 'action': 'turn_left'})
        task['commandId'] = command['id']
        self.sync([{'id': command['id'], 'status': 'failed',
                    'detail': '预设根运动无法安全通过：test-obstacle；已拒绝该片段'}])
        updated = goals.snapshot(self.client)
        self.assertEqual(updated['status'], 'waiting')
        self.assertIsNone(updated['commandId'])
        self.assertEqual(updated['steps'][1]['status'], 'pending')
        self.assertEqual(updated['events'][-1]['status'], 'replanning')
        self.assertFalse(any(c['kind'] == 'interrupt' and c['arguments'].get('goalId') == task['id']
                             for c in runtime.COMMANDS.values()))

    def test_repeated_navigation_failures_are_bounded_and_cancellation_is_not_retried(self):
        self.submit([{'kind': 'social'}])
        task = goals.TASKS[self.client]
        failed = {'kind': 'visual', 'status': 'failed', 'detail': '视觉避障：前方深度未知'}
        for attempt in range(12):
            self.assertTrue(goals.retry_navigation(task, task['steps'][0], failed, time.time()))
        self.assertFalse(goals.retry_navigation(task, task['steps'][0], failed, time.time()))
        task['navigationFailures'] = 0
        self.assertFalse(goals.retry_navigation(task, task['steps'][0], {**failed, 'status': 'interrupted'}, time.time()))

    def test_expired_preparation_preserves_goal_and_later_steps(self):
        self.submit([{'kind': 'social'}, {'kind': 'wait'}])
        task = goals.TASKS[self.client]
        command = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'], 'action': 'forward'})
        task['commandId'] = command['id']
        self.sync([{'id': command['id'], 'status': 'failed',
                    'detail': '规划准备超过保留的动作前缀，当前动作继续完成后重新规划'}])
        self.assertEqual(task['status'], 'waiting')
        self.assertIsNone(task['commandId'])
        self.assertEqual(task['steps'][1]['status'], 'pending')

    def test_short_step_failure_learns_executable_distance_and_replans(self):
        self.submit([{'kind': 'social'}])
        task = goals.TASKS[self.client]
        command = runtime.enqueue('visual', {'clientId': self.client, 'goalId': task['id'],
                                           'action': 'forward', 'distance': .19})
        task['commandId'] = command['id']
        self.sync([{'id': command['id'], 'status': 'failed',
                    'detail': '移动距离不足一个安全落脚步；请合并短步命令'}])
        updated = goals.snapshot(self.client)
        self.assertEqual(updated['status'], 'waiting')
        self.assertGreater(behavior.POLICY.snapshot(self.client)['mission']['minimumTravel'], .19)

    def test_stale_vision_waits_then_times_out_without_motion(self):
        self.submit([{'kind': 'social', 'timeout': 1}])
        self.assertEqual(goals.snapshot(self.client)['status'], 'waiting')
        goals.TASKS[self.client]['stepStartedAt'] = time.time()-2
        self.sync(vision={'observedAt': time.time()-10})
        self.assertEqual(goals.snapshot(self.client)['status'], 'failed')
        self.assertTrue(all(command['kind'] == 'interrupt' for command in runtime.COMMANDS.values()))

    def test_goal_survives_internal_action_dispatch(self):
        with patch('action_library.LIBRARY.describe'):
            self.submit([{'kind': 'action', 'action': 'wave', 'duration': 2}])
        task = goals.snapshot(self.client)
        self.assertEqual(task['status'], 'running')
        self.assertEqual(runtime.command_status(task['commandId'])['arguments']['goalId'], task['id'])

    def test_invalid_submission_preserves_existing_task(self):
        self.submit([{'kind': 'wait'}])
        old = goals.snapshot(self.client)
        for step in ({'kind': 'teleport'}, {'kind': 'face', 'heading': float('nan')},
                     {'kind': 'wait', 'duration': 30}, {'kind': 'text-touch', 'text': ''},
                     {'kind': 'action', 'action': 'wave', 'duration': 1}):
            with self.assertRaises(ValueError):
                self.submit([step])
            self.assertEqual(goals.snapshot(self.client)['id'], old['id'])

    def test_waypoint_refuses_unobserved_route(self):
        state = behavior.POLICY._state(self.client)
        import numpy as np
        state.memory.position, state.memory.rotation = np.zeros(3), np.eye(3)
        vision = {'stereo': {'method': 'stereo-sgbm-lr', 'mapping': True, 'observedAt': time.time(),
                            'sectors': [{'angle': 0, 'confidence': 1, 'clearance': 3}]}}
        with self.assertRaisesRegex(ValueError, '视觉可通行路径'):
            behavior.POLICY.waypoint(self.client, vision, [10, 0, 10])
        result = behavior.POLICY.waypoint(self.client, vision, [.1, 0, .1])
        self.assertTrue(result['completed'])
        # Grid quantization must not strand a settled avatar a few cm outside .3 m.
        result = behavior.POLICY.waypoint(self.client, vision, [.32, 0, 0])
        self.assertTrue(result['completed'])

    def test_old_autonomy_cannot_steal_task_control(self):
        self.submit([{'kind': 'social'}])
        result = runtime.autonomy_decide({'clientId': self.client, 'vision': {}})
        self.assertTrue(result['busy'])
        self.assertEqual(goals.snapshot(self.client)['status'], 'waiting')

    def test_social_goal_waves_once_only_after_confirmed_arrival_and_actual_stop(self):
        self.submit([{'kind': 'social'}, {'kind': 'wait'}])
        fresh = {'observedAt': time.time()}
        ready = {'completed': True, 'reason': '距离与面向已确认'}
        with patch('runtime.autonomy_decide', return_value=ready):
            self.sync(vision=fresh)
        task = goals.snapshot(self.client)
        greeting = task['commandId']
        self.assertEqual(task['steps'][0]['greetingCommand'], greeting)
        self.assertEqual(runtime.command_status(greeting)['kind'], 'gesture')
        with patch('runtime.autonomy_decide', return_value=ready) as decide:
            self.sync([{'id': greeting, 'status': 'playing'}], playing=True)
            decide.assert_not_called()
        self.assertEqual(goals.snapshot(self.client)['index'], 0)
        self.sync([{'id': greeting, 'status': 'completed'}])
        self.assertEqual(goals.snapshot(self.client)['index'], 1)

    def test_cancel_without_existing_goal_still_stops_character(self):
        self.assertEqual(goals.cancel({'clientId': self.client}), {'goal': None})
        self.assertEqual(next(reversed(runtime.COMMANDS.values()))['kind'], 'interrupt')

    def test_contact_uses_generic_target_and_preserves_goal_ownership(self):
        target = {'targetId': 'point-test'}
        self.sync(perceptionClientId='camera')
        with patch('touch.resolve', return_value=target) as resolver:
            self.submit([{'kind': 'contact', 'targetId': 'point-test', 'effector': 'head',
                          'operation': 'hold', 'duration': .8}, {'kind': 'wait'}])
        task = goals.snapshot(self.client)
        command = runtime.command_status(task['commandId'])
        self.assertEqual(task['status'], 'running')
        self.assertEqual(command['arguments']['goalId'], task['id'])
        self.assertEqual(command['arguments']['effector'], 'head')
        self.assertEqual(command['arguments']['selection']['targetId'], 'point-test')
        self.sync([{'id': command['id'], 'status': 'completed'}])
        self.assertEqual(goals.snapshot(self.client)['index'], 1)
        resolver.assert_called_once()

    def test_invalid_contact_rejected_before_replacing_task(self):
        self.submit([{'kind': 'wait'}])
        old = goals.snapshot(self.client)
        for extra in ({}, {'targetId': ''}, {'targetId': 'point-test', 'effector': 'tentacle'},
                      {'targetId': 'point-test', 'operation': 'invent'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                goals.submit({'clientId': self.client, 'steps': [{'kind': 'contact', **extra}]})
            self.assertEqual(goals.snapshot(self.client)['id'], old['id'])


if __name__ == '__main__':
    unittest.main()
