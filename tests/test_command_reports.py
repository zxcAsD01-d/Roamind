"""Retrying sync must never replay a terminal command or its behavior feedback."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import behavior
import goals
import runtime


class CommandReportTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        goals.TASKS.clear()
        behavior.POLICY.reset()
        runtime.sync({'clientId': 'reports', 'state': {}})

    def command(self, **arguments):
        return runtime.enqueue('visual', {'clientId': 'reports', 'action': 'forward', **arguments})

    def report(self, command, status, **fields):
        runtime.sync({'clientId': 'reports', 'state': {},
                      'reports': [{'id': command['id'], 'status': status, **fields}]})

    def test_late_reports_cannot_resurrect_terminal_commands(self):
        for terminal in ('completed', 'failed', 'interrupted'):
            command = self.command()
            self.report(command, terminal, detail='original')
            for late in ('received', 'generating', 'playing', 'completed', 'failed'):
                self.report(command, late, detail='late')
            stored = runtime.COMMANDS[command['id']]
            self.assertEqual(stored['status'], terminal)
            self.assertEqual(stored['detail'], 'original')

    def test_replayed_completion_keeps_original_time_and_feedback(self):
        command = self.command(behaviorDecisionId='decision')
        with patch.object(behavior.POLICY, 'feedback_for') as feedback:
            with patch('runtime.time.time', return_value=100):
                self.report(command, 'completed')
            with patch('runtime.time.time', return_value=200):
                self.report(command, 'completed')
                self.report(command, 'playing')
            self.assertEqual(runtime.COMMANDS[command['id']]['completedAt'], 100)
            feedback.assert_called_once()
        self.assertTrue(runtime.fresh_after_motion('reports', {'observedAt': 101}))

    def test_delayed_ack_does_not_regress_playing_or_generation(self):
        command = self.command()
        for status in ('generating', 'playing'):
            self.report(command, status)
            self.report(command, 'received')
            self.assertEqual(runtime.COMMANDS[command['id']]['status'], status)
        # A later segment may generate after an earlier preset has played.
        self.report(command, 'generating')
        self.assertEqual(runtime.COMMANDS[command['id']]['status'], 'generating')

    def test_old_playing_report_cannot_undo_pending_handoff(self):
        old = self.command()
        self.report(old, 'playing')
        self.command(continuationOf=old['id'])
        self.report(old, 'playing')
        self.report(old, 'received')
        self.assertEqual(runtime.COMMANDS[old['id']]['status'], 'blending')
        self.report(old, 'completed')
        self.assertEqual(runtime.COMMANDS[old['id']]['status'], 'completed')

    def test_retry_of_failed_handoff_restores_previous_only_once(self):
        old = self.command()
        self.report(old, 'playing')
        failed = self.command(continuationOf=old['id'])
        self.report(failed, 'failed', retainedCommandId=old['id'])
        self.assertEqual(runtime.COMMANDS[old['id']]['status'], 'playing')
        self.report(old, 'completed')
        self.report(failed, 'failed', retainedCommandId=old['id'])
        self.assertEqual(runtime.COMMANDS[old['id']]['status'], 'completed')

    def test_invalid_status_and_other_client_cannot_change_command(self):
        command = self.command()
        self.report(command, 'unexpected')
        runtime.apply_report('other', {'id': command['id'], 'status': 'completed'})
        self.assertEqual(runtime.COMMANDS[command['id']]['status'], 'queued')


if __name__ == '__main__':
    unittest.main()
