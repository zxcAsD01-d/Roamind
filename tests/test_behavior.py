import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import behavior
import runtime


class VisualContinuationTests(unittest.TestCase):
    def setUp(self):
        runtime.SESSIONS.clear()
        runtime.COMMANDS.clear()
        runtime.sync({"clientId": "visual-test", "state": {
            "playing": True, "generating": False, "bufferedSeconds": .2}})
        self.command = runtime.enqueue("visual", {"clientId": "visual-test", 'action': 'forward'})
        runtime.COMMANDS[self.command["id"]]["status"] = "playing"

    def test_only_matching_playing_visual_command_can_prefetch(self):
        session = runtime.active_session("visual-test")
        commands = list(runtime.COMMANDS.values())
        args = {"afterCommand": self.command["id"]}
        self.assertEqual(runtime._visual_continuation(args, session, commands), self.command["id"])
        self.assertIsNone(runtime._visual_continuation({"afterCommand": "other"}, session, commands))
        for changes in ({"playing": False}, {"generating": True}, {"bufferedSeconds": .8}, {"bufferedSeconds": 4}):
            modified = {**session, "state": {**session["state"], **changes}}
            self.assertIsNone(runtime._visual_continuation(args, modified, commands))

    def test_turn_must_finish_before_another_visual_decision(self):
        runtime.COMMANDS[self.command['id']]['arguments']['action'] = 'turn_left'
        self.assertIsNone(runtime._visual_continuation({'afterCommand': self.command['id']},
                          runtime.active_session('visual-test'), list(runtime.COMMANDS.values())))

    def test_complete_route_is_not_replaced_by_a_short_prefetch(self):
        self.command['arguments']['relativePath'] = [[0, 2], [0, 4]]
        self.command['arguments']['continuous'] = True
        self.assertIsNone(runtime._visual_continuation({'afterCommand': self.command['id']},
                          runtime.active_session('visual-test'), list(runtime.COMMANDS.values())))

    def test_continuous_walk_prefetches_before_reserved_braking_tail(self):
        self.command['arguments']['continuous'] = True
        session = runtime.active_session('visual-test')
        session['state'].update(bufferedSeconds=2.1, locomotion={'phase': 'walk', 'brakeIn': .9})
        self.assertEqual(runtime._visual_continuation({'afterCommand': self.command['id']},
                         session, list(runtime.COMMANDS.values())), self.command['id'])
        for data in ({'phase': 'stop', 'brakeIn': .9}, {'phase': 'turn', 'brakeIn': .5},
                     {'phase': 'walk', 'brakeIn': float('nan')}, {'phase': 'walk', 'brakeIn': 2}):
            session['state']['locomotion'] = data
            self.assertIsNone(runtime._visual_continuation({'afterCommand': self.command['id']},
                             session, list(runtime.COMMANDS.values())))

    def test_finished_motion_requires_a_post_completion_frame(self):
        runtime.sync({'clientId': 'visual-test', 'state': {},
                      'reports': [{'id': self.command['id'], 'status': 'completed'}]})
        finished = runtime.command_status(self.command['id'])['completedAt']
        self.assertFalse(runtime.fresh_after_motion('visual-test', {'observedAt': finished-.1}))
        self.assertTrue(runtime.fresh_after_motion('visual-test', {'observedAt': finished+.1}))

    def test_continuation_waits_for_real_playback_completion(self):
        runtime.enqueue("visual", {"clientId": "visual-test", "continuationOf": self.command["id"]})
        self.assertEqual(runtime.command_status(self.command["id"])["status"], "blending")
        runtime.sync({"clientId": "visual-test", "state": {},
                      "reports": [{"id": self.command["id"], "status": "completed"}]})
        self.assertEqual(runtime.command_status(self.command["id"])["status"], "completed")


    def test_failed_continuation_restores_only_its_playing_predecessor(self):
        next_command = runtime.enqueue('visual', {'clientId': 'visual-test', 'continuationOf': self.command['id']})
        runtime.sync({'clientId': 'visual-test', 'state': {}, 'reports': [
            {'id': next_command['id'], 'status': 'failed', 'retainedCommandId': self.command['id']}]})
        self.assertEqual(runtime.command_status(self.command['id'])['status'], 'playing')
        self.assertEqual(runtime.command_status(next_command['id'])['retainedCommandId'], self.command['id'])

    def test_failed_unrelated_command_cannot_resurrect_completed_motion(self):
        runtime.COMMANDS[self.command['id']]['status'] = 'completed'
        next_command = runtime.enqueue('visual', {'clientId': 'visual-test'})
        runtime.sync({'clientId': 'visual-test', 'state': {}, 'reports': [
            {'id': next_command['id'], 'status': 'failed', 'retainedCommandId': self.command['id']}]})
        self.assertEqual(runtime.command_status(self.command['id'])['status'], 'completed')
        self.assertNotIn('retainedCommandId', runtime.command_status(next_command['id']))


def vision(signature, boxes=None, labels=None, faces=None, stereo=True):
    labels = labels or []
    objects = [{"box": box, "confidence": .9,
                "label": labels[index] if index < len(labels) else "object"}
               for index, box in enumerate(boxes or [])]
    result = {"signature": signature, "objects": objects, "faces": faces or [],
              "width": 640, "height": 360}
    if stereo:
        result["stereo"] = {"method": "stereo-sgbm-lr", "observedAt": __import__('time').time(),
                             "mapping": True,
                             "sectors": [{"angle": angle, "clearance": 3., "confidence": .9}
                                         for angle in range(-45, 46, 15)]}
    return result


class VisualBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.policy = behavior.BehaviorPolicy()

    def test_uses_only_visual_observation(self):
        result = self.policy.decide({"clientId": "agent", "vision": vision([.1] * 64)})
        self.assertIn(result["action"], behavior.ACTIONS)
        self.assertEqual(self.policy.config()["input"], "ego-vision-only")

    def test_scene_change_updates_novelty(self):
        self.policy.decide({"clientId": "agent", "vision": vision([.1] * 64)})
        result = self.policy.decide({"clientId": "agent", "vision": vision([.9] * 64)})
        self.assertGreater(result["features"]["novelty"], .9)

    def test_central_visual_obstacle_changes_free_space(self):
        result = self.policy.decide({"clientId": "agent", "vision": vision([.1] * 64, [[220, 180, 420, 355]], stereo=False)})
        self.assertLess(result["features"]["freeSpace"], .2)

    def test_feedback_updates_weights_without_position(self):
        result = self.policy.decide({"clientId": "agent", "vision": vision([.1] * 64)})
        before = self.policy.snapshot("agent")["weights"]
        self.policy.feedback_for("agent", result["decisionId"], "failed", "视觉碰撞已阻止")
        after = self.policy.snapshot("agent")["weights"]
        self.assertNotEqual(before, after)

    def test_person_changes_intent_and_bearing(self):
        result = self.policy.decide({"clientId": "social", "vision": vision(
            [.2] * 64, [[40, 120, 180, 340]], ["person"])})
        self.assertGreater(result["social"]["personVisible"], .05)
        self.assertLess(result["relativeBearing"], 0)
        self.assertIn(result["intent"], ("face_person", "approach_person", "social_observe"))

    def test_repeated_empty_scene_increases_people_search(self):
        first = self.policy.decide({"clientId": "search", "vision": vision([.4] * 64)})
        last = first
        for _ in range(8):
            last = self.policy.decide({"clientId": "search", "vision": vision([.4] * 64)})
        self.assertGreater(last["exploration"]["need"], first["exploration"]["need"])
        self.assertEqual(last["intent"], "search_for_people" if last["action"] == "observe" else "explore")

    def test_face_counts_as_social_target(self):
        result = self.policy.decide({"clientId": "face-social", "vision": vision(
            [.3] * 64, faces=[{"box": [420, 80, 600, 300], "confidence": .95}])})
        self.assertGreater(result["social"]["personVisible"], .05)
        self.assertGreater(result["relativeBearing"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
