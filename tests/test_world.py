import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"server"))
from world import get_world, inside, route, segment_clear
from planning import navigation, timed_plan
from collision import body_collisions


class WorldTest(unittest.TestCase):
    def test_room_counts_and_reachable_targets(self):
        world = get_world()
        kinds = [r["kind"] for r in world["rooms"]]
        self.assertEqual(kinds.count("bedroom"), 3)
        self.assertEqual(kinds.count("computer"), 1)
        self.assertEqual(kinds.count("living"), 2)
        self.assertEqual(kinds.count("bathroom"), 3)
        self.assertEqual(kinds.count("kitchen"), 1)
        for room in world["rooms"]:
            path = route(world["spawn"], room["target"])
            self.assertEqual(path[-1], room["target"])
            for a,b in zip(path,path[1:]):
                self.assertTrue(segment_clear((a[0],a[2]),(b[0],b[2])), room["id"])
            plan = navigation({"destination":room["id"]})
            for a,b in zip(plan["rootTrack"],plan["rootTrack"][1:]):
                self.assertTrue(segment_clear((a[0],a[2]),(b[0],b[2])), room["id"])

    def test_all_room_pairs_route_without_corner_cutting(self):
        rooms = get_world()["rooms"]
        for a in rooms:
            for b in rooms:
                path = route(a["target"],b["target"])
                self.assertTrue(all(segment_clear((p[0],p[2]),(q[0],q[2])) for p,q in zip(path,path[1:])))

    def test_invalid_destinations_and_impossible_deadlines(self):
        with self.assertRaises(ValueError):
            navigation({"destination":[-8.8,.92,4.7]})
        with self.assertRaises(ValueError):
            navigation({"destination":"bedroom4","duration":1})
        with self.assertRaises(ValueError):
            timed_plan({"steps":[{"action":"wave","start":0,"end":2,"destination":"kitchen"}]})
        with self.assertRaises(ValueError):
            timed_plan({"steps":[{"action":"walk","start":0,"end":float('nan')}]})

    def test_timeline_gaps_boundaries_and_dense_clearance(self):
        plan = timed_plan({"steps":[{"action":"walk","start":0,"end":14,"destination":"dining"},
                                    {"action":"wave","start":15,"end":18}]})
        self.assertEqual(plan["frames"],360)
        self.assertEqual(len(plan["rootTrack"]),len(plan["promptTrack"]))
        self.assertEqual(plan["promptTrack"][300],"A person waves their hand.")
        for root in plan["rootTrack"]:
            self.assertFalse(inside(root[0],root[2]))
        np.testing.assert_allclose(plan["rootTrack"][299],plan["rootTrack"][280])

    def test_standing_actions_leave_pelvis_height_free_and_low_postures_constrain_it(self):
        plan = timed_plan({'steps': [{'action': 'walk', 'start': 0, 'end': 2},
                                     {'action': 'crouch', 'start': 2, 'end': 4}]})
        self.assertEqual(plan['heightTrack'][:40], [False]*40)
        self.assertEqual(plan['heightTrack'][40:], [True]*40)

    def test_short_actions_and_malformed_steps(self):
        plan = timed_plan({"steps":[{"action":"wave","start":0,"end":.05}]})
        self.assertEqual(plan["frames"], 1)
        for steps in [[None], [{"action":"wave","end":None}],
                      [{"action":"walk","end":.05,"destination":[-4.999,.92,3.2]}]]:
            with self.assertRaises(ValueError):
                timed_plan({"steps":steps})

    def test_bone_segment_collision_not_only_joint_endpoints(self):
        # Bone spans through the coffee table while its endpoints are outside.
        joints=np.array([[[-8,.4,5.4],[-5,.4,5.4]]],dtype=np.float32)
        self.assertTrue(body_collisions(joints,[-1,0]))
        self.assertFalse(body_collisions(joints+np.array([0,4,0]),[-1,0]))

    def test_round_capsule_allows_clear_diagonal_but_rejects_wall_crossing(self):
        self.assertTrue(segment_clear((-5.4,.6),(-7.2,-.8)))
        self.assertFalse(segment_clear((-5.4,.6),(-6.4,-2)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
