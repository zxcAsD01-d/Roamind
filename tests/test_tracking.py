import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"server"))
import tracking
from planning import navigation, timed_plan
from world import segment_clear
from audio_direction import direction


class TrackingTests(unittest.TestCase):
    def setUp(self):
        with tracking.LOCK:
            tracking.TARGETS.clear()

    def test_heading_shortest_turn_and_independent_of_movement(self):
        plan=timed_plan({"startHeading":170,"steps":[{"action":"wave","end":2,"heading":-170}]})
        angles=np.array(plan["headingTrack"])
        self.assertLess(np.max(np.abs(np.diff(angles))), math.radians(7))
        self.assertAlmostEqual(angles[-1],math.radians(190))
        move=navigation({"destination":"dining","heading":90})
        self.assertAlmostEqual(move["headingTrack"][-1],math.pi/2)

    def test_latest_only_stale_and_bearing_only(self):
        tracking.observe({"id":"object","position":[-5,.92,4.5]})
        with self.assertRaises(ValueError):
            tracking.observe({"id":"object","position":[-5,.92,4.5],"observedAt":time.time()-3})
        tracking.observe({"id":"lost","position":[-5,.92,4.5],"observedAt":time.time()-3,"ttl":1})
        with self.assertRaises(ValueError): tracking.latest("lost")
        tracking.observe({"id":"bearing","bearing":30})
        with self.assertRaises(ValueError): tracking.latest("bearing",True)

    def test_replanning_uses_new_observation_and_clear_path(self):
        args={"start":[-5,.92,3.2],"targetId":"moving","distance":1.2}
        tracking.observe({"id":"moving","position":[-5,.92,0]})
        first=tracking.tracking_plan(args)
        tracking.observe({"id":"moving","position":[0,.92,0]})
        second=tracking.tracking_plan(args)
        self.assertEqual(first["frames"],40)
        self.assertFalse(second["tracking"]["usesFutureTrajectory"])
        self.assertNotEqual(first["rootTrack"],second["rootTrack"])
        for a,b in zip(second["rootTrack"],second["rootTrack"][1:]):
            self.assertTrue(segment_clear((a[0],a[2]),(b[0],b[2])))

    def test_standoff_to_furniture_centre(self):
        tracking.observe({"id":"sofa","position":[-8.8,.92,4.7]})
        planned=tracking.tracking_plan({"start":[-5,.92,3.2],"targetId":"sofa"})
        self.assertEqual(planned["frames"],40)

    def test_following_backs_away_when_too_close(self):
        start=[-5,.92,3.2]
        target=[-5,.92,2.8]
        tracking.observe({"id":"close","position":target})
        planned=tracking.tracking_plan({"start":start,"targetId":"close","distance":1.2})
        initial=math.dist(start,target)
        final=math.dist(planned["rootTrack"][-1],target)
        self.assertGreater(final,initial+.5)
        self.assertAlmostEqual(final,1.2,places=5)

    def test_audio_direction_refuses_mono_and_duplicate(self):
        signal=np.random.default_rng(1).normal(0,.1,16000)
        self.assertFalse(direction(signal[:,None],16000)["available"])
        self.assertFalse(direction(np.stack([signal,signal],axis=1),16000,.1)["available"])

    def test_calibrated_pair_direction(self):
        signal=np.random.default_rng(2).normal(0,.1,16000)
        result=direction(np.stack([np.roll(signal,2),signal],axis=1),16000,.1)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["azimuth"],math.degrees(math.asin(2/16000*343/.1)),delta=1)
        self.assertTrue(result["frontBackAmbiguous"])


if __name__=="__main__": unittest.main(verbosity=2)
