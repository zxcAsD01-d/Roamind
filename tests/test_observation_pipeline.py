"""Concurrency must drop old work and preserve exposure identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from observation_pipeline import ObservationPipeline
from stereo_service import StereoSessions


class ObservationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ObservationPipeline()
        self.addCleanup(self.pipeline.close)
        self.body = {'left': 'rgb', 'observedAt': time.time(), 'frameId': 7, 'includeVision': True}

    def test_depth_and_detector_start_before_either_finishes(self):
        rendezvous = threading.Barrier(2, timeout=2)

        def depth(body):
            rendezvous.wait()
            return {'frameId': body['frameId']}

        def vision(body):
            self.assertEqual(body['image'], 'rgb')
            rendezvous.wait()
            return {'frameId': body['frameId'], 'observedAt': body['observedAt']}

        def fuse(body, result):
            self.assertEqual(result['frameId'], body['frameId'])
            return {**result, 'semanticFusion': {'map': {'revision': 2}}}

        result = self.pipeline.infer(self.body, depth, vision, fuse)
        self.assertEqual(result['depth']['map']['revision'], 2)
        self.assertEqual(result['vision']['observedAt'], self.body['observedAt'])
        self.assertEqual(result['pipeline']['queuedFrames'], 0)

    def test_busy_frames_are_rejected_instead_of_queued(self):
        entered, release = threading.Event(), threading.Event()

        def depth(body):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test did not release depth')
            return {}

        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.pipeline.infer, {**self.body, 'includeVision': False}, depth, None, None)
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaises(BlockingIOError):
                    self.pipeline.infer(self.body, None, None, None)
            finally:
                release.set()
            pending.result(2)

    def test_depth_failure_holds_capacity_until_running_detector_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def vision(body):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test did not release detector')
            return {}

        def depth(body):
            self.assertTrue(entered.wait(2))
            raise ValueError('invalid stereo')

        try:
            with self.assertRaisesRegex(ValueError, 'invalid stereo'):
                self.pipeline.infer(self.body, depth, vision, None)
            with self.assertRaises(BlockingIOError):
                self.pipeline.infer(self.body, None, None, None)
        finally:
            release.set()
        self.pipeline.close()
        result = self.pipeline.infer({**self.body, 'includeVision': False}, lambda _: {}, None, None)
        self.assertIsNone(result['vision'])

    def test_detector_failure_keeps_depth_with_explicit_error(self):
        def unavailable(body):
            raise RuntimeError('model unavailable')

        result = self.pipeline.infer(self.body, lambda _: {'frameId': 7}, unavailable, None)
        self.assertEqual(result['depth']['frameId'], 7)
        self.assertIn('model unavailable', result['pipeline']['visionError'])
        self.assertIsNone(result['vision'])

    def test_unexpected_detector_failure_propagates_and_releases_capacity(self):
        def broken(body):
            raise KeyError('schema')

        with self.assertRaises(KeyError):
            self.pipeline.infer(self.body, lambda _: {}, broken, None)
        result = self.pipeline.infer({**self.body, 'includeVision': False}, lambda _: {}, None, None)
        self.assertEqual(result['depth'], {})


class StereoCapacityTests(unittest.TestCase):
    def test_competing_sessions_drop_frame_and_errors_release_compute(self):
        sessions = StereoSessions()
        first = sessions.session({'clientId': 'first'})
        second = sessions.session({'clientId': 'second'})
        with first.lock:
            with self.assertRaises(BlockingIOError):
                sessions.infer({'clientId': 'first'})
        # Invalid input must release both the session and global locks.
        with self.assertRaises(ValueError):
            sessions.infer({'clientId': 'second'})
        self.assertTrue(sessions.compute.acquire(blocking=False))
        sessions.compute.release()
        self.assertTrue(second.lock.acquire(blocking=False))
        second.lock.release()
        with sessions.compute:
            with self.assertRaises(BlockingIOError):
                sessions.infer({'clientId': 'first'})


if __name__ == '__main__':
    unittest.main()
