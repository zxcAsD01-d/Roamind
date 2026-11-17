"""Deterministic overlap and worker lifecycle checks; no model/checkpoint needed."""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from dual import DualRunner
from overlap import MotionWindow, fuse_windows, overlap_weights, smooth_window


def window(angle=0., offset=0.):
    local = np.zeros((20, 81), dtype=np.float32)
    local[:, 1] = angle
    root = np.zeros((20, 3), dtype=np.float32)
    root[:, 2] = np.arange(20)*.05+offset
    return MotionWindow(local, root)


class OverlapTest(unittest.TestCase):
    def test_blend_preserves_endpoint_positions_and_velocities(self):
        old, new = window(.2), window(.6, .3)
        original = old.local.copy()
        result = fuse_windows(old, new)
        np.testing.assert_allclose(result.root[:2], old.root[:2])
        np.testing.assert_allclose(result.root[-2:], new.root[-2:])
        np.testing.assert_allclose(result.local[:2], old.local[:2], atol=1e-6)
        np.testing.assert_allclose(result.local[-2:], new.local[-2:], atol=1e-6)
        np.testing.assert_array_equal(old.local, original)
        self.assertTrue(np.all(np.diff(overlap_weights(20)) >= 0))

    def test_rotations_follow_short_arc_across_pi(self):
        result = fuse_windows(window(np.deg2rad(179)), window(np.deg2rad(-179)))
        angles = Rotation.from_rotvec(result.local[:, :3]).magnitude()
        self.assertGreater(float(angles.min()), np.deg2rad(178.9))

    def test_bad_shapes_and_nonfinite_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            fuse_windows(window(), window().slice(0, 18))
        invalid = window()
        invalid.root[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fuse_windows(window(), invalid)
        engine = DualRunner(lambda: SimpleNamespace(horizon=40))
        try:
            with self.assertRaisesRegex(ValueError, "非有限值"):
                engine._validate_output(invalid, np.zeros(3), None)
        finally:
            engine.executor.shutdown(wait=True, cancel_futures=True)

    def test_filter_reduces_jitter_without_moving_constrained_roots(self):
        sample = window()
        sample.local[:, 1] = np.tile([-.2, .2], 10)
        original = sample.local.copy()
        filtered = smooth_window(sample, sample.slice(0, 0), None, preserve_root=True)
        self.assertLess(float(np.max(np.abs(filtered.local[2:-2, 1]))), .01)
        np.testing.assert_array_equal(filtered.root, sample.root)
        np.testing.assert_array_equal(sample.local, original)

    def test_filter_accepts_one_frame_and_uses_short_rotation_arc(self):
        sample = window(np.deg2rad(179))
        sample.local[1::2, 1] = np.deg2rad(-179)
        result = smooth_window(sample, sample.slice(0, 0), None)
        self.assertGreater(float(Rotation.from_rotvec(result.local[:, :3]).magnitude().min()), 3.1)
        single = sample.slice(0, 1)
        output = smooth_window(single, single.slice(0, 0), None)
        self.assertEqual(output.local.shape, (1, 81))
        np.testing.assert_allclose(output.local, single.local, atol=1e-6)

    def test_lanes_alternate_from_committed_history_without_a_pair_barrier(self):
        factory = lambda: SimpleNamespace(horizon=40, history_limit=160)
        engine = DualRunner(factory)
        calls = []

        def infer(lane, local, root, prompt, start, origin, plan, cancelled, pending):
            calls.append((lane, start, len(root), root[-1].copy()))
            self.assertEqual(pending is None, start == 0)
            poses = np.zeros((40, 81), dtype=np.float32)
            roots = np.repeat([[lane+start, 0., 0.]], 40, axis=0)
            return MotionWindow(poses, roots, {"lane": lane, "startFrame": start})

        engine._infer = infer
        local, root = np.zeros((30, 81)), np.zeros((30, 3))
        pending = None
        try:
            for frame in (0, 20, 40):
                output, pending = engine._advance_window(local, root, pending, "walk", frame,
                    80, np.zeros(3), None, lambda: False)
                self.assertEqual(len(calls), frame//20+1)
                np.testing.assert_array_equal(calls[-1][3], root[-1])
                self.assertEqual(len(output.root), 20)
                self.assertEqual(len(pending.root), 20)
                local, root = np.concatenate((local, output.local)), np.concatenate((root, output.root))
            self.assertEqual([call[:3] for call in calls], [(0, 0, 30), (1, 20, 50), (0, 40, 70)])
            output, tail = engine._advance_window(local, root, pending, "walk", 60,
                80, np.zeros(3), None, lambda: False)
            np.testing.assert_array_equal(output.root, pending.root)
            self.assertIsNone(tail)
            self.assertEqual(len(calls), 3, "最后的保留段不应重复推理")
        finally:
            engine.executor.shutdown(wait=True, cancel_futures=True)

    def test_failed_turn_does_not_publish_or_replace_the_pending_window(self):
        engine = DualRunner(lambda: SimpleNamespace(horizon=40, history_limit=160))
        pending = window()
        original = pending.root.copy()
        try:
            def fail():
                raise ValueError("lane failed")
            engine._infer = lambda *args: fail()
            with self.assertRaisesRegex(ValueError, "lane failed"):
                engine._advance_window(np.zeros((20, 81)), np.zeros((20, 3)), pending,
                    "walk", 20, 60, np.zeros(3), None, lambda: False)
            np.testing.assert_array_equal(pending.root, original)
            self.assertEqual(engine.last_pair, [])
        finally:
            engine.executor.shutdown(wait=True, cancel_futures=True)

    def test_timeline_action_boundary_discards_speculation_from_the_previous_prompt(self):
        engine = DualRunner(lambda: SimpleNamespace(horizon=40, history_limit=160))
        plan = {'fps': 20, 'frames': 60, 'timeline': [
            {'start': 0, 'end': 1.4}, {'start': 1.4, 'end': 3}]}
        engine._next_window = lambda *args: MotionWindow(np.zeros((40, 81)), np.zeros((40, 3)))
        local, root = np.zeros((20, 81)), np.zeros((20, 3))
        try:
            output, pending = engine._advance_window(local, root, window(), 'walk', 20, 60,
                                                     np.zeros(3), plan, lambda: False)
            self.assertEqual(len(output.root), 8)
            self.assertIsNone(pending, '新动作不能被上一段提示词生成的保留姿态锁住')
        finally:
            engine.executor.shutdown(wait=True, cancel_futures=True)

    def test_cancellation_waits_for_old_workers_before_releasing_session(self):
        from realtime import GenerationCancelled
        engine = DualRunner(lambda: SimpleNamespace(horizon=40))
        entered, release, cancelling = threading.Event(), threading.Event(), threading.Event()
        failures = []

        def work():
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test worker blocked")
            return window()

        task = engine.executor.submit(work)
        self.assertTrue(entered.wait(timeout=1))

        def cancel():
            cancelling.set()
            return True

        def collect():
            try:
                engine._wait_window(task, cancel)
            except Exception as exc:
                failures.append(exc)

        collector = threading.Thread(target=collect)
        collector.start()
        try:
            self.assertTrue(cancelling.wait(timeout=1))
            self.assertTrue(collector.is_alive(), "旧推理未结束时不得释放会话")
        finally:
            release.set()
            collector.join(timeout=2)
            engine.executor.shutdown(wait=True, cancel_futures=True)
        self.assertFalse(collector.is_alive())
        self.assertTrue(task.done())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], GenerationCancelled)

    def test_worker_timeout_closes_pipe_and_terminates_late_generation(self):
        from dual_worker import ModelProcess
        worker = ModelProcess.__new__(ModelProcess)
        worker.connection = Mock()
        worker.connection.poll.return_value = False
        worker.process = Mock()
        worker.process.is_alive.side_effect = [True, True, False]
        with self.assertRaisesRegex(TimeoutError, "推理超时"):
            worker.receive(timeout=0)
        worker.connection.close.assert_called_once()
        worker.process.terminate.assert_called_once()
        worker.process.join.assert_called_once()

    def test_worker_eof_fails_closed(self):
        from dual_worker import ModelProcess
        worker = ModelProcess.__new__(ModelProcess)
        worker.connection = Mock()
        worker.connection.poll.return_value = True
        worker.connection.recv.side_effect = EOFError("worker disconnected")
        worker.process = Mock()
        worker.process.is_alive.return_value = False
        with self.assertRaises(EOFError):
            worker.receive()
        worker.connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
