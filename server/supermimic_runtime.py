"""Bounded, ordered G1 inference sessions. Late results never advance history."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from supermimic_policy import SuperMimicPolicy, checked_inputs
from supermimic_physics import default_positions, motor

PROFILE = 'unitree-g1-29dof'


class RequestFailure(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    sequence: int
    history: np.ndarray
    action: np.ndarray
    updated: float


def fields(request: dict) -> tuple[str, int, bool, float, np.ndarray, np.ndarray]:
    if request.get('profile') != PROFILE:
        raise RequestFailure('This policy requires unitree-g1-29dof feedback; VRM/SMPL input is unsupported', 422)
    identity, sequence = request.get('sessionId'), request.get('sequence')
    if not isinstance(identity, str) or not 1 <= len(identity) <= 96:
        raise RequestFailure('sessionId must contain 1 to 96 characters')
    if type(sequence) is not int or not 0 <= sequence <= 2**53 - 1:
        raise RequestFailure('sequence must be a nonnegative integer')
    reset, deadline = request.get('reset', False), request.get('deadlineMs', 20)
    if type(reset) is not bool or type(deadline) not in (int, float) or not np.isfinite(deadline) or not 1 <= deadline <= 100:
        raise RequestFailure('reset must be boolean; deadlineMs must be between 1 and 100')
    try:
        observed = np.asarray(request.get('observation'), dtype=np.float32)
        goal = np.asarray(request.get('goal'), dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise RequestFailure('Invalid numeric observations or goal') from error
    if observed.shape != (99,) or goal.shape != (730,):
        raise RequestFailure('Expected observation [99] and goal [730]')
    checked_inputs(np.repeat(observed[None, None], 10, axis=1), goal[None])
    return identity, sequence, reset, float(deadline), observed, goal


class RealtimePolicy:
    def __init__(self, policy: SuperMimicPolicy):
        self.policy, self.lock = policy, threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.times: deque[float] = deque(maxlen=512)
        self.completed, self.late = 0, 0
        self.names = policy.config['control']['joint_names']
        self.default = default_positions(self.names)
        self.scale = np.array([.25 * motor(name)[0] / motor(name)[2] for name in self.names])
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='supermimic-inference')
        self.worker.submit(self._warmup).result()

    def _warmup(self) -> None:
        for _ in range(20):
            self.policy.predict(np.zeros((1, 10, 99), np.float32), np.zeros((1, 730), np.float32))

    def close(self) -> None:
        self.worker.shutdown(wait=True, cancel_futures=True)

    def infer(self, request: dict) -> dict:
        started = time.perf_counter()
        values = fields(request)
        if not self.lock.acquire(blocking=False):
            raise RequestFailure('Inference busy; retain current motion instead of queueing', 429)
        try:
            return self._infer(values, started)
        finally:
            self.lock.release()

    def _history(self, identity: str, sequence: int, reset: bool, observed: np.ndarray) -> np.ndarray:
        session = self.sessions.get(identity)
        if reset or session is None:
            if sequence != 0 or (session is None and not reset):
                raise RequestFailure('Start or expired session requires reset=true and sequence=0', 409)
            if np.max(np.abs(observed[64:93])) > 1e-6:
                raise RequestFailure('Reset requires zero previous action', 409)
            return np.repeat(observed[None, None], 10, axis=1)
        if sequence != session.sequence + 1:
            raise RequestFailure('Stale, duplicate or missing sequence; reset with actual feedback', 409)
        if not np.allclose(observed[64:93], session.action, atol=1e-5, rtol=1e-5):
            raise RequestFailure('Previous action feedback differs from last accepted output', 409)
        return np.concatenate((session.history[:, 1:], observed[None, None]), axis=1)

    def _infer(self, values: tuple, started: float) -> dict:
        identity, sequence, reset, deadline, observed, goal = values
        now = time.monotonic()
        for key in list(self.sessions):
            if now - self.sessions[key].updated > 120:
                del self.sessions[key]
        history = self._history(identity, sequence, reset, observed)
        action = self.worker.submit(self.policy.predict, history, goal[None]).result()
        elapsed = (time.perf_counter() - started) * 1000
        self.times.append(elapsed)
        if elapsed > deadline:
            self.late += 1
            raise RequestFailure('Inference exceeded deadline; output discarded, retain original motion', 504)
        self.sessions[identity] = Session(sequence, history, action.copy(), now)
        self.sessions.move_to_end(identity)
        while len(self.sessions) > 16:
            self.sessions.popitem(last=False)
        self.completed += 1
        return {'sessionId': identity, 'sequence': sequence, 'profile': PROFILE, 'backend': self.policy.backend,
                'latencyMs': elapsed, 'action': action.tolist(), 'jointNames': self.names,
                'jointTargets': (self.default + self.scale * action).tolist(), 'appliedToVrm': False}

    def health(self) -> dict:
        with self.lock:
            return {'ready': True, 'profile': PROFILE, 'backend': self.policy.backend, 'controlHz': 50,
                    'sessions': len(self.sessions), 'completed': self.completed, 'lateDiscarded': self.late,
                    'p95Ms': float(np.percentile(self.times, 95)) if self.times else None,
                    'weights': 'community-g1-lafan1-040000', 'vrmEnabled': False,
                    'vrmBlockedReason': 'G1 feedback, physics and a validated humanoid skeleton adapter are required'}
