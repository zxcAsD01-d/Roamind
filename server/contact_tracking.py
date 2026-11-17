"""Uncertainty-weighted visual feedback; no environment or partner ground truth."""
from __future__ import annotations

import copy
import time
import numpy as np


def check_observation(initial, current):
    if (current.get('targetId') != initial.get('targetId')
            or current.get('personId') != initial.get('personId') or current['frame'] != initial['frame']):
        raise ValueError('接触目标身份或坐标系改变')
    vectors = np.asarray([current['position'], current['normal'], current['tangent']], dtype=float)
    sigma = float(current.get('sigmaMeters', 1))
    if vectors.shape != (3, 3) or not np.isfinite(vectors).all() or not 0 < sigma <= .06:
        raise ValueError('接触目标位置或双目不确定度无效')
    if not 0 <= time.time()-float(current['observedAt']) < 1.5:
        raise ValueError('接触视觉观测已过期')
    if not np.allclose(np.linalg.norm(vectors[1:], axis=1), 1, atol=.01):
        raise ValueError('接触目标表面方向不是单位向量')
    if np.linalg.norm(vectors[0]-initial['position']) > .18:
        raise ValueError('目标超出 18 cm 局部跟踪范围，需要移动后重新规划')
    if np.dot(current['normal'], initial['normal']) < .8:
        raise ValueError('目标表面朝向改变过大，需要重新规划')


class ContactTracker:
    def __init__(self, initial):
        self.initial = copy.deepcopy(initial)
        self.last = copy.deepcopy(initial)
        self.position = np.asarray(initial['position'], dtype=float)
        self.velocity = np.zeros(3)
        self.variance = float(initial['sigmaMeters'])**2
        self.samples = 1

    def update(self, current):
        check_observation(self.initial, current)
        dt = float(current['observedAt'])-self.last['observedAt']
        if dt <= 0 or current['frameId'] == self.last['frameId']:
            return self.snapshot()
        displacement = np.asarray(current['position'])-self.last['position']
        noise = 2*np.hypot(current['sigmaMeters'], self.last['sigmaMeters'])
        if dt > 1.5 or np.linalg.norm(displacement) > .45*dt+noise:
            raise ValueError('视觉目标跳变或观测间隔过长，停止接触')
        prediction = self.position+self.velocity*dt
        variance = self.variance+(.08*dt)**2
        gain = variance/(variance+current['sigmaMeters']**2)
        next_position = prediction+gain*(np.asarray(current['position'])-prediction)
        velocity = (next_position-self.position)/dt
        velocity *= min(1., .45/max(np.linalg.norm(velocity), 1e-8))
        self.velocity = .5*self.velocity+.5*velocity
        self.position, self.variance = next_position, (1-gain)*variance
        self.last = copy.deepcopy(current)
        self.samples += 1
        return self.snapshot()

    def snapshot(self):
        return {**copy.deepcopy(self.last), 'position': self.position.tolist(),
                'velocity': self.velocity.tolist(), 'filteredSigmaMeters': float(np.sqrt(self.variance)),
                'trackingSamples': self.samples, 'measurementPosition': list(self.last['position'])}
