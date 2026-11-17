"""Bounded robust position estimates; detector confidence is not metric accuracy."""
import math

import numpy as np


def position_estimate(raw, stamp, previous):
    point = np.asarray(raw['worldPosition'], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or not math.isfinite(stamp):
        raise ValueError('物体位置和观测时间必须有限')
    sigma = raw.get('positionSigmaMeters', .08)
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 < sigma <= 5:
        raise ValueError('物体位置误差需要正的有限米数')
    samples = [] if previous is None else [item for item in previous.samples if 0 < stamp-item[0] < 3.]
    if samples:
        center = np.median([item[1] for item in samples], axis=0)
        residuals = np.linalg.norm(np.asarray([item[1] for item in samples])-center, axis=1)
        prior_sigma = float(np.median([item[2] for item in samples]))
        gate = max(.15, 3*max(sigma, prior_sigma, float(np.median(residuals))*1.4826))
        if np.linalg.norm(point-center) > gate:
            return None
    samples.append((stamp, point.tolist(), float(sigma)))
    samples = samples[-8:]
    points = np.asarray([item[1] for item in samples])
    center = np.median(points, axis=0)
    spread = float(np.median(np.linalg.norm(points-center, axis=1)))*1.4826
    # A repeated box must not divide systematic stereo uncertainty by sqrt(N).
    uncertainty = max(float(np.median([item[2] for item in samples])), spread)
    return center.tolist(), uncertainty, samples
