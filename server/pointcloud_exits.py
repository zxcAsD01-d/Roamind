"""Metric opening hypotheses from remembered stereo surfaces, never room labels."""
import math
import time

import numpy as np


def wall_lines(points):
    """Bounded, deterministic RANSAC on the horizontal surface projection."""
    remaining = points.copy()
    rng = np.random.default_rng(17)
    for _ in range(6):
        if len(remaining) < 12:
            return
        best = np.zeros(len(remaining), dtype=bool)
        for _ in range(48):
            pair = remaining[rng.choice(len(remaining), 2, replace=False)]
            vector = pair[1]-pair[0]
            length = np.linalg.norm(vector)
            if length < .7:
                continue
            direction = vector/length
            normal = np.array([-direction[1], direction[0]])
            mask = np.abs((remaining-pair[0])@normal) < .09
            if mask.sum() > best.sum():
                best = mask
        if best.sum() < 12:
            return
        support = remaining[best]
        center = support.mean(axis=0)
        _, vectors = np.linalg.eigh((support-center).T@(support-center))
        direction = vectors[:, -1]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        mask = np.abs((remaining-center)@normal) < .1
        yield center, direction, normal, remaining[mask]
        remaining = remaining[~mask]


def line_gaps(center, direction, normal, support):
    order = np.argsort((support-center)@direction)
    support = support[order]
    projected = (support-center)@direction
    cuts = np.flatnonzero(np.diff(projected) > .22)
    starts, ends = np.r_[0, cuts+1], np.r_[cuts, len(projected)-1]
    for index in range(len(ends)-1):
        low, high = projected[ends[index]], projected[starts[index+1]]
        spans = (low-projected[starts[index]], projected[ends[index+1]]-high)
        if .65 <= high-low <= 2.6 and min(spans) >= .35:
            left = support[starts[index]:ends[index]+1]
            right = support[starts[index+1]:ends[index+1]+1]
            if consistent_jamb(left, direction) and consistent_jamb(right, direction):
                yield center+direction*(low+high)/2, direction, normal, high-low-.1, min(spans)


def consistent_jamb(points, direction):
    if len(points) < 5:
        return False
    centered = points-points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T@centered)
    return abs(vectors[:, -1]@direction) > .98 and values[0]/max(values[1], .001) < .04


class PointCloudExits:
    def __init__(self):
        self.last_stamp = -math.inf
        self.latest = {'source': 'remembered-stereo-pointcloud', 'candidates': [], 'sealed': False}

    def surfaces(self, world, floor, position, now):
        values = [voxel for voxel in world.voxels.values()
                  if world.labels[voxel['label']] not in world.dynamic_labels
                  and voxel.get('measured', True)
                  and now-voxel['lastSeen'] < 1800]
        if not values:
            return np.empty((0, 2)), np.empty((0, 2)), np.array([])
        points = np.asarray([voxel['point'] for voxel in values])
        near = np.linalg.norm(points[:, [0, 2]]-position[[0, 2]], axis=1) < 9
        heights = points@floor.normal+floor.offset
        walls = near & (heights > .18) & (heights < 1.9)
        floor_points = points[near & (np.abs(heights) < .10)][:, [0, 2]]
        wall_points = points[walls]
        if not len(wall_points):
            return np.empty((0, 2)), floor_points, np.array([])
        keys, inverse = np.unique(np.floor(wall_points[:, [0, 2]]/.1).astype(int), axis=0, return_inverse=True)
        low, high = np.full(len(keys), math.inf), np.full(len(keys), -math.inf)
        np.minimum.at(low, inverse, heights[walls])
        np.maximum.at(high, inverse, heights[walls])
        frames = np.zeros(len(keys))
        np.maximum.at(frames, inverse, [voxel.get('frames', 1) for voxel, keep in zip(values, walls) if keep])
        # Low furniture edges are not wall jambs. Require a tall vertical
        # surface, while leaving missing columns unknown rather than sealed.
        supported = (high-low >= .6) & (high >= 1.5)
        return (keys[supported]+.5)*.1, floor_points, frames[supported]

    def evidence(self, gap, wall_points, floor_points, frames, position, uncertainty):
        point, tangent, normal, width, span = gap
        relative = floor_points-point
        along, across = relative@normal, relative@tangent
        side = float((position[[0, 2]]-point)@normal)
        beyond = (along*(-1 if side >= 0 else 1) > .35) & (np.abs(along) < 2.) & (np.abs(across) < width/2)
        through = len(np.unique(np.floor(floor_points[beyond]/.15).astype(int), axis=0))
        close = (np.abs((wall_points-point)@normal) < .15) & (np.abs((wall_points-point)@tangent) < width/2+.8)
        observations = int(np.median(frames[close])) if close.any() else 1
        support = .3+.2*min(1., span)+.25*min(1., through/8)+.25*min(1., max(0, observations-1)/3)
        confidence = min(.98, support/(1+(uncertainty/.25)**2))
        return {'position': [round(float(point[0]), 4), 0., round(float(point[1]), 4)],
                'normal': normal.round(5).tolist(), 'width': round(float(width), 3),
                'confidence': round(confidence, 3), 'observedFrames': observations,
                'throughSupport': through, 'uncertaintyMeters': round(uncertainty, 3),
                'kind': 'pointcloud-wall-gap', 'state': 'supported-opening' if through >= 4 else 'unconfirmed-opening'}

    def analyze(self, world, floor, pose, now, calibration, ground):
        if now-self.last_stamp < .6:
            return self.latest
        started = time.perf_counter()
        self.last_stamp = now
        candidates = []
        if floor.normal is not None:
            walls, floor_points, frames = self.surfaces(world, floor, pose[:3, 3], now)
            for line in wall_lines(walls):
                for gap in line_gaps(*line):
                    distance = float(np.linalg.norm(gap[0]-pose[[0, 2], 3]))
                    residual = ground['residual'] if ground is not None else .04
                    uncertainty = residual+.5*distance**2/(calibration.fx*calibration.baseline)
                    candidate = self.evidence(gap, walls, floor_points, frames, pose[:3, 3], uncertainty)
                    candidates.append(candidate)
        self.latest = {'source': 'remembered-stereo-pointcloud', 'frame': 'visual-odometry',
                       'observedAt': now, 'revision': world.revision,
                       'candidates': sorted(candidates, key=lambda item: -item['confidence'])[:32],
                       'sealed': False, 'inferenceMs': round((time.perf_counter()-started)*1000, 2)}
        return self.latest
