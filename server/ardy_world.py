"""Walkable support and clearance in a fused, measured visual-odometry map."""
from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

CELL = .24
BODY_RADIUS = .28
MAX_POINTS = 40000


def point3(value, label):
    if (not isinstance(value, list) or len(value) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in value)):
        raise ValueError(f'{label} 必须是三个有限的米制坐标')
    return np.asarray(value, dtype=float)


class FusedWorld:
    def __init__(self, snapshot):
        if snapshot.get('frame') != 'visual-odometry' or not snapshot.get('mapId'):
            raise ValueError('需要融合点云的 visual-odometry 快照')
        self.id, self.revision = snapshot['mapId'], snapshot['revision']
        self.normal = point3(snapshot.get('floorNormal'), 'floorNormal')
        self.offset, self.voxel = float(snapshot['floorOffset']), float(snapshot['voxelSize'])
        if (not math.isfinite(self.offset) or not .02 <= self.voxel <= .15 or
                abs(np.linalg.norm(self.normal)-1) > .01 or self.normal[1] < .99):
            raise ValueError('当前 ARDY 站姿规划需要有效的近水平实测地面')
        raw = snapshot.get('measuredPoints')
        if not isinstance(raw, list) or not 3 <= len(raw) <= MAX_POINTS:
            raise ValueError('融合点云的实测体素不足或超出容量')
        points = np.asarray(raw, dtype=float)
        if points.shape != (len(raw), 3) or not np.isfinite(points).all():
            raise ValueError('融合点云坐标无效')
        height = points@self.normal+self.offset
        floor = points[np.abs(height) <= .06]
        obstacles = points[(height > .06) & (height < 2.3)]
        self.obstacle_tree = cKDTree(obstacles)
        self.planar_tree = cKDTree(obstacles[:, [0, 2]])
        support = Counter(self.key(xz) for xz in floor[:, [0, 2]])
        self.floor_cells = {cell for cell, count in support.items() if count >= 2}
        centers = np.asarray([self.center(cell) for cell in sorted(self.floor_cells)]).reshape(-1, 2)
        clear = self.planar_tree.query(centers)[0] >= self.clearance
        self.walkable = {self.key(xz) for xz in centers[clear]}

    @property
    def clearance(self):
        return BODY_RADIUS+self.voxel/math.sqrt(2)

    @staticmethod
    def key(xz):
        return tuple(np.floor(np.asarray(xz)/CELL).astype(int))

    @staticmethod
    def center(key):
        return (np.asarray(key)+.5)*CELL

    def floor_y(self, xz):
        return -(np.asarray(xz)@self.normal[[0, 2]]+self.offset)/self.normal[1]

    def clear_route(self, a, b):
        count = max(1, math.ceil(np.linalg.norm(b-a)/(.25*CELL)))
        samples = np.linspace(a, b, count+1)
        return (all(self.key(xz) in self.floor_cells for xz in samples) and
                bool(np.all(self.planar_tree.query(samples)[0] >= self.clearance)))

    def route(self, start, destination):
        if self.clear_route(start, destination):
            return [start, destination]
        a, b = self.key(start), self.key(destination)
        if (a not in self.walkable or b not in self.walkable or
                not self.clear_route(start, self.center(a)) or
                not self.clear_route(self.center(b), destination)):
            return None
        cells = self._search(a, b)
        if cells is None:
            return None
        return self._shortcut([start, *(self.center(cell) for cell in cells), destination])

    def _search(self, start, end):
        frontier = [(0, start)]
        costs, parents = {start: 0}, {start: None}
        while frontier:
            score, cell = heapq.heappop(frontier)
            estimate = abs(cell[0]-end[0])+abs(cell[1]-end[1])
            if score > costs[cell]+estimate:
                continue
            if cell == end:
                route = [cell]
                while parents[route[-1]] is not None:
                    route.append(parents[route[-1]])
                return route[::-1]
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                near = (cell[0]+dx, cell[1]+dz)
                cost = costs[cell]+1
                if near not in self.walkable or cost >= costs.get(near, math.inf):
                    continue
                if self.clear_route(self.center(cell), self.center(near)):
                    costs[near], parents[near] = cost, cell
                    heapq.heappush(frontier, (cost+abs(near[0]-end[0])+abs(near[1]-end[1]), near))
        return None

    def _shortcut(self, points):
        route, i = [points[0]], 0
        while i < len(points)-1:
            j = i+1
            while j+1 < len(points) and self.clear_route(points[i], points[j+1]):
                j += 1
            route.append(points[j])
            i = j
        return route

    def clear_sight(self, start, target):
        distance = np.linalg.norm(target-start)
        if distance <= .25:
            return False
        end = start+(target-start)*(1-.22/distance)
        samples = np.linspace(start, end, max(2, math.ceil(distance/.05)))
        return bool(np.all(self.obstacle_tree.query(samples)[0] > self.voxel))

    def clear_body(self, joints, parents):
        """Sample bone capsules, including pose midpoints between rendered samples."""
        posed = np.asarray(joints)
        if len(posed) > 1:
            posed = np.concatenate((posed, (posed[:-1]+posed[1:])*.5))
        weights = np.linspace(0, 1, 9)[None, None, :, None]
        samples = posed[:, parents[1:], None]*(1-weights)+posed[:, 1:, None]*weights
        radii = np.full(len(parents)-1, .045)
        radii[5] = .10  # head
        radii[[9, 10, 11, 15, 16, 17]] = .035  # hands
        distances = self.obstacle_tree.query(samples.reshape(-1, 3))[0].reshape(samples.shape[:-1])
        return bool(np.all(distances > radii[None, :, None]+self.voxel*math.sqrt(3)/2))
