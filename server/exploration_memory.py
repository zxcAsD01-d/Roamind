"""Bounded evidence map built only from stereo sectors and visual odometry."""
from __future__ import annotations

import math
import time
import heapq
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from search_coverage import SearchCoverage
from floor_topology import FloorTopology
from exit_memory import ExitMemory


@dataclass
class Reachability:
    parents: dict
    lengths: dict
    penalties: dict
    costs: dict | None = None

    def __contains__(self, key):
        return key in self.parents

    def path(self, key):
        if key not in self.parents:
            return None
        result = []
        while key is not None:
            result.append(key)
            key = self.parents[key]
        return result[::-1]


class ExplorationMemory:
    resolution = .35

    def __init__(self):
        self.cells = {}
        self.people = {}
        self.visits = {}
        self.coverage = SearchCoverage()
        self.topology = FloorTopology(self.resolution)
        self.exit_memory = ExitMemory()
        self.ground = None
        self.last_stamp = 0.
        self.position = None
        self.rotation = None
        self.scan_sign = 1
        self.stalls = 0
        self.last_action = ''
        self.previous_position = None
        self.goal = None
        self.goal_intent = 'explore_frontier'
        self.goal_since = 0.
        self.arrived = {}
        self.rejected = {}
        self.rejected_motions = []
        self.last_visit = None
        self.map_revision = 0
        self.reach_cache = None
        self.reach_signature = None
        self.explore_cache = None
        self.explore_signature = None
        self.planner_runs = 0
        self.planner_hits = 0
        self.planner_ms = 0.
        self.planner_total_ms = 0.
        self.last_cleanup = 0.
        self.clearance_revision = -1
        self.clearance_cache = {}

    def key(self, point):
        return tuple(int(value) for value in np.floor(np.asarray(point)[[0, 2]]/self.resolution))

    def mark(self, key, evidence, stamp):
        old = self.cells.get(key, (0., 0., 0))
        if stamp < old[1]:
            return
        value = max(-4., min(4., old[0]+evidence))
        # Old free-floor frames cannot count as repeated occupied evidence.
        frames = 1 if old[0]*evidence < 0 else old[2]+int(stamp != old[1])
        if (key not in self.cells or self.cell_class(old[0]) != self.cell_class(value)
                or (old[0] > 1.5 and old[2] >= 3) != (value > 1.5 and frames >= 3)):
            self.map_revision += 1
        self.cells[key] = (value, stamp, frames)

    @staticmethod
    def cell_class(value):
        return -1 if value < -.15 else 1 if value > .4 else 0

    def update(self, stereo, people, now):
        if not isinstance(stereo, dict) or not stereo.get('mapping'):
            return False
        pose = stereo.get('pose', {})
        matrix = np.asarray(pose.get('cameraToWorld', []), dtype=float)
        stamp = stereo.get('observedAt', 0)
        if matrix.size != 16 or not np.isfinite(matrix).all() or not 0 <= now-stamp <= 1.5:
            return False
        if stamp < self.last_stamp:
            return False
        if stamp == self.last_stamp:
            return True
        transform = matrix.reshape(4, 4)
        body = np.asarray(pose.get('bodyPosition', transform[:3, 3]), dtype=float)
        if body.shape != (3,) or not np.isfinite(body).all():
            raise ValueError('视觉身体定位必须是三个有限坐标')
        self.position, self.rotation = body, transform[:3, :3]
        self.last_stamp = stamp
        ground = stereo.get('groundInfo')
        if isinstance(ground, dict):
            self.ground = {**ground, 'observedAt': stamp, 'source': 'stereo-ransac-gravity'}
        self.coverage.observe(self.position, self.rotation, stereo.get('sectors', []), stamp)
        current = self.key(self.position)
        if current != self.last_visit:
            self.visits[current] = self.visits.get(current, 0)+1
            self.last_visit = current
        grid = stereo.get('navigationGrid')
        measured_grid = isinstance(grid, dict) and grid.get('frame') == 'visual-odometry'
        for sector in stereo.get('sectors', []):
            self.ray(sector, stamp, mark_obstacle=not measured_grid)
        if measured_grid:
            self.integrate_grid(grid, stamp)
        self.exit_memory.integrate(stereo.get('exitEvidence'), now)
        self.exit_memory.observe(self.position, stamp)
        self.confirm_arrival(stamp)
        self.remember_people(people, now)
        self.cleanup(now)
        self.topology.update(self.cells, self.map_revision, now, self.exit_memory.portals)
        if self.topology.connections.observe(self.position, stamp):
            self.topology.version += 1
        return True

    def observe_semantics(self, observation, now):
        if observation.get('semanticReady') is False:
            return
        self.topology.observe(observation, now)
        stamp = observation.get('semanticObservedAt', 0)
        if type(stamp) in (int, float) and 0 <= now-stamp < 1.5:
            self.remember_people(observation.get('semanticFusion', {}).get('associations', []), stamp)

    def maintain(self, now):
        self.cleanup(now)
        self.exit_memory.expire(now)
        self.topology.update(self.cells, self.map_revision, now, self.exit_memory.portals)

    def cleanup(self, now):
        if now-self.last_cleanup < 5 and len(self.cells) <= 16000:
            return
        self.last_cleanup = now
        previous_size = len(self.cells)
        self.cells = {key: value for key, value in self.cells.items() if now-value[1] < 1800}
        if len(self.cells) > 16000:
            self.cells = dict(sorted(self.cells.items(), key=lambda pair: pair[1][1])[-16000:])
        if len(self.cells) != previous_size:
            self.map_revision += 1
        self.people = {key: value for key, value in self.people.items() if now-value['seen'] < 300}
        self.rejected = {key: expiry for key, expiry in self.rejected.items() if expiry > now}
        self.expire_motions(now)
        if len(self.visits) > 16000:
            self.visits = {key: self.visits[key] for key in self.cells if key in self.visits}

    def integrate_grid(self, grid, stamp):
        for name, evidence in [('free', -.35), ('occupied', .9)]:
            points = grid.get(name, [])
            if not isinstance(points, list) or len(points) > 6000:
                raise ValueError('视觉地面网格尺寸无效')
            keys = set()
            for point in points:
                if (not isinstance(point, list) or len(point) != 2
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)):
                    raise ValueError('视觉地面网格坐标无效')
                keys.add(self.key([point[0], 0, point[1]]))
            for key in keys:
                self.mark(key, evidence, stamp)

    def ray(self, sector, stamp, mark_obstacle=True):
        clearance, confidence = sector.get('clearance'), sector.get('confidence', 0)
        angle = sector.get('angle')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (clearance, confidence, angle)) or confidence < .25:
            return
        angle = math.radians(angle)
        axes = self.horizontal_axes()
        if axes is None:
            return
        forward, right = axes
        direction = forward*math.cos(angle)+right*math.sin(angle)
        distances = np.arange(0, min(clearance, 8), self.resolution*.5)
        # The depth sector checks a 0.8 m corridor, not an infinitesimal ray.
        lateral = np.cross(direction, np.array([0., 1., 0.]))[[0, 2]]
        center = self.position[[0, 2]]+distances[:, None]*direction[[0, 2]]
        points = (center[:, None, :]+np.array([-.35, -.175, 0., .175, .35])[None, :, None]*lateral).reshape(-1, 2)
        keys = np.floor(points/self.resolution).astype(int)
        for key in dict.fromkeys(map(tuple, keys)):
            self.mark(key, -.35*confidence, stamp)
        obstacle = sector.get('obstacleDistance')
        if mark_obstacle and isinstance(obstacle, (int, float)) and math.isfinite(obstacle) and .1 < obstacle < 8:
            self.mark(self.key(self.position+direction*(obstacle+.15)), .9*confidence, stamp)

    def horizontal_axes(self):
        forward = self.rotation[:, 2].copy()
        forward[1] = 0
        length = np.linalg.norm(forward)
        if length < .2:
            return None
        forward /= length
        return forward, np.cross(forward, np.array([0., 1., 0.]))

    def bearing_to(self, delta):
        axes = self.horizontal_axes()
        if axes is None:
            return 180.
        forward, right = axes
        return math.degrees(math.atan2(delta@right, delta@forward))

    def remember_people(self, people, now):
        for person in people:
            point = person.get('worldPosition')
            if person.get('label') not in ('person', 'face', 'human') or not isinstance(point, list) or len(point) != 3:
                continue
            point = np.asarray(point, dtype=float)
            if not np.isfinite(point).all():
                continue
            key = person.get('id', str(self.key(point)))
            self.people[key] = {'position': point.tolist(), 'seen': now, 'confidence': person.get('confidence', .5)}

    @staticmethod
    def neighbors(key):
        x, z = key
        return [(x+1, z), (x-1, z), (x, z+1), (x, z-1)]

    def reachable(self, *, explore_at=None):
        start = self.key(self.position)
        axes = self.horizontal_axes()
        forward = axes[0] if axes is not None else np.zeros(3)
        directions = tuple(sorted(self.neighbors((0, 0)),
                                  key=lambda step: -(step[0]*forward[0]+step[1]*forward[2])))
        signature = (self.map_revision, start, directions)
        trail = [] if explore_at is None else self.exit_memory.recent_trail(explore_at)
        if trail:
            signature += (trail[0][1], trail[-1][1])
            cached = self.explore_cache if signature == self.explore_signature else None
        else:
            cached = self.reach_cache if signature == self.reach_signature else None
        if cached is not None:
            self.planner_hits += 1
            return cached
        started = time.perf_counter()
        clearance = self.clearance_penalties()
        penalties = self.exploration_penalties(clearance, trail) if trail else clearance
        paths = self.search_routes(start, directions, clearance, penalties)
        if trail:
            self.explore_signature, self.explore_cache = signature, paths
        else:
            self.reach_signature, self.reach_cache = signature, paths
        self.planner_runs += 1
        self.planner_ms = (time.perf_counter()-started)*1000
        self.planner_total_ms += self.planner_ms
        return paths

    def exploration_penalties(self, clearance, trail):
        if not clearance:
            return clearance
        points = [point[[0, 2]] for point, _ in trail]
        cells = np.asarray(list(clearance))
        distance = cKDTree(points).query((cells+.5)*self.resolution)[0]
        cost = 2.5*np.maximum(0., 1-distance/.7)
        return {key: base+extra for (key, base), extra in zip(clearance.items(), cost.tolist())}

    def search_routes(self, start, directions, clearance, penalties):
        unsafe = {(key[0]+dx, key[1]+dz) for key, cell in self.cells.items() if cell[0] > .4
                  for dx in range(-1, 2) for dz in range(-1, 2)}
        unsafe.update(key for key, cell in self.cells.items() if cell[0] < -.15 and not self.floor_supported(key))
        queue, parents, lengths, costs = [(0., 0, start)], {start: None}, {start: 1}, {start: 0.}
        sequence = 0
        while queue:
            cost, _, key = heapq.heappop(queue)
            if cost > costs[key]:
                continue
            for dx, dz in directions:
                neighbor = (key[0]+dx, key[1]+dz)
                cell = self.cells.get(neighbor)
                if cell is None or cell[0] >= -.15:
                    continue
                if neighbor in unsafe and (key not in unsafe or clearance.get(neighbor, 1.) >= clearance.get(key, 1.)):
                    continue
                if self.rejected_motions and self.motion_blocked(self.cell_point(key), self.cell_point(neighbor)):
                    continue
                if neighbor not in parents and len(parents) >= 6000:
                    continue
                candidate = cost+penalties.get(neighbor, 1.)
                if candidate >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = candidate
                parents[neighbor], lengths[neighbor] = key, lengths[key]+1
                sequence += 1
                heapq.heappush(queue, (candidate, sequence, neighbor))
        return Reachability(parents, lengths, penalties, costs)

    def near_obstacle(self, key):
        return any(self.cells.get((key[0]+dx, key[1]+dz), (0,))[0] > .4
                   for dx in range(-1, 2) for dz in range(-1, 2))

    def floor_supported(self, key):
        return all(self.cells.get((key[0]+dx, key[1]+dz), (0,))[0] < -.15
                   for dx in range(-1, 2) for dz in range(-1, 2))

    def clearance_penalties(self):
        if self.clearance_revision == self.map_revision:
            return self.clearance_cache
        self.clearance_cache = self.compute_clearance_penalties()
        self.clearance_revision = self.map_revision
        return self.clearance_cache

    def compute_clearance_penalties(self):
        free = [key for key, cell in self.cells.items() if cell[0] < -.15]
        observed = set(free)
        boundary = {neighbor for key in free for neighbor in self.neighbors(key) if neighbor not in observed}
        occupied = {key for key, cell in self.cells.items() if cell[0] > .4}
        constraints = list(occupied | boundary)
        if not free or not constraints:
            return {}
        distances = cKDTree(constraints).query(free)[0]*self.resolution
        penalties = 1+6*np.maximum(0., 1-distances/1.2)**2
        return dict(zip(free, penalties.tolist()))

    def cell_point(self, key):
        return (np.asarray(key)+.5)*self.resolution

    def reject_motion(self, motion, now):
        if motion is None:
            return
        origin, target = np.asarray(motion[0]), np.asarray(motion[1])
        delta = target-origin
        length = np.linalg.norm(delta)
        if length < .01:
            return
        self.rejected_motions.append((origin, delta/length, min(1., length), now+30))
        self.rejected_motions = self.rejected_motions[-16:]
        self.map_revision += 1

    def expire_motions(self, now):
        active = [motion for motion in self.rejected_motions if motion[3] > now]
        if len(active) != len(self.rejected_motions):
            self.rejected_motions = active
            self.map_revision += 1

    def motion_blocked(self, start, target):
        for origin, direction, length, _ in self.rejected_motions:
            # Reject only travel deeper into the failed direction. A stop may
            # already be inside that region; backing out must remain possible.
            if (target-start)@direction <= .01:
                continue
            if np.linalg.norm(start-origin) > length+.4+np.linalg.norm(target-start):
                continue
            for point in np.linspace(start, target, 5):
                offset = point-origin
                along = offset@direction
                lateral = np.linalg.norm(offset-direction*along)
                if .08 < along < length+.2 and lateral < .15+along*.4:
                    return True
        return False

    def frontier_novelty(self, key, visited):
        # A new grid cell beside an old footprint is not a new exploration area.
        nearest = 4. if visited is None else min(4., visited.query(key)[0]*self.resolution)
        pressure = 0.
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                visits = self.visits.get((key[0]+dx, key[1]+dz), 0)
                if not visits:
                    continue
                distance = math.hypot(dx, dz)*self.resolution
                pressure += min(3, visits)*max(0., 1-distance/.9)
        return nearest, pressure

    def confirm_arrival(self, stamp):
        if self.goal is None or self.distance_to(self.goal) > .3:
            return
        self.arrived[self.goal] = stamp+90.
        if len(self.arrived) > 64:
            del self.arrived[min(self.arrived, key=self.arrived.get)]
        self.goal = None

    def goal_on_cooldown(self, key, now):
        # A neighboring cell in the same arrival pocket is not a new target.
        return any(self.arrived.get((key[0]+dx, key[1]+dz), 0.) > now
                   for dx in range(-1, 2) for dz in range(-1, 2))

    def unknown_directions(self, key):
        return [index for index, (dx, dz) in enumerate(self.neighbors((0, 0)))
                if (key[0]+dx, key[1]+dz) not in self.cells or (
                    self.cells[key[0]+dx, key[1]+dz][0] < -.15 and (key[0]+2*dx, key[1]+2*dz) not in self.cells)]

    def choose_goal(self, paths, now):
        candidates = []
        visited = cKDTree(list(self.visits)) if self.visits else None
        for key, length in paths.lengths.items():
            if length < 3 or self.rejected.get(key, 0) > now or self.goal_on_cooldown(key, now):
                continue
            frontier_directions = self.unknown_directions(key)
            frontier = len(frontier_directions)
            point = np.array([(key[0]+.5)*self.resolution, 0, (key[1]+.5)*self.resolution])
            social = max((p['confidence']*math.exp(-(now-p['seen'])/90)/
                          (1+np.linalg.norm((point-np.asarray(p['position']))[[0, 2]]))
                          for p in self.people.values()), default=0)
            if not frontier and social < .1:
                continue
            exit_penalty = 0.
            if frontier:
                normal = frontier_directions[0]
                exit = self.coverage.mark_exit(self.cell_point(key), normal, now)
                exit_penalty = 2. if exit['status'] == 'attempted' and now < exit.get('retryAfter', 0) else 0.
            novelty, visits = self.frontier_novelty(key, visited)
            bearing = abs(self.bearing_to(point-self.position))
            cost = paths.costs[key] if paths.costs is not None else length-1
            score = 2*min(1, frontier)+social*14+1.2*novelty-.15*cost*self.resolution-.35*visits-.008*bearing-exit_penalty
            candidates.append((score, key, social))
        ranked = sorted(candidates, key=lambda value: -value[0])[:32]
        scored = []
        for score, key, social in ranked:
            x, z = self.cell_point(key)
            penalty = self.exit_memory.return_penalty(self.position, [x, 0., z], now)
            scored.append((score-penalty, key, social))
        return max(scored, default=None, key=lambda value: value[0])

    def guidance(self, now):
        if self.position is None:
            return {'available': False, 'reason': '等待视觉定位'}
        self.expire_motions(now)
        self.confirm_arrival(now)
        self.arrived = {key: until for key, until in self.arrived.items() if until > now}
        paths = self.reachable(explore_at=now)
        if self.stalls >= 2:
            self.reject_goal(now)
        if (self.goal in paths and paths.lengths[self.goal] >= 2 and self.stalls < 2
                and self.distance_to(self.goal) > .3
                and self.rejected.get(self.goal, 0) <= now):
            return self.route_to(self.goal, self.goal_intent, paths)
        goal = self.choose_goal(paths, now)
        exit_key = self.topology.preferred_exit(paths, self.position, self.rejected, now)
        if exit_key is not None:
            x, z = self.cell_point(exit_key)
            if (self.goal_on_cooldown(exit_key, now) or
                    self.exit_memory.return_penalty(self.position, [x, 0., z], now) > 4):
                exit_key = None
        if exit_key is not None:
            goal = (0., exit_key, 0.)
        if goal is None:
            self.goal = None
            return {'available': True, 'bearing': self.scan_sign*35, 'intent': 'scan_unknown',
                    'goal': None, 'stalls': self.stalls}
        _, key, social = goal
        self.goal, self.goal_since = key, now
        self.goal_intent = ('seek_remembered_people' if social > .15 else
                            'navigate_region_exit' if exit_key is not None else 'explore_frontier')
        return self.route_to(key, self.goal_intent, paths)

    def route_to(self, key, intent, paths=None):
        paths = self.reachable() if paths is None else paths
        path = paths.path(key)
        if not path:
            return None
        # Keep a complete, bounded route. Each shortcut crosses measured free
        # floor, with the same clearance cost as the grid search.
        waypoints = self.route_points(path, paths)
        if not waypoints:
            return {'available': False, 'goal': None, 'intent': intent, 'stalls': self.stalls,
                    'reason': '等待当前位置到路径的新鲜净空证据'}
        next_key = waypoints[0]
        delta = self.cell_point(next_key)-self.position[[0, 2]]
        relative = []
        for cell in waypoints:
            vector = np.array([(cell[0]+.5)*self.resolution, self.position[1], (cell[1]+.5)*self.resolution])-self.position
            relative.append([round(self.bearing_to(vector), 3), round(float(np.linalg.norm(vector[[0, 2]])), 4)])
        vector = np.array([delta[0], 0., delta[1]])
        return {'available': True, 'bearing': self.bearing_to(vector), 'goal': list(key),
                'intent': intent, 'stalls': self.stalls, 'distance': float(np.linalg.norm(delta)),
                'relativePath': relative, 'path': [list(cell) for cell in path],
                'source': 'visual-evidence-only'}

    def route_points(self, path, paths):
        result, index, distance = [], 0, 0.
        start = self.position[[0, 2]].copy()
        while index < len(path)-1 and len(result) < 32:
            end, route_cost = index, 0.
            for candidate in range(index+1, min(len(path), index+25)):
                target = self.cell_point(path[candidate])
                length = float(np.linalg.norm(target-start))
                if length > 3. or distance+length > 12.:
                    break
                route_cost += paths.penalties.get(path[candidate], 1.)*self.resolution
                if candidate != index+1 and self.shortcut_cost(path[candidate], paths, start) > route_cost*1.08:
                    continue
                if self.clear_segment(path[candidate], start):
                    end = candidate
            if end == index:
                break
            distance += float(np.linalg.norm(self.cell_point(path[end])-start))
            result.append(path[end])
            start, index = self.cell_point(path[end]), end
        return result

    def distance_to(self, key):
        return float(np.linalg.norm((np.asarray(key)+.5)*self.resolution-self.position[[0, 2]]))

    def clear_segment(self, key, start=None):
        start = self.position[[0, 2]] if start is None else np.asarray(start)
        target = (np.asarray(key)+.5)*self.resolution
        if self.motion_blocked(start, target):
            return False
        count = max(2, math.ceil(np.linalg.norm(target-start)/(self.resolution*.1))+1)
        previous = tuple(np.floor(start/self.resolution).astype(int))
        cached = self.reach_cache is not None and self.reach_signature[0] == self.map_revision
        penalties = self.reach_cache.penalties if cached else self.clearance_penalties()
        escaping = self.near_obstacle(previous) or not self.floor_supported(previous)
        for point in np.linspace(start, target, count):
            current = tuple(np.floor(point/self.resolution).astype(int))
            crossed = [current]
            if current[0] != previous[0] and current[1] != previous[1]:
                crossed.extend([(previous[0], current[1]), (current[0], previous[1])])
            for cell in crossed:
                if self.cells.get(cell, (0,))[0] >= -.15:
                    return False
                if self.near_obstacle(cell) or not self.floor_supported(cell):
                    if not escaping or (cell != previous and penalties.get(cell, 1.) >= penalties.get(previous, 1.)-.01):
                        return False
                else:
                    escaping = False
            previous = current
        return True

    def shortcut_cost(self, key, paths, start=None):
        start = self.position[[0, 2]] if start is None else np.asarray(start)
        target = self.cell_point(key)
        distance = np.linalg.norm(target-start)
        points = np.linspace(start, target, max(2, math.ceil(distance/.07)))
        keys = np.floor(points/self.resolution).astype(int)
        return distance*sum(paths.penalties.get(tuple(cell), 1.) for cell in keys)/len(keys)

    def reject_goal(self, now):
        if self.goal is not None:
            self.rejected[self.goal] = now+30
            point = self.cell_point(self.goal)
            for exit in self.coverage.exits.values():
                if np.linalg.norm(np.asarray(exit['position'])[[0, 2]]-point) < .6:
                    exit['status'] = 'attempted'
                    exit['retryAfter'] = now+30
        self.goal = None

    def retreat_goal(self, person, now):
        if self.position is None:
            return None
        angle = math.radians(person['bearing'])
        target = self.position+self.rotation@np.array([math.sin(angle), 0, math.cos(angle)])*person['distance']
        options = []
        for key, length in self.reachable().lengths.items():
            point = np.array([(key[0]+.5)*self.resolution, self.position[1], (key[1]+.5)*self.resolution])
            distance = np.linalg.norm((point-target)[[0, 2]])
            if .95 <= distance <= 1.6 and 2 <= length <= 7 and not self.rejected.get(key, 0) > now:
                options.append((abs(distance-1.3)+.1*length, key))
        return min(options, default=(0, None))[1]

    def social_detour(self, person, now):
        if self.position is None:
            return None
        angle = math.radians(person['bearing'])
        target = self.position+self.rotation@np.array([math.sin(angle), 0, math.cos(angle)])*person['distance']
        options = []
        for key, length in self.reachable().lengths.items():
            if not 3 <= length <= 16 or self.rejected.get(key, 0) > now:
                continue
            point = np.array([(key[0]+.5)*self.resolution, self.position[1], (key[1]+.5)*self.resolution])
            distance = np.linalg.norm((point-target)[[0, 2]])
            if .95 <= distance < person['distance']-.15:
                options.append((abs(distance-1.3)+.12*length, key))
        return min(options, default=(0, None))[1]

    def approach_goal(self, person, now, previous=None):
        point = person.get('worldPosition')
        if self.position is None or not isinstance(point, list) or len(point) != 3 or not np.isfinite(point).all():
            return None
        target, paths = np.asarray(point)[[0, 2]], self.reachable()
        direction = target-self.position[[0, 2]]
        direction /= max(float(np.linalg.norm(direction)), .01)
        if (previous in paths and self.rejected.get(previous, 0) <= now
                and 1.25 <= np.linalg.norm(self.cell_point(previous)-target) <= 1.7):
            return previous
        options = []
        for key, length in paths.lengths.items():
            if length < 2 or self.rejected.get(key, 0) > now:
                continue
            distance = np.linalg.norm(self.cell_point(key)-target)
            if 1.35 <= distance <= 1.65 and self.floor_supported(key):
                offset = self.cell_point(key)-self.position[[0, 2]]
                lateral = abs(offset[0]*direction[1]-offset[1]*direction[0])
                options.append((2*abs(distance-1.5)+lateral+.04*length, key))
        return min(options, default=(0, None))[1]

    def record_action(self, action):
        if self.last_action == 'forward' and self.position is not None and self.previous_position is not None:
            moved = np.linalg.norm((self.position-self.previous_position)[[0, 2]])
            self.stalls = min(12, self.stalls+1) if moved < .12 else 0
        self.previous_position = None if self.position is None else self.position.copy()
        self.last_action = action

    def snapshot(self):
        counts = {'open': 0, 'closed': 0, 'uncertain': 0}
        for evidence, _, _ in self.cells.values():
            counts['open' if evidence < -.15 else 'closed' if evidence > .4 else 'uncertain'] += 1
        return {'source': 'stereo-rays-visual-odometry', 'resolution': self.resolution, 'cells': counts,
                'position': None if self.position is None else self.position.tolist(),
                'observedAt': self.last_stamp,
                'knownCells': [[int(key[0]), int(key[1]), self.cell_class(value[0]), self.topology.assignments.get(key)]
                               for key, value in list(self.cells.items())[-6000:]],
                'unknown': 'unobserved-cells', 'peopleRegions': list(self.people.values()),
                'visitedRegions': len(self.visits), 'stalls': self.stalls,
                'searchCoverage': self.coverage.snapshot(),
                'floor': self.ground, 'topology': self.topology.snapshot(self.position),
                'exitMemory': self.exit_memory.snapshot(),
                'goal': None if self.goal is None else list(self.goal),
                'rejectedGoals': len(self.rejected), 'memorySeconds': 1800, 'sealedRoomClaim': False,
                'rejectedMotions': len(self.rejected_motions),
                'planner': {'runs': self.planner_runs, 'cacheHits': self.planner_hits,
                            'lastMs': round(self.planner_ms, 3),
                            'meanMs': round(self.planner_total_ms/max(1, self.planner_runs), 3),
                            'reachableCells': len(self.reach_cache.parents) if self.reach_cache else 0}}
