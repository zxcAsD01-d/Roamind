"""Observed floor regions and passages; never reads simulator rooms or doors."""
from collections import Counter, deque
import math

import numpy as np
from scipy.spatial import cKDTree
from spatial_boundaries import describe_boundaries, describe_spaces
from spatial_connections import SpatialConnections


def neighbors(key):
    x, z = key
    return ((x-1, z), (x+1, z), (x, z-1), (x, z+1))


def components(cells):
    pending, groups = set(cells), []
    while pending:
        group, queue = [], deque([pending.pop()])
        while queue:
            key = queue.popleft()
            group.append(key)
            for other in neighbors(key):
                if other in pending:
                    pending.remove(other)
                    queue.append(other)
        groups.append(group)
    return groups


class FloorTopology:
    def __init__(self, resolution=.35):
        self.resolution = resolution
        self.assignments, self.regions, self.doors, self.passages = {}, {}, {}, []
        self.next_id, self.revision, self.updated_at = 1, -1, 0.
        self.semantic_stamp = 0.
        self.observations = deque(maxlen=512)
        self.connections = SpatialConnections(resolution)
        self.portals = {}
        self.version, self.expires_at = 0, 0.
        self.evidence_signature = None
        self.cells = {}
        self.spaces = []

    def point(self, key):
        return (np.asarray(key)+.5)*self.resolution

    def region_at(self, position):
        key = tuple(np.floor(np.asarray(position)[[0, 2]]/self.resolution).astype(int))
        return self.assignments.get(key)

    def update(self, cells, revision, now, portals=None):
        self.portals = portals if portals is not None else self.portals
        signature = (revision, tuple((key, item['lastSeen'], item['crossings'])
                                     for key, item in self.portals.items()))
        elapsed = now-self.updated_at
        if elapsed < 1. or (signature == self.evidence_signature and now < self.expires_at and elapsed < 5.):
            return
        self.evidence_signature = signature
        self.revision, self.updated_at = revision, now
        self.cells = {key: value for key, value in cells.items() if now-value[1] < 1800}
        free = {key for key, value in self.cells.items() if value[0] < -.15}
        if not free:
            self.assignments, self.regions, self.passages = {}, {}, []
        elif free != self.assignments.keys():
            self.partition_floor(free)
            self.find_passages()
        self.describe(self.cells, now)
        for passage in self.passages:
            passage['observedAt'] = min(self.cells[key][1] for key in passage['cells'])
        self.doors = {key: item for key, item in self.doors.items() if now-item['lastSeen'] < 1800}
        self.observations = deque((item for item in self.observations if now-item[3] < 1800), maxlen=512)
        self.apply_observations()
        self.refresh_connections(now)
        self.expires_at = self.next_expiry(now)

    def partition_floor(self, free):
        if not free:
            return
        boundary = [key for key in free if any(other not in free for other in neighbors(key))]
        ordered = sorted(free)
        distances = cKDTree(boundary).query(ordered)[0]*self.resolution
        # A doorway must remain a narrow connection rather than merging both
        # broad spaces. This is a metric clearance scale, not a room layout.
        cores = {key for key, distance in zip(ordered, distances) if distance >= .9}
        seeds = sorted((group for group in components(cores) if len(group) >= 4), key=lambda group: -len(group))[:128]
        self.partition(free, seeds)

    def next_expiry(self, now):
        expiries = [stamp+age for evidence, stamp, frames in self.cells.values()
                    for age in ((120, 1800) if evidence > 1.5 and frames >= 3 else (1800,)) if stamp+age > now]
        expiries.extend(item['lastSeen']+1800 for item in [*self.doors.values(), *self.portals.values()]
                        if item['lastSeen']+1800 > now)
        return min(expiries, default=now+1800)

    def partition(self, free, seeds):
        previous, assigned, queue, used = self.assignments, {}, deque(), set()
        for group in sorted(seeds, key=lambda group: (-len(group), min(group))):
            counts = Counter(previous[key] for key in group if key in previous)
            identity = next((key for key, _ in counts.most_common() if key not in used), None)
            if identity is None:
                identity, self.next_id = self.next_id, self.next_id+1
            used.add(identity)
            for key in sorted(group):
                assigned[key] = identity
                queue.append(key)
        self.expand(queue, assigned, free)
        for group in sorted(components(free-set(assigned)), key=lambda group: (-len(group), min(group)))[:128]:
            counts = Counter(previous[key] for key in group if key in previous)
            identity = next((key for key, _ in counts.most_common() if key not in used), None)
            if identity is None:
                identity, self.next_id = self.next_id, self.next_id+1
            used.add(identity)
            for key in group:
                assigned[key] = identity
        self.assignments = assigned

    @staticmethod
    def expand(queue, assigned, free):
        while queue:
            key = queue.popleft()
            for other in neighbors(key):
                if other in free and other not in assigned:
                    assigned[other] = assigned[key]
                    queue.append(other)

    def describe(self, cells, now):
        boundaries = describe_boundaries(self.assignments, cells, self.resolution, now, neighbors)
        grouped = {}
        for key, identity in self.assignments.items():
            grouped.setdefault(identity, []).append(key)
        self.regions = {}
        for identity, keys in grouped.items():
            boundary = boundaries[identity]
            counts = boundary['counts']
            unknown, supported = counts.get('unknown', 0), counts.get('supported', 0)
            external = sum(counts.get(state, 0) for state in ('unknown', 'supported', 'uncertain'))
            confidence = supported/max(1, sum(counts.values()))
            closed = external > 0 and supported == external and not counts.get('opening', 0)
            center = self.point(np.mean(keys, axis=0)).tolist()
            self.regions[identity] = {'id': identity, 'center': [center[0], 0., center[1]],
                                     'area': round(len(keys)*self.resolution**2, 2),
                                     'boundary': ('open-unobserved' if unknown else 'observed-enclosed' if closed else
                                                  'uncertain-boundary' if supported < external else 'observed-open'),
                                     'enclosureConfidence': round(confidence, 3),
                                     'boundaryBlocksTravel': False,
                                     'unknownBoundaryCells': unknown, 'views': set(), 'lastPersonAt': None,
                                     'boundaryEvidence': counts, 'boundarySegments': boundary['segments'],
                                     'perimeterMeters': round(sum(counts.values())*self.resolution, 3),
                                     'searchedAt': None, 'source': 'measured-free-floor-clearance-partition'}

    def find_passages(self):
        boundaries = {}
        for key, identity in self.assignments.items():
            for other in neighbors(key):
                target = self.assignments.get(other)
                if target is not None and identity != target:
                    pair = tuple(sorted((identity, target)))
                    if identity < target:
                        boundaries.setdefault(pair, []).append((key, other))
        self.passages = []
        for pair, edges in sorted(boundaries.items()):
            cells = {key for edge in edges for key in edge}
            for group in sorted(components(cells), key=min):
                if len(self.passages) >= 256:
                    return
                center = self.point(np.mean(group, axis=0)).tolist()
                selected = set(group)
                links = [(a, b) for a, b in edges if a in selected]
                normal = np.mean([np.subtract(b, a) for a, b in links], axis=0)
                if np.linalg.norm(normal) < .1:
                    normal = np.subtract(links[0][1], links[0][0])
                normal = normal/np.linalg.norm(normal)
                tangent = np.array([-normal[1], normal[0]])
                projected = np.asarray(group)@tangent
                width = float((np.ptp(projected)+1)*self.resolution)
                self.passages.append({'regions': list(pair), 'position': [center[0], 0., center[1]],
                                      'normal': normal.tolist(), 'width': round(width, 3),
                                      'kind': 'geometric-passage', 'cells': group})

    def refresh_connections(self, now):
        self.connections.update(self, self.cells, self.portals, now)
        for region in self.regions.values():
            exits = [item for item in self.connections.items.values() if region['id'] in item['regions']]
            region['exitIds'] = [item['id'] for item in exits]
            status = 'connected' if region['connectedRegions'] else region['boundary']
            if status == 'observed-enclosed' and any(item['normal'] is not None and item['state'] != 'blocked'
                                                     for item in exits):
                status = 'uncertain-boundary'
            region['enclosure'] = {'status': status, 'scope': 'observed-floor-boundary',
                                   'closedBoundary': status == 'observed-enclosed', 'observedAt': now}
            region['boundary'] = status if status == 'uncertain-boundary' else region['boundary']
        for item in [*self.doors.values(), *self.passages]:
            connection = self.connections.items.get(item.get('exitId'))
            item['traversable'] = bool(connection and connection['traversable'])
            item['state'] = connection['state'] if connection else 'unconfirmed'
        self.spaces = describe_spaces(self.regions, self.connections.items.values(), self.resolution, now)
        self.version += 1

    def observe(self, observation, now):
        stamp = observation.get('semanticObservedAt')
        if type(stamp) not in (int, float) or not 0 <= now-stamp < 1.5 or stamp <= self.semantic_stamp:
            return
        raw = observation.get('semanticPose', observation.get('stereo', {}).get('pose', {}).get('cameraToWorld'))
        matrix = np.asarray(raw, dtype=float)
        if matrix.size != 16 or not np.isfinite(matrix).all():
            return
        self.semantic_stamp = stamp
        matrix = matrix.reshape(4, 4)
        yaw = math.degrees(math.atan2(matrix[0, 2], matrix[2, 2]))
        people = any(item.get('confidence', 0) >= .5 and item.get('label') in ('person', 'human', 'face')
                     for item in [*observation.get('objects', []), *observation.get('faces', [])])
        people = people or bool(observation.get('people'))
        position = np.asarray(observation.get('semanticBodyPosition', matrix[:3, 3]), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('语义帧的视觉身体位置无效')
        self.observations.append((position.copy(), yaw, people, stamp))
        self.remember_doors(observation.get('semanticFusion', {}).get('associations', []), stamp)
        self.apply_observations()
        self.refresh_connections(now)
        self.expires_at = min(self.expires_at, self.next_expiry(now))
        for door in self.doors.values():
            if np.linalg.norm((matrix[:3, 3]-door['position'])[[0, 2]]) < .6:
                identity = self.region_at(matrix[:3, 3])
                if identity is not None and identity not in door['visitedFrom']:
                    door['visitedFrom'].append(identity)

    def remember_doors(self, associations, stamp):
        for item in associations:
            point = item.get('worldPosition')
            if (item.get('label') != 'door' or item.get('confidence', 0) < .5
                    or not isinstance(point, list) or len(point) != 3 or not np.isfinite(point).all()):
                continue
            previous = self.doors.get(item['id'], {})
            if stamp < previous.get('lastSeen', 0):
                continue
            self.doors[item['id']] = {'id': item['id'], 'position': point, 'lastSeen': stamp,
                                      'confidence': item['confidence'], 'kind': 'yolo-stereo-door',
                                      'visitedFrom': previous.get('visitedFrom', []),
                                      'traversable': False}
        self.doors = dict(sorted(self.doors.items(), key=lambda item: item[1]['lastSeen'])[-128:])
        self.doors = {key: item for key, item in self.doors.items() if stamp-item['lastSeen'] < 1800}

    def apply_observations(self):
        for position, yaw, people, stamp in self.observations:
            region = self.regions.get(self.region_at(position))
            if region is None:
                continue
            region['views'].update(round((yaw+angle)/15) % 24 for angle in range(-45, 46, 15))
            if people:
                region['lastPersonAt'] = max(region['lastPersonAt'] or 0, stamp)
            if len(region['views']) == 24:
                region['searchedAt'] = max(region['searchedAt'] or 0, stamp)

    def preferred_exit(self, reachable, position, rejected, now):
        current, options = self.region_at(position), []
        hop = self.unsearched_hop(current)
        for passage in self.connections.items.values():
            if not passage['connectsRegions'] or current not in passage['regions'] or hop not in passage['regions']:
                continue
            other = next(identity for identity in passage['regions'] if identity != current)
            center = np.asarray(passage['position'])[[0, 2]]/self.resolution
            origin = np.floor(center).astype(int)
            nearby = ((int(origin[0]+x), int(origin[1]+z)) for x in range(-5, 6) for z in range(-5, 6))
            for key in nearby:
                if key not in reachable:
                    continue
                if self.assignments.get(key) != other or rejected.get(key, 0) > now:
                    continue
                point = self.point(key)
                distance = np.linalg.norm(point-np.asarray(passage['position'])[[0, 2]])
                if .7 <= distance <= 1.5:
                    searched = self.regions[other]['searchedAt'] is not None
                    options.append((int(searched)*10+reachable.lengths[key]*.05+abs(distance-1.), key))
        return min(options, default=(0, None), key=lambda item: item[0])[1] or self.door_goal(reachable, position, rejected, now)

    def unsearched_hop(self, current):
        if current not in self.regions:
            return None
        queue, seen = deque([(current, None)]), {current}
        while queue:
            identity, first = queue.popleft()
            if identity != current and self.regions[identity]['searchedAt'] is None:
                return first
            for other in sorted(self.connections.adjacency.get(identity, ())):
                if other not in seen:
                    seen.add(other)
                    queue.append((other, other if first is None else first))
        return None

    def door_goal(self, reachable, position, rejected, now):
        current, options = self.region_at(position), []
        for door in self.doors.values():
            if not door['traversable'] or current in door['visitedFrom']:
                continue
            point = np.asarray(door['position'])[[0, 2]]
            origin = np.floor(point/self.resolution).astype(int)
            for x in range(-3, 4):
                for z in range(-3, 4):
                    key = (int(origin[0]+x), int(origin[1]+z))
                    if key not in reachable or rejected.get(key, 0) > now or reachable.lengths[key] < 3:
                        continue
                    distance = np.linalg.norm(self.point(key)-point)
                    if distance < .7:
                        options.append((distance+reachable.lengths[key]*.025, key))
        return min(options, default=(0, None), key=lambda item: item[0])[1]

    def snapshot(self, position):
        current = None if position is None else self.region_at(position)
        component = self.regions.get(current, {}).get('componentId')
        segments = [{**segment, 'regionId': item['id']} for item in self.regions.values()
                    for segment in item['boundarySegments']]
        return {'currentRegion': current, 'revision': self.version,
                'regions': [{**{key: value for key, value in item.items() if key != 'boundarySegments'},
                             'views': sorted(item['views']), 'personChecked': item['searchedAt'] is not None}
                            for item in self.regions.values()],
                'boundarySegments': segments[:2048], 'boundarySegmentsTruncated': len(segments) > 2048,
                'exits': self.connections.snapshot(),
                'reachableRegions': [item['id'] for item in self.regions.values() if item['componentId'] == component],
                'components': len({item['componentId'] for item in self.regions.values()}),
                'spaces': self.spaces,
                'passages': [{key: value for key, value in item.items() if key != 'cells'} for item in self.passages],
                'doors': list(self.doors.values()), 'sealedRoomClaim': False,
                'boundaryMeaning': 'unknown-boundary-is-open-unobserved-not-a-wall'}
