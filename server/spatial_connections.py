"""Stable, evidence-gated exits shared by floor topology and visual portals."""
from collections import deque

import numpy as np
from exit_memory import observe_crossing


class SpatialConnections:
    def __init__(self, resolution):
        self.resolution = resolution
        self.items = {}
        self.next_id = 1
        self.adjacency = {}
        self.last_observed_at = 0.

    def match(self, position, normal=None, source=None, used=()):
        choices = []
        for item in self.items.values():
            if item['id'] in used:
                continue
            if source is not None and source in item['_references']:
                return item
            distance = np.linalg.norm((np.asarray(position)-item['position'])[[0, 2]])
            aligned = normal is None or item['normal'] is None or abs(np.asarray(normal)@item['normal']) > .85
            if distance < .8 and aligned:
                choices.append((distance, item['id'], item))
        return min(choices, default=(0, '', None), key=lambda row: row[:2])[2]

    def remember(self, position, normal, stamp, source, reference=None, used=()):
        item = self.match(position, normal, reference, used)
        if item is None:
            identity = f'exit-{self.next_id}'
            self.next_id += 1
            item = {'id': identity, 'position': list(position), 'normal': normal, 'width': None,
                    'confidence': 0., 'uncertaintyMeters': 0., 'sources': [], 'regions': [],
                    'crossings': 0, 'lastCrossedAt': None, 'firstSeen': stamp, 'lastSeen': stamp,
                    '_references': set(), '_geometric': False, '_crossedEvents': deque(maxlen=128), '_motion': None}
            self.items[identity] = item
        item['lastSeen'] = max(item['lastSeen'], stamp)
        if source not in item['sources']:
            item['sources'].append(source)
        if reference is not None:
            item['_references'].add(reference)
        return item

    def update(self, topology, cells, portals, now):
        self._floor_cache = {}
        references = {f'portal:{key}' for key in portals} | {f'door:{key}' for key in topology.doors}
        for item in self.items.values():
            item['_geometric'] = False
            item['_references'].intersection_update(references)
        self.integrate_passages(topology.passages, now)
        self.integrate_portals(portals, now)
        self.integrate_doors(topology.doors, now)
        self.items = {key: item for key, item in self.items.items() if now-item['lastSeen'] < 1800}
        if len(self.items) > 512:
            self.items = dict(sorted(self.items.items(), key=lambda pair: pair[1]['lastSeen'])[-512:])
        for item in self.items.values():
            self.classify(item, topology, cells, now)
        self.build_graph(topology.regions)

    def integrate_passages(self, passages, now):
        used = set()
        for passage in passages:
            item = self.remember(passage['position'], passage['normal'], passage['observedAt'],
                                 'measured-free-floor', used=used)
            used.add(item['id'])
            item.update(position=passage['position'], normal=passage['normal'], width=passage['width'],
                        confidence=1., uncertaintyMeters=self.resolution/2, _geometric=True)
            passage['exitId'] = item['id']

    def integrate_portals(self, portals, now):
        for portal in portals.values():
            if now-portal['lastSeen'] >= 1800:
                continue
            item = self.remember(portal['point'], portal['normal'].tolist(), portal['lastSeen'],
                                 'pointcloud-opening', f"portal:{portal['id']}")
            if not item['_geometric']:
                item.update(position=portal['point'].tolist(), normal=portal['normal'].tolist(), width=portal['width'],
                            confidence=portal['confidence'], uncertaintyMeters=portal['uncertaintyMeters'])
            if portal['crossedAt']:
                self.record_crossing(item, portal['crossedAt'])
                item['crossings'] = max(item['crossings'], portal['crossings'])

    @staticmethod
    def record_crossing(item, stamp):
        if any(abs(stamp-previous) <= 2. for previous in item['_crossedEvents']):
            return False
        item['_crossedEvents'].append(stamp)
        item['crossings'] += 1
        item['lastCrossedAt'] = max(item['lastCrossedAt'] or 0, stamp)
        return True

    def observe(self, position, stamp):
        if stamp <= self.last_observed_at:
            return False
        gap = stamp-self.last_observed_at > 2.
        self.last_observed_at = stamp
        changed = False
        for item in self.items.values():
            if item['normal'] is None or not item['traversable']:
                item['_motion'] = None
                continue
            motion = item['_motion']
            if motion is None or gap:
                motion = {'point': np.asarray(item['position']), 'normal': np.asarray(item['normal']),
                          'width': item['width'], 'stableSide': 0, 'pendingSide': 0,
                          'lastPosition': None, 'crossings': 0}
                item['_motion'] = motion
            if observe_crossing(motion, position, stamp) and self.record_crossing(item, stamp):
                item['state'] = 'crossed'
                changed = True
        return changed

    def integrate_doors(self, doors, now):
        for door in doors.values():
            if now-door['lastSeen'] >= 1800:
                continue
            item = self.remember(door['position'], None, door['lastSeen'], 'yolo-stereo-door', f"door:{door['id']}")
            if item['normal'] is None and door['lastSeen'] >= item['lastSeen']:
                item['position'] = door['position']
            item['confidence'] = max(item['confidence'], door['confidence'])
            door['exitId'] = item['id']

    def side_regions(self, item, topology):
        if item['normal'] is None:
            x, z = np.floor(np.asarray(item['position'])[[0, 2]]/self.resolution).astype(int)
            nearby = [(dx*dx+dz*dz, topology.assignments.get((x+dx, z+dz)))
                      for dx in range(-2, 3) for dz in range(-2, 3)]
            result = []
            for _, identity in sorted(nearby, key=lambda row: row[0]):
                if identity is not None and identity not in result:
                    result.append(identity)
            return result[:2]
        point, normal = np.asarray(item['position']), np.asarray(item['normal'])
        result = []
        for side in (-1, 1):
            for distance in (.35, .7, 1.05):
                sample = point.copy()
                sample[[0, 2]] += normal*distance*side
                identity = topology.region_at(sample)
                if identity is not None:
                    if identity not in result:
                        result.append(identity)
                    break
        return result

    def aperture_state(self, item, cells, now):
        if item['normal'] is None or item['width'] is None:
            key = tuple(np.floor(np.asarray(item['position'])[[0, 2]]/self.resolution).astype(int))
            return 'blocked' if self.supported_obstacle(cells.get(key), now) else 'unconfirmed'
        normal = np.asarray(item['normal'])
        tangent = np.array([-normal[1], normal[0]])
        center = np.asarray(item['position'])[[0, 2]]
        width = item['width']
        offsets = np.linspace(-max(0., width/2-.15), max(0., width/2-.15), max(3, int(width/.175)))
        for along in (-.175, 0., .175):
            keys = [tuple(np.floor((center+normal*along+tangent*offset)/self.resolution).astype(int))
                    for offset in offsets]
            if all(self.supported_obstacle(cells.get(key), now) for key in keys):
                return 'blocked'
        if width-2*item['uncertaintyMeters'] < .7-1e-6:
            return 'narrow' if item['_geometric'] else 'unconfirmed'
        margin = max(0., width/2-.35-item['uncertaintyMeters'])
        for offset in np.linspace(-margin, margin, max(1, int(2*margin/.175)+1)):
            points = [center+tangent*offset+normal*distance for distance in np.arange(-.7, .71, .175)]
            if all(self.floor_clear(point, cells, now) for point in points):
                return 'traversable'
        return 'observed-opening' if item['_geometric'] else 'unconfirmed'

    @staticmethod
    def supported_obstacle(cell, now):
        return cell is not None and cell[0] > 1.5 and cell[2] >= 3 and now-cell[1] < 120

    def floor_clear(self, point, cells, now):
        x, z = np.floor(point/self.resolution).astype(int)
        key = (x, z)
        if key not in self._floor_cache:
            self._floor_cache[key] = all(
                (cell := cells.get((x+dx, z+dz))) is not None and cell[0] < -.15 and now-cell[1] < 1800
                for dx in (-1, 0, 1) for dz in (-1, 0, 1))
        return self._floor_cache[key]

    def classify(self, item, topology, cells, now):
        item['regions'] = self.side_regions(item, topology)
        item['state'] = self.aperture_state(item, cells, now)
        item['traversable'] = item['state'] == 'traversable'
        if item['traversable'] and item['crossings']:
            item['state'] = 'crossed'
        item['connectsRegions'] = item['traversable'] and len(item['regions']) == 2

    def build_graph(self, regions):
        self.adjacency = {identity: set() for identity in regions}
        for item in self.items.values():
            if item['connectsRegions']:
                a, b = item['regions']
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)
        pending = set(regions)
        while pending:
            start = min(pending)
            queue, group = deque([start]), set()
            pending.remove(start)
            while queue:
                identity = queue.popleft()
                group.add(identity)
                for other in sorted(self.adjacency[identity] & pending):
                    pending.remove(other)
                    queue.append(other)
            for identity in group:
                regions[identity]['componentId'] = start
                regions[identity]['connectedRegions'] = sorted(self.adjacency[identity])
                regions[identity]['reachableRegionCount'] = len(group)

    def snapshot(self):
        return [{key: value for key, value in item.items() if not key.startswith('_')}
                for item in self.items.values()]
