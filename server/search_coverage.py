"""Bounded directional observations and frontier exits; no room/world oracle."""
from __future__ import annotations

import math


class SearchCoverage:
    region_size = 1.4
    bins = 24

    def __init__(self):
        self.regions = {}
        self.exits = {}

    def region(self, position):
        return tuple(math.floor(float(position[index])/self.region_size+.5) for index in (0, 2))

    def observe(self, position, rotation, sectors, stamp):
        key = self.region(position)
        item = self.regions.setdefault(key, {'views': set(), 'seen': stamp})
        item['seen'] = stamp
        yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
        for sector in sectors:
            angle, clearance, confidence = (sector.get(name) for name in ('angle', 'clearance', 'confidence'))
            if (not all(type(value) in (float, int) and math.isfinite(value) for value in (angle, clearance, confidence))
                    or confidence < .25 or clearance < 0):
                continue
            item['views'].add(round((yaw-angle)/15) % self.bins)
        for exit in self.exits.values():
            angle = math.degrees(math.atan2(math.sin(math.radians(yaw-exit['direction'])),
                                           math.cos(math.radians(yaw-exit['direction']))))
            if math.hypot(position[0]-exit['position'][0], position[2]-exit['position'][2]) < .6 and abs(angle) <= 50:
                exit['lookedAt'] = stamp
        if len(self.regions) > 2048:
            oldest = min(self.regions, key=lambda cell: self.regions[cell]['seen'])
            del self.regions[oldest]

    def views_at(self, position):
        return self.regions.get(self.region(position), {}).get('views', set())

    def next_view(self, position, rotation, body_bearing=0):
        views = self.views_at(position)
        return self.choose_view(views, rotation, body_bearing)

    def choose_view(self, views, rotation, body_bearing=0, excluded=()):
        if len(views) == self.bins:
            return None
        yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
        candidates = []
        for bearing in range(-140, 141, 5):
            center = round((yaw-bearing)/15)
            destination = round((yaw-bearing)/5) % 72
            if destination in excluded:
                continue
            gain = sum((center+offset) % self.bins not in views for offset in range(-3, 4))
            if gain == 0:
                continue
            alignment = math.degrees(math.atan2(math.sin(math.radians(bearing-body_bearing)),
                                                math.cos(math.radians(bearing-body_bearing))))
            # Information per estimated motor time: a head look costs less
            # than a complete stepping turn. Zero-gain views remain excluded.
            seconds = .45+abs(bearing)/65 if abs(alignment) <= 75 else 2.4+abs(alignment)/60
            candidates.append((gain/seconds-.003*abs(bearing), bearing))
        if not candidates:
            return None
        score, bearing = max(candidates)
        return bearing if score > 0 else None

    def exit_key(self, point):
        return tuple(round(float(value)/.7) for value in point)

    def mark_exit(self, point, normal, stamp, status='unvisited'):
        # Coalesce nearby boundary cells with the same outward direction.
        key = (*self.exit_key(point), normal)
        previous = self.exits.get(key)
        if previous and previous['status'] == 'attempted' and stamp >= previous.get('retryAfter', math.inf):
            previous['status'] = 'unvisited'
        self.exits[key] = {'position': [float(point[0]), 0., float(point[1])], 'direction': (90, -90, 0, 180)[normal],
                           'status': previous['status'] if previous and status == 'unvisited' else status,
                           'seen': stamp, 'kind': 'visual-frontier-candidate'}
        if previous:
            for field in ('retryAfter', 'lookedAt'):
                if field in previous:
                    self.exits[key][field] = previous[field]
        if len(self.exits) > 512:
            del self.exits[min(self.exits, key=lambda cell: self.exits[cell]['seen'])]
        return self.exits[key]

    def snapshot(self):
        return {'regionSize': self.region_size, 'angularResolution': 15,
                'regions': [{'key': list(key), 'observedDirections': sorted(item['views']),
                             'confirmed': len(item['views']) == self.bins} for key, item in self.regions.items()],
                'exits': list(self.exits.values()), 'exitMeaning': 'observed-free-to-unknown-boundary-not-confirmed-door'}
