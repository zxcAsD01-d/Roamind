"""Visual portal crossings and recent travel; region partitions are advisory."""
from collections import deque
import math

import numpy as np

RECENT_TRAVEL_SECONDS = 90.


def observe_crossing(item, position, stamp):
    """Advance a stable portal plane; return whether a measured crossing completed."""
    point, normal = item['point'][[0, 2]], item['normal']
    current = position[[0, 2]]
    signed = float((current-point)@normal)
    side = 1 if signed > .35 else -1 if signed < -.35 else 0
    previous, crossed = item['lastPosition'], False
    if previous is not None:
        old_signed = float((previous-point)@normal)
        delta = current-previous
        if ((old_signed <= 0 < signed or old_signed >= 0 > signed) and np.linalg.norm(delta) <= 1.5):
            crossing = previous+delta*(-old_signed/(signed-old_signed))
            tangent = np.array([-normal[1], normal[0]])
            if abs((crossing-point)@tangent) <= item['width']/2+.15:
                item['pendingSide'] = 1 if signed > 0 else -1
    if side and side != item['stableSide']:
        if item['stableSide'] and item['pendingSide'] == side:
            item.update(crossedAt=stamp, arrivalSide=side, crossings=item['crossings']+1)
            crossed = True
        item['stableSide'], item['pendingSide'] = side, 0
    item['lastPosition'] = current.copy()
    return crossed


class ExitMemory:
    def __init__(self):
        self.portals = {}
        self.trail = deque(maxlen=256)
        self.next_id = 1
        self.crossings = deque(maxlen=128)
        self.last_observed_at = 0.

    def expire(self, now):
        self.portals = {key: value for key, value in self.portals.items() if now-value['lastSeen'] < 1800}
        while self.trail and now-self.trail[0][1] >= 1800:
            self.trail.popleft()

    def integrate(self, evidence, now):
        self.expire(now)
        if evidence is None:
            return
        if not isinstance(evidence, dict) or evidence.get('frame') != 'visual-odometry':
            raise ValueError('点云出口需要视觉坐标系')
        stamp, candidates = evidence.get('observedAt', 0), evidence.get('candidates', [])
        if type(stamp) not in (int, float) or not 0 <= now-stamp < 2:
            return
        if not isinstance(candidates, list) or len(candidates) > 32:
            raise ValueError('点云出口候选数量无效')
        for candidate in candidates:
            self.remember(candidate, stamp)
        if len(self.portals) > 128:
            self.portals = dict(sorted(self.portals.items(), key=lambda item: item[1]['lastSeen'])[-128:])

    def remember(self, candidate, stamp):
        point = np.asarray(candidate.get('position', []), dtype=float)
        normal = np.asarray(candidate.get('normal', []), dtype=float)
        values = [candidate.get(key) for key in ('width', 'confidence', 'uncertaintyMeters')]
        if (point.shape != (3,) or normal.shape != (2,) or not np.isfinite(point).all()
                or not np.isfinite(normal).all() or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
            raise ValueError('点云出口几何或置信度无效')
        width, confidence, uncertainty = values
        if not .4 <= width <= 3 or not 0 <= confidence <= 1 or uncertainty < 0 or np.linalg.norm(normal) < .9:
            raise ValueError('点云出口参数超出范围')
        normal /= np.linalg.norm(normal)
        nearby = [item for item in self.portals.values()
                  if np.linalg.norm((point-item['point'])[[0, 2]]) < .75 and abs(normal@item['normal']) > .85]
        item = min(nearby, key=lambda row: np.linalg.norm(point-row['point'])) if nearby else None
        if item is None:
            item = {'id': self.next_id, 'point': point, 'normal': normal, 'stableSide': 0,
                    'lastPosition': None, 'pendingSide': 0, 'crossedAt': 0., 'arrivalSide': 0, 'crossings': 0}
            self.portals[self.next_id] = item
            self.next_id += 1
        if stamp <= item.get('lastSeen', 0):
            return
        # Keep the crossing plane stable after first use; changing room labels
        # and jamb estimates must not manufacture an opposite-side transition.
        if not item['stableSide']:
            item['point'] = point
        item.update(lastSeen=stamp, width=width, confidence=confidence,
                    uncertaintyMeters=uncertainty, kind=candidate.get('kind', 'pointcloud-wall-gap'))

    def observe(self, position, stamp):
        if stamp <= self.last_observed_at:
            return
        if self.last_observed_at and stamp-self.last_observed_at > 2:
            for item in self.portals.values():
                item.update(lastPosition=None, stableSide=0, pendingSide=0)
        self.last_observed_at = stamp
        self.expire(stamp)
        for item in self.portals.values():
            self.observe_portal(item, position, stamp)
        if not self.trail or np.linalg.norm((position-self.trail[-1][0])[[0, 2]]) > .35:
            self.trail.append((position.copy(), stamp))
        while self.trail and stamp-self.trail[0][1] > 1800:
            self.trail.popleft()

    def observe_portal(self, item, position, stamp):
        if observe_crossing(item, position, stamp):
            self.crossings.append({'portalId': item['id'], 'at': stamp, 'arrivalSide': item['arrivalSide']})

    def return_penalty(self, start, target, now):
        if start is None:
            return 0.
        origin, destination = np.asarray(start)[[0, 2]], np.asarray(target)[[0, 2]]
        penalty = 0.
        for item in self.portals.values():
            if not item['crossings']:
                continue
            point, normal = item['point'][[0, 2]], item['normal']
            a, b = float((origin-point)@normal), float((destination-point)@normal)
            if a*item['arrivalSide'] < -.2 or b*item['arrivalSide'] >= .2 or abs(a-b) < .01:
                continue
            crossing = origin+(destination-origin)*a/(a-b)
            if np.linalg.norm(crossing-point) < item['width']/2+.3:
                strength = 10.+10.*max(0., 1-(now-item['crossedAt'])/120)
                penalty = max(penalty, strength*item['confidence'])
        return penalty

    def recent_trail(self, now, min_age=0.):
        return [(point, stamp) for point, stamp in self.trail if min_age <= now-stamp < RECENT_TRAVEL_SECONDS]

    def travel_penalty(self, start, target, now):
        return self.return_penalty(start, target, now)+self.retrace_penalty(start, target, now)

    def retrace_penalty(self, start, target, now):
        if start is None or len(self.trail) < 3:
            return 0.
        penalty = 0.
        origin, destination = np.asarray(start)[[0, 2]], np.asarray(target)[[0, 2]]
        trail = self.recent_trail(now, min_age=2.)
        recent = np.asarray([point[[0, 2]] for point, _ in trail])
        if len(recent):
            samples = origin+(destination-origin)*np.linspace(.25, 1., 6)[:, None]
            distance = np.linalg.norm(samples[:, None, :]-recent[None, :, :], axis=2).min(axis=1)
            penalty += 7*float(np.maximum(0., 1-distance/.8).mean())
            prior = next((point[[0, 2]] for point, _ in reversed(trail)
                          if np.linalg.norm(point[[0, 2]]-origin) > .8), None)
            if prior is not None:
                vector = destination-origin
                length = np.linalg.norm(vector)*np.linalg.norm(prior-origin)
                if length > .01:
                    penalty += 5*max(0., float(vector@(prior-origin))/length)
        return penalty

    def snapshot(self):
        return {'source': 'visual-pointcloud-crossing-memory', 'crossings': list(self.crossings),
                'portals': [{key: value.tolist() if isinstance(value, np.ndarray) else value
                             for key, value in item.items() if key != 'lastPosition'} for item in self.portals.values()],
                'trailPoints': len(self.trail), 'returnCooldownSeconds': 120, 'hardClosure': False}
