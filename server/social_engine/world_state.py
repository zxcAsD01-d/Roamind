"""Bounded, timestamped visual entities. These tracks never authorize contact."""
from collections import deque
import math

import numpy as np

from social_navigation import (pose_people, fused_people, image_person,
                               locate_person_from_body, angle_delta)


def point3(value):
    if (not isinstance(value, list) or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        return None
    return np.asarray(value, dtype=float)


def candidates_from_vision(observation, now):
    stamp = observation.get('semanticObservedAt', observation.get('observedAt', observation.get('stereo', {}).get('observedAt', 0)))
    if (observation.get('semanticReady') is False or type(stamp) not in (int, float)
            or not math.isfinite(stamp) or not 0 <= now-stamp <= 1.5):
        return stamp, []
    width, height = observation.get('width', 640), observation.get('height', 360)
    focal = height/(2*math.tan(math.radians(39)))
    people = pose_people(observation, stamp) or fused_people(observation, stamp, focal, width)
    if not people:
        person = image_person(observation, stamp, focal, width)
        people = [person] if person else []
    for person in people:
        if locate_person_from_body(person, observation):
            continue
        headings = [observation.get(k) for k in ('viewHeading', 'semanticViewHeading')]
        if all(type(v) in (int, float) and math.isfinite(v) for v in headings):
            person['bearing'] = angle_delta(person['bearing']+headings[0]-headings[1])
    return stamp, people


class WorldState:
    def __init__(self):
        self.tracks = {}
        self.observed_at = 0.
        self.self_state = {'position': None, 'observedAt': 0., 'localized': False}
        self.revision = 0
        self.view = {}

    def update(self, observation, now):
        self.view = observation
        self._update_self(observation, now)
        self.tracks = {key: row for key, row in self.tracks.items() if now-row['observedAt'] < 12}
        if observation.get('semanticReady') is False:
            return
        stamp = observation.get('semanticObservedAt', observation.get('observedAt', observation.get('stereo', {}).get('observedAt', 0)))
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or stamp <= self.observed_at or not 0 <= now-stamp <= 1.5):
            return
        _, candidates = candidates_from_vision(observation, now)
        self.observed_at = stamp
        unmatched = set(self.tracks)
        for candidate in sorted(candidates, key=lambda p: p['confidence'], reverse=True)[:64]:
            key = self._associate(candidate, unmatched, stamp)
            if key is None:
                key = str(candidate['id'])
                if key in self.tracks:
                    continue  # A reused detector ID must not teleport a track.
            previous = self.tracks.get(key)
            self.tracks[key] = self._integrate(key, candidate, previous, stamp)
            unmatched.discard(key)
        self.tracks = dict(sorted(self.tracks.items(), key=lambda item: item[1]['observedAt'])[-64:])
        self.revision += 1

    def _update_self(self, observation, now):
        stereo = observation.get('stereo', {})
        pose, stamp = stereo.get('pose', {}), stereo.get('observedAt', 0)
        matrix = np.asarray(pose.get('cameraToWorld', []), dtype=float)
        localized = (stereo.get('mapping') is True and matrix.size == 16
                     and np.isfinite(matrix).all() and 0 <= now-stamp <= 1.5)
        position = point3(pose.get('bodyPosition'))
        if localized and position is None:
            position = matrix.reshape(4, 4)[:3, 3]
        self.self_state = {'position': position.tolist() if localized and position is not None else None,
                           'observedAt': stamp, 'localized': bool(localized), 'frame': 'visual-odometry'}

    def _associate(self, candidate, unmatched, stamp):
        direct = next((key for key in unmatched if candidate['id'] in
                       (key, self.tracks[key]['observationId'])), None)
        point = point3(candidate.get('worldPosition'))
        if direct is not None:
            row = self.tracks[direct]
            origin = point3(row.get('worldPosition'))
            if (point is None or origin is None
                    or np.linalg.norm((point-origin)[[0, 2]]) <= .35+1.6*(stamp-row['observedAt'])):
                return direct
            return None
        if point is None:
            return None
        plausible = []
        for key in unmatched:
            row = self.tracks[key]
            origin = point3(row.get('worldPosition'))
            dt = stamp-row['observedAt']
            if origin is None or not 0 < dt <= 2.5:
                continue
            prediction = origin[[0, 2]]+np.asarray(row['velocity'])*dt
            if np.linalg.norm((point-origin)[[0, 2]]) <= .35+1.6*dt:
                plausible.append((float(np.linalg.norm(point[[0, 2]]-prediction)), key))
        plausible.sort()
        if not plausible or len(plausible) > 1 and plausible[1][0]-plausible[0][0] < .3:
            return None
        return plausible[0][1]

    def _integrate(self, key, candidate, previous, stamp):
        velocity = np.zeros(2)
        point = point3(candidate.get('worldPosition'))
        history = deque(previous['history'] if previous else [], maxlen=24)
        if previous and point is not None:
            origin = point3(previous.get('worldPosition'))
            if origin is not None and stamp > previous['observedAt']:
                measured = (point-origin)[[0, 2]]/(stamp-previous['observedAt'])
                measured *= min(1., 1.6/max(1e-6, np.linalg.norm(measured)))
                velocity = .5*np.asarray(previous['velocity'])+.5*measured
        history.append({'at': stamp, 'position': point.tolist() if point is not None else None})
        return {**candidate, 'id': key, 'observationId': candidate['id'],
                'velocity': velocity.tolist(), 'navigationVelocity': velocity.tolist(),
                'history': list(history), 'firstSeen': previous['firstSeen'] if previous else stamp,
                'samples': (previous['samples'] if previous else 0)+1,
                'viewHeadingAtObservation': self.view.get('viewHeading'),
                'source': 'visual-person-navigation-only'}

    def visible(self, now):
        return [self._relative(row) for row in self.tracks.values() if row['observedAt'] == self.observed_at
                and 0 <= now-row['observedAt'] <= 1.5]

    def _relative(self, row):
        person = dict(row)
        if locate_person_from_body(person, self.view):
            return person
        previous, current = row.get('viewHeadingAtObservation'), self.view.get('viewHeading')
        if all(type(v) in (int, float) and math.isfinite(v) for v in (previous, current)):
            person['bearing'] = angle_delta(row['bearing']+current-previous)
        return person

    def snapshot(self, now):
        visible = {row['id'] for row in self.visible(now)}
        return {'frame': 'visual-odometry', 'observedAt': self.observed_at,
                'self': self.self_state, 'revision': self.revision,
                'entities': [{**{key: value for key, value in self._relative(row).items() if key != 'history'},
                              'history': row['history'][-4:], 'visible': row['id'] in visible,
                              'ageSeconds': round(max(0, now-row['observedAt']), 2)}
                             for row in self.tracks.values()]}
