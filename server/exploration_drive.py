"""Persistent exploratory motion; vision ranks options but cannot veto travel."""
import math

import numpy as np


def wrap(angle):
    return math.degrees(math.atan2(math.sin(math.radians(angle)), math.cos(math.radians(angle))))


class ExplorationDrive:
    def __init__(self):
        self.heading = 0.
        self.view = 0.
        self.requested_heading = 0.
        self.back_heading = None
        self.escape_distance = 0.
        self.rejected = []
        self.commit_heading = None
        self.commit_until = 0.
        self.commit_target = None
        self.attempted_targets = []
        self.last_intent = ''

    def observe_heading(self, observation):
        view = observation.get('viewHeading', 0.)
        body = observation.get('bodyHeading', view)
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (view, body)):
            raise ValueError('探索朝向必须是有限数值')
        self.view, self.heading = view, body

    def blocked(self, heading, memory, now):
        self.rejected = [item for item in self.rejected if now < item['until']]
        for item in self.rejected:
            if item['origin'] is not None and memory.position is not None:
                if np.linalg.norm((memory.position-item['origin'])[[0, 2]]) > 2.5:
                    continue
            if abs(wrap(heading-item['heading'])) < 40:
                return True
        return False

    def failure(self, memory, now, turning=False):
        retry_escape = self.last_intent in ('escape_deadlock', 'leave_deadlock')
        origin = None if memory.position is None else memory.position.copy()
        self.rejected.append({'heading': self.requested_heading, 'origin': origin, 'until': now+60})
        self.rejected = self.rejected[-16:]
        if self.commit_target is not None:
            self.attempted_targets.append((self.commit_target.copy(), now+90))
            self.attempted_targets = self.attempted_targets[-32:]
        self.commit_heading, self.commit_until = None, 0.
        self.commit_target = None
        self.back_heading = wrap(self.heading+(-90*memory.scan_sign if turning else -180*memory.scan_sign))
        if retry_escape:
            self.back_heading = self.alternate_escape(memory, now)
        self.escape_distance = 1.5

    def alternate_escape(self, memory, now):
        candidates = []
        for angle in (-90*memory.scan_sign, 90*memory.scan_sign, -60, 60, -120, 120, 180):
            heading = wrap(self.heading+angle)
            if self.blocked(heading, memory, now):
                continue
            target = self.world_target(wrap(self.view-heading), 1.5, memory)
            penalty = memory.exit_memory.travel_penalty(memory.position, target, now) if target is not None else 0.
            candidates.append((penalty+.01*abs(abs(angle)-90), heading))
        return min(candidates, key=lambda item: item[0])[1] if candidates else wrap(self.heading+90*memory.scan_sign)

    def recovery(self):
        if self.back_heading is None:
            return None
        bearing = wrap(self.view-self.back_heading)
        alignment = wrap(self.heading-self.back_heading)
        if abs(alignment) > 12:
            return self.command('turn_right' if alignment > 0 else 'turn_left',
                                bearing, 0, 'escape_deadlock',
                                '移动受阻，持续回头；完成转身后离开失败方向')
        self.back_heading = None
        return self.command('forward', bearing, self.escape_distance, 'leave_deadlock',
                            '已回头，连续离开卡住位置后重新选择出口')

    def command(self, action, bearing, distance, intent, reason):
        self.requested_heading = wrap(self.view-bearing)
        self.last_intent = intent
        return {'action': action, 'bearing': bearing, 'distance': distance,
                'intent': intent, 'reason': reason}

    def exit_candidates(self, observation, memory):
        candidates = []
        stereo = observation.get('stereo', {})
        now = observation.get('observedAt', memory.last_stamp)
        if stereo.get('mapping') and memory.position is not None and memory.rotation is not None:
            candidates.extend(self.cloud_candidates(memory, now))
            current = memory.topology.region_at(memory.position)
            exits = [(door['position'], 8.) for door in memory.topology.doors.values()
                     if current not in door['visitedFrom'] and door.get('state') != 'blocked']
            exits += [(item['position'], 7.) for item in memory.topology.passages
                      if current in item['regions'] and item.get('traversable')]
            # A free-to-unknown frontier is a search goal, not a measured
            # doorway. Let the route planner reach it around furniture.
            for position, score in exits:
                delta = np.asarray(position)-memory.position
                distance = float(np.linalg.norm(delta[[0, 2]]))
                if distance > .35:
                    target = memory.position+delta*(distance+1.2)/distance
                    score -= memory.exit_memory.travel_penalty(memory.position, target, now)
                    candidates.append((score-.1*distance, memory.bearing_to(delta), min(3., distance+1.2)))
        for item in observation.get('objects', []):
            box = item.get('box')
            if (item.get('label') not in ('door', 'doorway', 'opening') or item.get('confidence', 0) < .4
                    or not isinstance(box, list) or len(box) != 4 or not np.isfinite(box).all()):
                continue
            focal = observation.get('height', 360)/(2*math.tan(math.radians(39)))
            bearing = math.degrees(math.atan(((box[0]+box[2])/2-observation.get('width', 640)/2)/focal))
            target = self.world_target(bearing, 3., memory) if stereo.get('mapping') else None
            penalty = memory.exit_memory.travel_penalty(memory.position, target, now) if target is not None else 0.
            candidates.append((4.+5*item['confidence']-penalty, bearing, 3.))
        return candidates

    def cloud_candidates(self, memory, now):
        candidates = []
        blocked = {reference for item in memory.topology.connections.items.values() if item['state'] == 'blocked'
                   for reference in item['_references']}
        for portal in memory.exit_memory.portals.values():
            if f"portal:{portal['id']}" in blocked:
                continue
            delta = portal['point']-memory.position
            distance = float(np.linalg.norm(delta[[0, 2]]))
            if distance > 9 or now-portal['lastSeen'] > 1800:
                continue
            side = float(delta[[0, 2]]@portal['normal'])
            # Aim beyond the measured opening, rather than stopping on its plane.
            target = portal['point'].copy()
            target[[0, 2]] += portal['normal']*(1.3 if side >= 0 else -1.3)
            vector = target-memory.position
            length = float(np.linalg.norm(vector[[0, 2]]))
            score = 3.+9*portal['confidence']-.12*distance
            score -= memory.exit_memory.travel_penalty(memory.position, target, now)
            candidates.append((score, memory.bearing_to(vector), min(3., length)))
        return candidates

    def world_target(self, bearing, distance, memory):
        if memory.position is None or memory.rotation is None:
            return None
        axes = memory.horizontal_axes()
        if axes is None:
            return None
        angle = math.radians(bearing)
        return memory.position+(axes[0]*math.cos(angle)+axes[1]*math.sin(angle))*distance

    def attempt_penalty(self, target, now):
        self.attempted_targets = [(point, until) for point, until in self.attempted_targets if now < until]
        if target is None:
            return 0.
        return max((12*max(0., 1-np.linalg.norm((target-point)[[0, 2]])/1.5)
                    for point, _ in self.attempted_targets), default=0.)

    def committed(self, observation, memory, now):
        if self.commit_target is None or now >= self.commit_until or not observation.get('stereo', {}).get('mapping'):
            return None
        delta = self.commit_target-memory.position
        distance = float(np.linalg.norm(delta[[0, 2]]))
        if distance < .45:
            self.commit_target = None
            return None
        bearing = memory.bearing_to(delta)
        if self.blocked(wrap(self.view-bearing), memory, now):
            self.commit_target = None
            return None
        return self.command('forward', bearing, min(3., distance), 'seek_possible_exit',
                            '持续完成同一出口的穿越目标，避免门口重复选门或立即折返')

    def plan(self, observation, navigation, memory, now):
        committed = self.committed(observation, memory, now)
        if committed is not None:
            return committed
        exits = self.exit_candidates(observation, memory)
        candidates = [(score-.015*abs(wrap(bearing-navigation['bodyBearing'])), bearing, distance, 'seek_possible_exit')
                      for score, bearing, distance in exits]
        if self.commit_heading is not None and now < self.commit_until:
            candidates.append((6., wrap(self.view-self.commit_heading), 3., 'explore'))
        # Missing, stale or short depth is an information gap, never a stop sign.
        forward = navigation['bodyBearing']
        candidates.append((2., forward, 3., 'explore'))
        for sector in navigation.get('sectors', []):
            bearing = sector['angle']
            score = 2.+.2*min(4., sector['clearance'])-.02*abs(wrap(bearing-forward))
            candidates.append((score, bearing, 3., 'explore'))
        candidates += [(0., wrap(forward+angle), 2., 'explore') for angle in (-60, 60, -120, 120, 180)]
        if observation.get('stereo', {}).get('mapping'):
            candidates = [(score-self.attempt_penalty(target, now)-(memory.exit_memory.travel_penalty(memory.position, target, now)
                                  if intent == 'explore' and target is not None else 0.), bearing, distance, intent)
                          for score, bearing, distance, intent in candidates
                          for target in [self.world_target(bearing, distance, memory)]]
        viable = [item for item in candidates if not self.blocked(wrap(self.view-item[1]), memory, now)]
        if not viable:
            self.failure(memory, now, turning=True)
            return self.recovery()
        _, bearing, distance, intent = max(viable, key=lambda item: item[0])
        self.commit_heading, self.commit_until = wrap(self.view-bearing), now+10
        if intent == 'seek_possible_exit' and observation.get('stereo', {}).get('mapping'):
            self.commit_target = self.world_target(bearing, distance, memory)
            self.commit_until = now+30
        return self.command('forward', bearing, distance, intent,
                            '优先疑似出口，未知深度继续米级探索；实际受阻后回头换路')
