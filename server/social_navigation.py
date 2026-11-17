"""Deterministic social navigation over visual evidence; no scene coordinates."""
from __future__ import annotations

import math

import numpy as np
from visual_navigation import safe_travel
from visual_attention import person_attention
from visual_person_tracking import associate_person
from exploration_drive import ExplorationDrive
from intent_route import route_to_target


def attach_route(result, guidance, travel):
    """Batch only the known route; a new obstacle truncates it at the first leg."""
    path = guidance.get('relativePath')
    if path and travel >= guidance['distance']-.02:
        result['relativePath'] = path
        points = [np.zeros(2), *[np.array([math.sin(math.radians(b)), math.cos(math.radians(b))])*d for b, d in path]]
        length = sum(float(np.linalg.norm(b-a)) for a, b in zip(points, points[1:]))
        result.update(duration=max(1.2, length/.5), continuous=True, routeLength=round(length, 3))
    return result


def angle_delta(angle):
    return math.degrees(math.atan2(math.sin(math.radians(angle)), math.cos(math.radians(angle))))


def select_person(candidates, preferred):
    return next((person for person in candidates if person['id'] == preferred),
                min(candidates, key=lambda person: person['distance'], default=None))


def pose_people(observation, stamp):
    candidates = []
    for person in observation.get('people', []):
        point, confidence = person.get('opticalPosition'), person.get('confidence', 0)
        if confidence < .5 or not isinstance(point, list) or len(point) != 3:
            continue
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        matrix = np.asarray(observation.get('semanticPose', []), dtype=float)
        world = (matrix.reshape(4, 4)[:3, :3]@point+matrix.reshape(4, 4)[:3, 3]).tolist() if matrix.size == 16 else None
        candidates.append({'id': person['id'], 'distance': math.hypot(point[0], point[2]),
                           'worldPosition': world,
                           'bearing': math.degrees(math.atan2(point[0], point[2])),
                           'attention': person_attention(person, observation.get('faces', [])),
                           'confidence': confidence, 'observedAt': stamp})
    return candidates


def fused_people(observation, stamp, focal, width):
    candidates = []
    for item in observation.get('semanticFusion', {}).get('associations', []):
        box, depth = item.get('box'), item.get('depth')
        if item.get('label') not in ('person', 'human', 'face') or not isinstance(box, list) or len(box) != 4:
            continue
        if not isinstance(depth, (int, float)) or not math.isfinite(depth) or depth <= 0:
            continue
        confidence = item.get('confidence')
        if confidence is None:
            confidence = max((d.get('confidence', 0) for d in observation.get('objects', [])
                              if d.get('label') == item['label'] and d.get('box') == box), default=0)
        if confidence < .5:
            continue
        horizontal = ((box[0]+box[2])/2-width/2)/focal
        candidates.append({'id': item['id'], 'distance': depth*math.hypot(1, horizontal),
                           'bearing': math.degrees(math.atan(horizontal)),
                           'attention': person_attention({'box': box}, observation.get('faces', [])),
                           'worldPosition': item.get('worldPosition'),
                           'confidence': confidence, 'observedAt': stamp})
    return candidates


def person_observation(observation, now, preferred=None, previous=None):
    """A metric person observation is required; image occupancy is not distance."""
    stamp = observation.get('semanticObservedAt', observation.get('observedAt', observation.get('stereo', {}).get('observedAt', 0)))
    if not isinstance(stamp, (int, float)) or not 0 <= now-stamp <= 1.5:
        return None
    width, height = observation.get('width', 640), observation.get('height', 360)
    focal = height/(2*math.tan(math.radians(39)))
    candidates = pose_people(observation, stamp) or fused_people(observation, stamp, focal, width)
    person = associate_person(candidates, previous, stamp) if previous else select_person(candidates, preferred)
    if person is None and (previous is None or not candidates):
        person = image_person(observation, stamp, focal, width)
    if person is not None and locate_person_from_body(person, observation):
        return person
    headings = [observation.get(key) for key in ('viewHeading', 'semanticViewHeading')]
    if person is not None and all(type(value) in (int, float) and math.isfinite(value) for value in headings):
        person['bearing'] = angle_delta(person['bearing']+headings[0]-headings[1])
    return person


def locate_person_from_body(person, observation):
    """Use horizontal body-to-person range, not a pitched camera's Z depth."""
    pose = observation.get('stereo', {}).get('pose', {})
    body, point = pose.get('bodyPosition'), person.get('worldPosition')
    matrix = np.asarray(pose.get('cameraToWorld', []), dtype=float)
    if not isinstance(body, list) or not isinstance(point, list) or len(body) != 3 or len(point) != 3 or matrix.size != 16:
        return False
    if not np.isfinite([*body, *point, *matrix.ravel()]).all():
        return False
    delta = np.asarray(point)-body
    delta[1] = 0
    forward = matrix.reshape(4, 4)[:3, 2].copy()
    forward[1] = 0
    if np.linalg.norm(forward) < .2:
        return False
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., 1., 0.]))
    person.update(distance=float(np.linalg.norm(delta)), bearing=math.degrees(math.atan2(delta@right, delta@forward)))
    return True


def image_person(observation, stamp, focal, width):
    detections = [*observation.get('objects', []),
                  *[{**face, 'label': 'face'} for face in observation.get('faces', [])]]
    candidates = []
    for item in detections:
        box = item.get('box')
        if item.get('label') not in ('person', 'human', 'face') or item.get('confidence', 0) < .5:
            continue
        if not isinstance(box, list) or len(box) != 4 or not np.isfinite(box).all():
            continue
        bearing = math.degrees(math.atan(((box[0]+box[2])/2-width/2)/focal))
        candidates.append({'id': 'unlocalized-person', 'distance': None, 'bearing': bearing,
                           'confidence': item['confidence'], 'observedAt': stamp})
    return min(candidates, key=lambda person: abs(person['bearing']), default=None)


class SocialNavigator:
    """Keep a mission until a person is stably reachable for interaction."""
    def __init__(self):
        self.phase = 'searching'
        self.target = None
        self.confirmed_since = None
        self.confirmed_frames = 0
        self.last_person_stamp = 0.
        self.ready = False
        self.scanned = 0.
        self.scan_yaw = None
        self.scan_origin = None
        self.failed_actions = 0
        self.retreat = None
        self.detour = None
        self.recovery_turns = 0
        self.recovery_direction = None
        self.alignment_goal = None
        self.alignment_turns = 0
        self.social_finished_at = 0.
        self.social_wait_until = 0.
        self.survey_views = set()
        self.survey_complete = False
        self.minimum_travel = .15
        self.blocked_turn = False
        self.escape_bearing = None
        self.greeting_command = None
        self.search_region = None
        self.interrupt_person = None
        self.interrupt_started_at = None
        self.last_interrupt = 0.
        self.route_wait_since = None
        self.survey_finished_at = 0.
        self.person_route_goal = None
        self.reacquire_turns = 0
        self.recovery_wait_since = None
        self.recovery_looks = 0
        self.last_tracking_view = None
        self.look_attempts = {}
        self.drive = ExplorationDrive()
        self.aggressive = False

    def snapshot(self):
        return {'phase': self.phase, 'ready': self.ready, 'target': self.target,
                'confirmationFrames': self.confirmed_frames, 'scanDegrees': round(self.scanned, 1),
                'failedActions': self.failed_actions,
                'alignmentTurns': self.alignment_turns,
                'surveyCoverage': round(len(self.survey_views)/24, 3),
                'minimumTravel': round(self.minimum_travel, 3),
                'conditions': {'minDistance': .9, 'maxDistance': 1.7, 'maxBearing': 12,
                               'stableSeconds': .8, 'freshObservations': 3, 'otherFacingCamera': True}}

    def step(self, observation, navigation, memory, guidance, now, selected_person=None, managed_target=False):
        aggressive = navigation.get('visualAvoidance') is False
        self.aggressive = aggressive
        self.drive.observe_heading(observation)
        self.recovery_turns = 0
        self.recovery_direction = None
        self.recovery_wait_since = None
        self.recovery_looks = 0
        heading = observation.get('viewHeading')
        if type(heading) in (int, float) and math.isfinite(heading):
            self.last_tracking_view = heading
        if self.ready:
            return self.result('observe', 0, 0, 'social_ready', '社交条件已达成，等待交互')
        if aggressive:
            recovery = self.drive.recovery()
            if recovery:
                return self.result(**recovery)
        if (not aggressive and observation.get('semanticObservedAt', now) < self.survey_finished_at
                and not self.travel_ready(navigation, guidance)):
            return {'wait': True, 'reason': '等待环顾结束后的新语义帧，再选择未观察方向'}
        if self.blocked_turn:
            self.blocked_turn = False
            return self.turn(memory.scan_sign*45, 'inspect_blockage', '移动被拒绝，先迈步转向获取侧向视觉，再绕开失败方向')
        if self.escape_bearing is not None:
            bearing, self.escape_bearing = self.escape_bearing, None
            return self.turn(bearing, 'inspect_blockage', '转身受阻，先向相反方向迈步脱离，再重新选路')
        person = selected_person if managed_target else person_observation(observation, now, previous=self.target)
        if self.awaiting_person(person, observation, now):
            return {'wait': True, 'reason': '等待转向结束后的新人物视觉帧'}
        if (self.target is not None and self.phase in ('approach_person', 'face_person', 'retain_person', 'await_attention')
                and self.retreat is None and self.detour is None and now-self.target['observedAt'] < 2.5
                and (person is None or not managed_target and person['id'] != self.target['id'])):
            return self.result('observe', 0, 0, 'retain_person', '暂时失去同一人物，保持目标，等待新鲜视觉；不切回探索')
        retained = self.reacquire(person, observation, navigation, now)
        if retained is not None:
            return retained
        if person is not None and self.retreat is None and self.detour is None:
            return self.approach(person, navigation, memory, now)
        self.confirmed_since, self.confirmed_frames = None, 0
        if aggressive:
            return self.explore_aggressively(observation, navigation, memory, guidance, now)
        # A full person survey is a memory fact, not a motion prerequisite.
        # Keep missing views unknown while progressing on measured floor.
        if self.retreat is None and self.detour is None and self.travel_ready(navigation, guidance):
            result = self.follow_guidance(navigation, memory, guidance, now)
            if result is not None:
                return result
        was_surveying = not self.survey_complete
        survey = self.survey(memory, navigation.get('bodyBearing', 0)) if self.retreat is None and self.detour is None else None
        if survey is not None:
            return survey
        if was_surveying and self.survey_complete:
            memory.goal = None
            guidance = memory.guidance(now)
        return self.explore(navigation, memory, guidance, now)

    def explore_aggressively(self, observation, navigation, memory, guidance, now):
        committed = self.drive.committed(observation, memory, now)
        if committed is not None:
            return self.routed_exploration(committed, navigation, memory, now)
        exits = self.drive.exit_candidates(observation, memory)
        if not exits and self.travel_ready(navigation, guidance):
            heading = self.drive.view-guidance.get('bearing', 0)
            path = guidance.get('relativePath')
            bearing, distance = path[-1] if path else (guidance.get('bearing', 0), guidance.get('distance', 3.))
            target = self.drive.world_target(bearing, distance, memory)
            retrace = memory.exit_memory.travel_penalty(memory.position, target, now) if target is not None else 0.
            if retrace < 4 and not self.drive.blocked(heading, memory, now):
                result = self.follow_guidance(navigation, memory, guidance, now)
                if result is not None:
                    self.drive.requested_heading = heading
                    return result
        command = self.drive.plan(observation, navigation, memory, now)
        return self.routed_exploration(command, navigation, memory, now)

    def routed_exploration(self, command, navigation, memory, now):
        if command['action'] == 'forward' and navigation.get('tracking'):
            target = self.drive.commit_target
            if target is None:
                target = self.drive.world_target(command['bearing'], command['distance'], memory)
            route = route_to_target(memory, target, command['intent']) if target is not None else None
            if route is not None:
                result = self.follow_guidance(navigation, memory, route, now)
                if result is not None and result.get('action') == 'forward':
                    self.drive.requested_heading = self.drive.view-route['bearing']
                    result['reason'] = '保持探索目标，沿点云记忆中的完整通路连续前进'
                    return result
        return self.result(**command)

    def travel_ready(self, navigation, guidance):
        if not guidance.get('available') or guidance.get('goal') is None:
            return False
        distance = guidance.get('distance', 0.)
        travel = min(distance, safe_travel(navigation, guidance.get('bearing', 0), distance))
        # A short first grid leg must not hide a complete multi-metre route.
        length = attach_route({}, guidance, travel).get('routeLength', travel)
        return length >= .65

    def survey(self, memory, body_bearing=0):
        if memory.rotation is None:
            return None
        identity = memory.topology.region_at(memory.position)
        region = memory.topology.regions.get(identity)
        if region is not None:
            if identity != self.search_region:
                self.search_region, self.survey_complete = identity, False
                self.look_attempts.clear()
            self.survey_views = set(region['views'])
        elif self.survey_complete:
            return None
        self.update_scan(memory)
        if region is None:
            self.survey_views.update(memory.coverage.views_at(memory.position))
        self.survey_complete = len(self.survey_views) == 24
        if self.survey_complete:
            return None
        stamp = memory.last_stamp
        self.look_attempts = {key: seen for key, seen in self.look_attempts.items() if stamp-seen < 8.}
        bearing = memory.coverage.choose_view(self.survey_views, memory.rotation, body_bearing, self.look_attempts)
        bearing = 65 if bearing is None else bearing
        # Bearings share the current exposure frame. Head-only looks must be
        # reachable from the body, not incremental offsets from the last look.
        if abs(angle_delta(bearing-body_bearing)) <= 75:
            yaw = math.degrees(math.atan2(memory.rotation[0, 2], memory.rotation[2, 2]))
            self.look_attempts[round((yaw-bearing)/5) % 72] = stamp
            return self.result('look', bearing, 0, 'survey_surroundings', '转头观察尚未确认的方向，身体保持当前姿态')
        return self.turn(bearing, 'survey_surroundings',
                         '头颈范围外的未知方向使用迈步转身观察', body_bearing)

    def monitor(self, observation, arguments, now, selected_person=None, managed_target=False):
        if arguments.get('action') != 'forward' or now-self.last_interrupt < 2:
            return None
        person = selected_person if managed_target else person_observation(observation, now, previous=self.interrupt_person or self.target)
        if person is None or person['distance'] is None:
            return None
        candidate = self.interrupt_person
        if candidate is None or candidate['id'] != person['id']:
            self.interrupt_person = person
            self.interrupt_started_at = person['observedAt']
            return None
        self.interrupt_person = person
        if person['observedAt']-self.interrupt_started_at < .35:
            return None
        following = arguments.get('intent') in ('approach_person', 'join_group')
        target = arguments.get('targetPosition')
        point = person.get('worldPosition')
        moved = (isinstance(target, list) and isinstance(point, list)
                 and np.linalg.norm((np.asarray(target)-point)[[0, 2]]) > .65)
        if not following or person['distance'] < 1.05 or moved:
            self.last_interrupt, self.interrupt_person = now, None
            return '人物目标已移动，停步后更新完整路线' if following else '持续确认发现人物，结束出口路线并转入接近'
        return None

    def awaiting_person(self, person, observation, now):
        stamp = observation.get('semanticObservedAt', observation.get('observedAt', 0))
        return now < self.social_wait_until and (person is None or stamp < self.social_finished_at)

    def reacquire(self, person, observation, navigation, now):
        if person is not None or self.target is None or not 2.5 <= now-self.target['observedAt'] < 8:
            return None
        remembered = dict(self.target)
        if not locate_person_from_body(remembered, observation):
            return None
        bearing = remembered['bearing']
        if abs(angle_delta(bearing-navigation.get('bodyBearing', 0))) > 12 and self.reacquire_turns < 2:
            self.reacquire_turns += 1
            return self.turn(bearing, 'retain_person', '人物短暂遮挡，转向最后视觉位置重新寻找', navigation.get('bodyBearing', 0))
        return self.result('observe', 0, 0, 'retain_person', '注视人物最后视觉位置，等待重新确认；不盲目前进')

    def local_social(self, observation, navigation, memory, now):
        if self.ready:
            return None
        person = person_observation(observation, now, previous=self.target)
        if self.awaiting_person(person, observation, now):
            return {'wait': True, 'reason': '等待转向结束后的新人物视觉帧'}
        if person is not None and (person['distance'] is None or abs(person['bearing']) > 12
                                   or .9 <= person['distance'] <= 1.7):
            return self.approach(person, navigation, memory, now)
        return None

    def recover(self, observation, memory, now):
        stereo = observation.get('stereo', {})
        if self.ready or stereo.get('method') != 'stereo-sgbm-lr':
            return None
        stamp = stereo.get('observedAt', 0)
        fresh = isinstance(stamp, (int, float)) and 0 <= now-stamp <= 1
        ratio = stereo.get('validRatio')
        angles = (15, 30, 45, 60, 60, 60, 60, 30)
        if (not fresh or not isinstance(ratio, (int, float)) or not math.isfinite(ratio)
                or not 0 <= ratio <= 1 or self.recovery_turns >= len(angles)):
            return None
        if stereo.get('pose', {}).get('status') == 'lost':
            if self.recovery_wait_since is None:
                self.recovery_wait_since = now
            if now-self.recovery_wait_since < .8:
                return {'wait': True, 'reason': '短暂失锁，停稳等待视觉重定位；不立即转身'}
        look = self.recover_view(observation)
        if look is not None:
            return look
        if self.recovery_direction is None:
            self.recovery_direction = -memory.scan_sign
        bearing = self.recovery_direction*angles[self.recovery_turns]
        self.recovery_turns += 1
        return self.turn(bearing, 'recovering', '视觉定位丢失，扩大回看范围寻找已知特征；禁止平移，最多巡视一周')

    def recover_view(self, observation):
        """Return the head to a previously tracked visual exposure first."""
        view, body = observation.get('viewHeading'), observation.get('bodyHeading')
        if (self.last_tracking_view is None or self.recovery_looks >= 3
                or not all(type(value) in (int, float) and math.isfinite(value) for value in (view, body))):
            return None
        offset = (0, -18, 18)[self.recovery_looks]
        self.recovery_looks += 1
        bearing = angle_delta(view-self.last_tracking_view-offset)
        if abs(angle_delta(bearing-(view-body))) > 45:
            return None
        return self.result('look', bearing, 0, 'recovering_gaze', '保持身体位置，回看最后成功定位的视角及邻近纹理')

    def approach(self, person, navigation, memory, now):
        self.reacquire_turns = 0
        changed = self.target is not None and self.target['id'] != person['id']
        if changed:
            self.person_route_goal = None
        self.target = person
        distance, bearing = person['distance'], person['bearing']
        alignment = angle_delta(bearing-navigation.get('bodyBearing', 0))
        if distance is None:
            self.confirmed_since, self.confirmed_frames = None, 0
            if navigation.get('visualAvoidance') is False:
                self.drive.requested_heading = self.drive.view-bearing
                return self.result('forward', bearing, 1.5, 'approach_person',
                                   '发现人物，连续接近并更新距离；缺少测距不声明社交成功')
            if abs(bearing) > 12:
                return self.turn(bearing, 'face_person', '发现人物，先转向获取可信双目距离')
            return self.result('observe', 0, 0, 'social_observe', '人物已对齐，等待有效双目测距')
        near = .9 <= distance <= 1.7 and abs(alignment) <= 12
        stable = near and person.get('attention', {}).get('facingCamera') is True
        if not stable or changed:
            self.confirmed_since, self.confirmed_frames = None, 0
        if person['observedAt'] > self.last_person_stamp:
            self.last_person_stamp = person['observedAt']
            self.confirmed_frames = self.confirmed_frames+1 if stable else 0
            if stable and self.confirmed_since is None:
                self.confirmed_since = person['observedAt']
        if stable:
            self.ready = self.confirmed_frames >= 3 and self.confirmed_since is not None and person['observedAt']-self.confirmed_since >= .8
            return self.result('observe', 0, 0, 'social_ready' if self.ready else 'confirming_social',
                               '社交条件已达成，等待交互' if self.ready else '确认人物距离、面向与持续可见性')
        if near:
            return self.result('observe', 0, 0, 'await_attention', '已到人物面前；等待清晰面部朝向证据，暂不挥手')
        if distance <= 1.7 and abs(alignment) > 12:
            return self.turn(bearing, 'face_person', '调整面向，保持同一社交目标', navigation.get('bodyBearing', 0))
        if distance < .9:
            self.retreat = memory.retreat_goal(person, now)
            if self.retreat is not None:
                return self.explore(navigation, memory, memory.guidance(now), now)
            return self.result('observe', 0, 0, 'personal_space', '距离过近，等待可观测的退让路径')
        self.person_route_goal = memory.approach_goal(person, now, self.person_route_goal)
        if self.person_route_goal is not None:
            route = memory.route_to(self.person_route_goal, 'approach_person')
            result = self.follow_guidance(navigation, memory, route or {}, now)
            if result is not None:
                return result
        if abs(alignment) > 18:
            return self.turn(bearing, 'face_person', '调整面向，保持同一社交目标', navigation.get('bodyBearing', 0))
        travel = min(safe_travel(navigation, bearing), max(0, distance-1.3))
        if travel >= self.minimum_travel:
            return self.result('forward', bearing, travel, 'approach_person', '沿双目确认的净空接近人物')
        self.detour = memory.social_detour(person, now)
        if self.detour is not None:
            return self.explore(navigation, memory, guidance={}, now=now)
        return self.turn(memory.scan_sign*30, 'find_social_route', '直达路线受阻，观察侧向通路')

    def explore(self, navigation, memory, guidance, now):
        self.update_scan(memory)
        route = self.retreat if self.retreat is not None else self.detour
        if route is not None:
            intent = 'make_social_space' if self.retreat is not None else 'social_detour'
            guidance = memory.route_to(route, intent)
            if not guidance or guidance.get('distance', 0) < .2:
                self.retreat, self.detour = None, None
                return self.result('observe', 0, 0, 'searching', '局部路线结束，重新定位人物')
        result = self.follow_guidance(navigation, memory, guidance, now)
        if result is not None:
            return result
        if self.scanned >= 350:
            return self.result('observe', 0, 0, 'waiting_for_change', '已巡视一周，暂无可信通路；持续等待新视觉证据')
        axes = memory.horizontal_axes() if memory.rotation is not None else None
        blocked = bool(memory.rejected_motions) and (axes is None or memory.motion_blocked(
            memory.position[[0, 2]], (memory.position+axes[0]*.6)[[0, 2]]))
        if (not blocked and navigation['maxDistance'] >= max(.3, self.minimum_travel)
                and memory.stalls < 2 and self.failed_actions < 2):
            return self.result('forward', 0, navigation['maxDistance'], 'explore', '沿已确认净空连续前往未访问区域')
        bearing = memory.coverage.next_view(memory.position, memory.rotation) if memory.rotation is not None else memory.scan_sign*35
        if bearing is None:
            return self.result('observe', 0, 0, 'waiting_for_change', '本观察区方向已确认，等待新通路或人物；不重复环视')
        return self.turn(bearing, 'scan_unknown', '选择信息增益最大的未观察角度寻找出口或人物')

    def follow_guidance(self, navigation, memory, guidance, now):
        if guidance.get('goal'):
            if navigation.get('visualAvoidance') is not False and not navigation.get('valid', True) and not navigation.get('sectors'):
                return self.result('observe', 0, 0, 'inspect_route_floor', '地板当前不可见，保持路线并补看脚前地面；不反复左右转身')
            bearing = guidance['bearing']
            goal = tuple(guidance['goal'])
            if goal != self.alignment_goal:
                self.alignment_goal, self.alignment_turns = goal, 0
                self.route_wait_since = None
            travel = min(safe_travel(navigation, bearing, guidance.get('distance')), guidance.get('distance', .6))
            # Known, visible floor permits one turn-and-walk clip. Do not send
            # a separate alignment command followed by an unnecessary idle.
            length = attach_route({}, guidance, travel).get('routeLength', travel)
            if length >= self.minimum_travel:
                self.route_wait_since = None
                self.alignment_turns = 0
                return attach_route(self.result('forward', bearing, travel, guidance['intent'],
                                    '转向与完整路线连续播放，到达出口后再选路'), guidance, travel)
            if abs(angle_delta(bearing-navigation.get('bodyBearing', 0))) > 18:
                if self.alignment_turns >= 4:
                    memory.rejected[goal] = now+30
                    memory.reject_goal(now)
                    self.retreat, self.detour = None, None
                    return self.result('observe', 0, 0, 'replan_frontier', '连续对齐未取得进展，暂时排除该区域并重新规划')
                self.alignment_turns += 1
                return self.turn(bearing, guidance['intent'], '沿持续区域目标调整行进方向', navigation.get('bodyBearing', 0))
            if self.route_wait_since is None:
                self.route_wait_since = now
            if now-self.route_wait_since < 2.5:
                return self.result('observe', 0, 0, 'inspect_route_floor', '保持出口目标，等待头部视野中的地板与障碍证据稳定')
            self.route_wait_since = None
            memory.reject_goal(now)
            self.retreat, self.detour = None, None
        return None

    def update_scan(self, memory):
        if memory.rotation is None:
            return
        forward = memory.rotation[:, 2]
        yaw = math.atan2(forward[0], forward[2])
        if self.scan_origin is None or np.linalg.norm((memory.position-self.scan_origin)[[0, 2]]) > .4:
            self.scanned = 0
            self.scan_origin = memory.position.copy()
        if self.scan_yaw is not None:
            self.scanned += abs(math.degrees(math.atan2(math.sin(yaw-self.scan_yaw), math.cos(yaw-self.scan_yaw))))
        self.scan_yaw = yaw

    def feedback(self, action, status, memory, now, detail='', motion=None):
        if status == 'interrupted' and detail.startswith(('人物目标已移动', '持续确认发现人物')):
            self.failed_actions = 0
            self.person_route_goal = None
            return
        if status == 'completed':
            self.failed_actions = 0
            if (action.startswith('turn_') or action == 'look') and self.phase == 'survey_surroundings':
                self.survey_finished_at = now
            if self.target is not None and self.phase in ('face_person', 'approach_person'):
                self.social_finished_at, self.social_wait_until = now, now+2
        elif status in ('failed', 'interrupted'):
            self.failed_actions += 1
            if action in ('forward', 'turn_left', 'turn_right'):
                if self.detour is not None:
                    memory.rejected[self.detour] = now+30
                self.retreat, self.detour = None, None
                memory.reject_goal(now)
                if action.startswith('turn_'):
                    memory.scan_sign *= -1
                    if status == 'failed':
                        self.escape_bearing = 35 if action == 'turn_left' else -35
            if (status == 'failed' and action == 'forward'
                    and detail.startswith(('预设根运动无法安全通过', '视觉避障：', '播放碰撞：', '移动死锁：'))):
                memory.reject_motion(motion, now)
                self.blocked_turn = True
            if self.aggressive and status == 'failed' and action in ('forward', 'turn_left', 'turn_right') and detail.startswith(
                    ('预设根运动无法安全通过', '播放碰撞：', '移动死锁：', '移动距离不足一个安全落脚步')):
                self.drive.failure(memory, now, turning=action.startswith('turn_'))
                self.blocked_turn, self.escape_bearing = False, None
        if self.failed_actions >= 3:
            self.phase = 'recovering'

    def turn(self, bearing, intent, reason, body_bearing=0):
        alignment = angle_delta(bearing-body_bearing)
        limited = math.copysign(min(70, max(5, abs(alignment))), alignment)
        return self.result('turn_right' if alignment > 0 else 'turn_left',
                           angle_delta(body_bearing+limited), 0, intent, reason)

    def result(self, action, bearing, distance, intent, reason):
        self.phase = intent
        result = {'action': action, 'relativeBearing': round(bearing, 2), 'distance': round(distance, 3),
                'duration': max(1.2, distance/.5) if action == 'forward' else .6 if action == 'observe' else 1.2,
                'continuous': action == 'forward' and distance >= 1.2, 'intent': intent,
                'reason': reason, 'mission': self.snapshot()}
        if intent == 'approach_person' and self.target:
            result.update(targetId=self.target['id'], targetPosition=self.target.get('worldPosition'))
        if intent in ('waiting_for_change', 'inspect_route_floor', 'await_attention', 'retain_person'):
            result['wait'] = True
        return result
