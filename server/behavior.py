"""Vision-only scripted social navigation with diagnostic behavior feedback.

The policy consumes an ego-camera observation only: pixels reduced to a scene
signature, detector boxes, stereo evidence and visual odometry. It builds its
own memory and does not read room lists, simulation target coordinates,
navigation meshes, audio, or simulation geometry.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from visual_navigation import exploratory_navigation, body_bearing, safe_travel
from exploration_memory import ExplorationMemory
from social_navigation import SocialNavigator, attach_route, angle_delta
from social_engine import SocialNavigationEngine


FEATURES = ("novelty", "freeSpace", "visualSalience", "leftBias", "rightBias",
            "personVisible", "personCenter", "explorationNeed")
ACTIONS = ("forward", "turn_left", "turn_right", "observe", "look")


@dataclass
class VisualState:
    weights: np.ndarray = field(default_factory=lambda: np.array(
        [1.2, 1.0, .8, .15, .15, 1.6, .45, .9], dtype=np.float64))
    baseline: float = 0.
    previous_signature: np.ndarray | None = None
    pending: dict[str, dict] = field(default_factory=dict)
    decisions: int = 0
    feedback: int = 0
    steps_since_person: int = 0
    memory: ExplorationMemory = field(default_factory=ExplorationMemory)
    last_observation: float = 0.
    navigator: SocialNavigator = field(default_factory=SocialNavigator)
    engine: SocialNavigationEngine = field(default_factory=SocialNavigationEngine)
    last_access: float = 0.


class BehaviorPolicy:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.states: dict[str, VisualState] = {}
        self.learning_rate = .08
        self.temperature = .2
        self.signature_limit = 2000

    def _state(self, client: str) -> VisualState:
        now = time.monotonic()
        if len(self.states) >= 128 and client not in self.states:
            stale = sorted(self.states.items(), key=lambda item: item[1].last_access)
            for key, _ in stale[:32]:
                self.states.pop(key, None)
        state = self.states.get(client)
        if state is None:
            state = VisualState()
            self.states[client] = state
        state.last_access = now
        return state

    def configure(self, args: dict) -> dict:
        with self.lock:
            if "learningRate" in args:
                value = float(args["learningRate"])
                if not math.isfinite(value) or not 0 < value <= .5:
                    raise ValueError("learningRate 必须在 0–0.5")
                self.learning_rate = value
            if "temperature" in args:
                value = float(args["temperature"])
                if not math.isfinite(value) or not .01 <= value <= 2:
                    raise ValueError("temperature 必须在 0.01–2")
                self.temperature = value
            return self.config()

    def config(self) -> dict:
        return {"learningRate": self.learning_rate, "temperature": self.temperature,
                "features": list(FEATURES), "actions": list(ACTIONS), "input": "ego-vision-only",
                'navigation': 'persistent-aggressive-exploration', 'unknownDepth': 'advance',
                'visualAvoidance': False, 'collisionResponse': 'turn-back-and-replan',
                'socialGoal': 'find-people-through-visual-exploration', 'controller': 'hierarchical-social-navigation',
                'learningControlsMovement': False}

    def reset(self, client: str | None = None) -> dict:
        with self.lock:
            if client is None:
                self.states.clear()
            else:
                self.states.pop(client, None)
            return {"reset": client or "all"}

    @staticmethod
    def _number(value: object, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 必须是有限数值") from exc
        if not math.isfinite(result):
            raise ValueError(f"{label} 必须是有限数值")
        return result

    def _observation(self, raw: object) -> tuple[np.ndarray, list[dict], int, int]:
        if not isinstance(raw, dict):
            raise ValueError("vision 必须是第一人称视觉观测对象")
        signature = raw.get("signature")
        if not isinstance(signature, list) or not 8 <= len(signature) <= self.signature_limit:
            raise ValueError("vision.signature 必须是 8–2000 个像素特征")
        vector = np.asarray([self._number(value, "vision.signature") for value in signature], dtype=np.float32)
        objects = raw.get("objects", [])
        faces = raw.get("faces", [])
        if not isinstance(objects, list) or not isinstance(faces, list):
            raise ValueError("vision.objects/faces 必须是数组")
        detections = [item for item in objects if isinstance(item, dict)]
        detections.extend({**item, "label": item.get("label", "face")}
                          for item in faces if isinstance(item, dict))
        width = int(raw.get("width", 640))
        height = int(raw.get("height", 360))
        if width < 1 or height < 1:
            raise ValueError("vision 尺寸无效")
        return vector, detections, width, height

    def _visual_features(self, state: VisualState, signature: np.ndarray,
                         detections: list[dict], width: int, height: int) -> tuple[float, float, float, float, float]:
        novelty = 1. if state.previous_signature is None or state.previous_signature.shape != signature.shape else float(np.mean(np.abs(signature - state.previous_signature)))
        novelty = min(1., novelty * 4.)
        state.previous_signature = signature
        central = left = right = salience = 0.
        for item in detections:
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x0, y0, x1, y1 = [self._number(value, "vision.box") for value in box]
            center_x = ((x0 + x1) / 2) / width
            bottom = y1 / height
            area = max(0., min(1., (x1 - x0) * (y1 - y0) / (width * height)))
            confidence = min(1., max(0., self._number(item.get("confidence", .5), "confidence")))
            salience = max(salience, area * confidence * 4.)
            if bottom < .45:
                continue
            occupancy = min(1., area * 8.)
            if .3 <= center_x <= .7:
                central = max(central, occupancy)
            elif center_x < .5:
                left = max(left, occupancy)
            else:
                right = max(right, occupancy)
        return novelty, 1. - central, min(1., salience), left, right

    @staticmethod
    def _person_signal(detections: list[dict], width: int, height: int) -> tuple[float, float, float]:
        """Return confidence, signed image center and visual occupancy for a person/face."""
        best = (0., 0., 0.)
        for item in detections:
            label = str(item.get('label', '')).lower()
            if label not in {'person', 'face', 'human', 'people'}:
                continue
            box = item.get('box')
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(value) for value in box]
                confidence = min(1., max(0., float(item.get('confidence', .5))))
            except (TypeError, ValueError):
                continue
            area = max(0., min(1., (x1-x0)*(y1-y0)/(width*height)))
            signal = confidence * min(1., area*8.)
            center = max(-1., min(1., (((x0+x1)/2)/width-.5)*2.))
            if signal > best[0]:
                best = (signal, center, area)
        return best

    @staticmethod
    def _exploration_need(state: VisualState, novelty: float, person_visible: float) -> float:
        if person_visible > .05:
            state.steps_since_person = 0
        else:
            state.steps_since_person = min(32, state.steps_since_person + 1)
        repetition = max(0., 1.-novelty)
        return min(1., .08*state.steps_since_person + .35*repetition)

    def _action_vector(self, action: str, features: tuple[float, ...]) -> np.ndarray:
        novelty, free_space, salience, left, right, person, center, explore = features
        seek_left = person * max(0., -center)
        seek_right = person * max(0., center)
        if action == "forward":
            return np.array([novelty, free_space, salience * .4, 1. - left, 1. - right,
                             person*(1.-abs(center)), 0., explore])
        if action == "turn_left":
            return np.array([.35 + .5 * (1. - novelty), 1. - left, salience, 1. - left, right,
                             seek_left, -seek_left, explore*.6])
        if action == "turn_right":
            return np.array([.35 + .5 * (1. - novelty), 1. - right, salience, left, 1. - right,
                             seek_right, seek_right, explore*.6])
        return np.array([0., free_space, salience, 1. - novelty, 1. - novelty,
                         person*.8, 0., explore])

    def _prepare_vision(self, state: VisualState, raw: dict) -> tuple:
        signature, detections, width, height = self._observation(raw)
        observation = raw
        if observation.get('semanticReady') is False:
            observation = {**raw, 'objects': [], 'people': [], 'faces': [],
                           'semanticFusion': {}, 'semanticObservedAt': 0}
            detections = []
        features = self._visual_features(state, signature, detections, width, height)
        person = self._person_signal(detections, width, height)
        need = self._exploration_need(state, features[0], person[0])
        return observation, (*features, person[0], person[1], need), person

    def _observe_state(self, state: VisualState, observation: dict, now: float) -> None:
        stereo = observation.get('stereo', {})
        if not isinstance(stereo, dict):
            return
        stamp = stereo.get('observedAt', 0)
        if stereo.get('pose', {}).get('status') == 'initialized' and stamp > state.memory.last_stamp:
            state.memory, state.navigator = ExplorationMemory(), SocialNavigator()
            state.engine = SocialNavigationEngine()
        state.memory.update(stereo, [], now)
        state.memory.observe_semantics(observation, now)
        state.engine.observe(observation, now)
        heading = observation.get('viewHeading')
        if stereo.get('mapping') and type(heading) in (int, float) and math.isfinite(heading):
            state.navigator.last_tracking_view = heading

    def decide(self, args: dict) -> dict:
        client = args.get('clientId')
        if not isinstance(client, str) or not 1 <= len(client) <= 100:
            raise ValueError('clientId 无效')
        with self.lock:
            state = self._state(client)
            observation, features, person = self._prepare_vision(state, args.get('vision'))
            stamp = max(observation.get('observedAt', 0), observation.get('stereo', {}).get('observedAt', 0))
            if stamp <= state.last_observation:
                return {'wait': True, 'reason': '等待未使用的新视觉帧', 'features': dict(zip(FEATURES, features))}
            state.last_observation = stamp
            now = time.time()
            self._observe_state(state, observation, now)
            navigation = exploratory_navigation(observation.get('stereo'))
            navigation['bodyBearing'] = body_bearing(observation)
            guidance = state.memory.guidance(now) if navigation.get('tracking') else {}
            features = (features[0], min(1., navigation['front']/2.5), *features[2:])
            scripted = state.engine.step(observation, navigation, state.memory, guidance, state.navigator, now,
                                         social_enabled=args.get('exploreOnly') is not True)
            if scripted.get('wait'):
                return scripted
            decision_id = self._record_decision(state, scripted, features, now)
            return self._decision_result(state, scripted, decision_id, navigation, guidance, features, person)

    def _record_decision(self, state: VisualState, scripted: dict, features: tuple, now: float) -> str:
        action = scripted['action']
        if action in ('turn_left', 'turn_right'):
            state.memory.scan_sign = -1 if action == 'turn_left' else 1
        state.memory.record_action(action)
        motion = None
        axes = state.memory.horizontal_axes() if state.memory.rotation is not None else None
        if action == 'forward' and axes is not None:
            angle = math.radians(scripted['relativeBearing'])
            direction = axes[0]*math.cos(angle)+axes[1]*math.sin(angle)
            origin = state.memory.position[[0, 2]].copy()
            motion = (origin, origin+direction[[0, 2]]*scripted['distance'])
        decision_id = uuid.uuid4().hex
        state.pending[decision_id] = {'action': action, 'features': self._action_vector(action, features).tolist(),
                                      'created': now, 'motion': motion}
        state.decisions += 1
        if len(state.pending) > 16:
            del state.pending[next(iter(state.pending))]
        return decision_id

    @staticmethod
    def _decision_result(state: VisualState, scripted: dict, decision_id: str, navigation: dict,
                         guidance: dict, features: tuple, person: tuple) -> dict:
        return {'decisionId': decision_id, **scripted, 'navigation': navigation,
                'social': {'personVisible': round(person[0], 4), 'personCenter': round(person[1], 4),
                           'personArea': round(person[2], 4)},
                'exploration': {'need': round(features[-1], 4), 'stepsSincePerson': state.steps_since_person,
                                'guidance': guidance, 'memory': state.memory.snapshot()},
                'features': dict(zip(FEATURES, [round(v, 4) for v in features])),
                'policy': {'weights': [round(float(v), 4) for v in state.weights],
                           'decisions': state.decisions, 'feedback': state.feedback}}

    def feedback_for(self, client: str, decision_id: str, status: str, detail: str = "", stale_ok=False) -> dict:
        with self.lock:
            state = self._state(client)
            pending = state.pending.pop(decision_id, None)
            if pending is None:
                if stale_ok:
                    return {'decisionId': decision_id, 'ignored': True, 'reason': '任务已反馈或探索记忆已重置'}
                raise ValueError("视觉行为决策不存在或已反馈")
            state.navigator.feedback(pending['action'], status, state.memory, time.time(),
                                     detail, pending.get('motion'))
            state.engine.feedback(pending['action'], status, detail, time.time())
            reward = .45 if status == "completed" else -.8 if status == "failed" else -.1
            if "碰撞" in detail or "阻止" in detail:
                reward -= .6
            advantage = reward - state.baseline
            state.baseline = .9 * state.baseline + .1 * reward
            state.weights += self.learning_rate * advantage * np.asarray(pending["features"], dtype=np.float64)
            state.weights = np.clip(state.weights, -4., 4.)
            state.feedback += 1
            return {"decisionId": decision_id, "reward": round(reward, 4),
                    "baseline": round(state.baseline, 4),
                    "weights": [round(float(v), 4) for v in state.weights]}

    def snapshot(self, client: str) -> dict:
        with self.lock:
            state = self._state(client)
            state.memory.maintain(time.time())
            return {"clientId": client, "config": self.config(),
                    "weights": dict(zip(FEATURES, state.weights.round(4).tolist())),
                    "baseline": round(state.baseline, 4), "decisions": state.decisions,
                    "feedback": state.feedback, "pending": len(state.pending),
                    "hasPreviousVision": state.previous_signature is not None,
                    "stepsSincePerson": state.steps_since_person, 'spatialMemory': state.memory.snapshot(),
                    'mission': state.navigator.snapshot(), 'engine': state.engine.snapshot(time.time())}

    def observe(self, client: str, observation: dict) -> None:
        """Collect visual memory even while a manual task owns locomotion."""
        with self.lock:
            state = self._state(client)
            self._observe_state(state, observation, time.time())

    def begin_social(self, client: str) -> None:
        with self.lock:
            state = self._state(client)
            minimum = state.navigator.minimum_travel
            state.navigator = SocialNavigator()
            state.navigator.minimum_travel = minimum
            deferred = state.engine.arbiter.deferred
            state.engine.arbiter = type(state.engine.arbiter)()
            state.engine.arbiter.deferred = deferred
            state.engine.phase, state.engine.intent = 'Explore', 'searching'

    def monitor(self, client: str, observation: dict, arguments: dict, now: float) -> str | None:
        with self.lock:
            state = self._state(client)
            return state.engine.monitor(observation, arguments, state.navigator, now)

    def gesture_feedback(self, client: str, action: str, status: str, detail: str) -> None:
        with self.lock:
            state = self._state(client)
            state.engine.gesture_feedback(action, status, detail, time.time(), state.navigator.ready)

    def reject_short_step(self, client: str, distance: float) -> None:
        with self.lock:
            distance = self._number(distance, '无法执行的步长')
            state = self._state(client)
            state.navigator.minimum_travel = max(state.navigator.minimum_travel, distance*1.25)
            state.memory.reject_goal(time.time())

    def waypoint(self, client: str, observation: dict, position: list, tolerance: float | None = None) -> dict:
        """Follow only known free cells in the visual odometry coordinate frame."""
        with self.lock:
            state = self._state(client)
            memory = state.memory
            navigation = exploratory_navigation(observation.get('stereo'))
            navigation['bodyBearing'] = body_bearing(observation)
            if not navigation['tracking'] or memory.position is None:
                return state.navigator.recover(observation, memory, time.time()) or {'reason': navigation['reason']}
            state.navigator.recovery_turns, state.navigator.recovery_direction = 0, None
            state.navigator.recovery_wait_since = None
            arrival = memory.resolution if tolerance is None else tolerance
            if not math.isfinite(arrival) or arrival <= 0:
                raise ValueError('到达容差必须为正有限数')
            if np.linalg.norm((np.asarray(position)-memory.position)[[0, 2]]) < arrival:
                return {'completed': True, 'reason': f'视觉定位确认已到达目标附近（{arrival:.3f} 米）'}
            key = memory.key(position)
            if key not in memory.reachable():
                raise ValueError('目标尚无视觉可通行路径；请先探索并从已知点选择')
            route = memory.route_to(key, 'user_waypoint') or {'available': False, 'reason': '视觉路径暂不可用'}
            if tolerance is not None:
                route = precise_endpoint(memory, position, key, route)
            if not route.get('available'):
                return {'reason': route['reason']}
            bearing = route['bearing']
            distance = min(route['distance'], safe_travel(navigation, bearing, route['distance']))
            length = attach_route({}, route, distance).get('routeLength', distance)
            if length < state.navigator.minimum_travel and abs(angle_delta(bearing-navigation['bodyBearing'])) > 18:
                return state.navigator.turn(bearing, 'user_waypoint', '沿视觉记忆路径迈步转向', navigation['bodyBearing'])
            if length < state.navigator.minimum_travel:
                return {'reason': '等待前方新鲜可通行深度'}
            return attach_route({'action': 'forward', 'duration': max(.6, distance/.5), 'relativeBearing': round(bearing, 2),
                    'distance': distance, 'continuous': distance >= 1.2, 'reason': '沿视觉记忆完整路线连续低速移动'}, route, distance)


def precise_endpoint(memory, position, key, route):
    """Refine a known-free final grid cell; no scene geometry enters the route."""
    delta = np.asarray(position)-memory.position
    final = [round(memory.bearing_to(delta), 3), round(float(np.linalg.norm(delta[[0, 2]])), 4)]
    if memory.key(memory.position) == key:
        return {'available': True, 'bearing': final[0], 'distance': final[1],
                'relativePath': [final], 'goal': list(key), 'intent': 'contact_approach'}
    path = route.get('relativePath')
    if path:
        last = path[-1]
        target_distance = memory.distance_to(key)
        target_bearing = memory.bearing_to(np.array([(key[0]+.5)*memory.resolution,
            memory.position[1], (key[1]+.5)*memory.resolution])-memory.position)
        if abs(last[1]-target_distance) < .002 and abs(angle_delta(last[0]-target_bearing)) < .01:
            route = {**route, 'relativePath': [*path[:-1], final]}
            if len(path) == 1:
                route.update(bearing=final[0], distance=final[1])
    return route


POLICY = BehaviorPolicy()
