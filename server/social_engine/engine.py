"""Orchestration boundary between visual world, social intent and motion control."""
from collections import deque

import numpy as np

from .world_state import WorldState
from .social_graph import social_graph
from .arbitration import BehaviorArbiter


def phase_for(intent):
    if 'recover' in intent or intent in ('turn_back', 'inspect_blockage', 'escape_blockage', 'escape_deadlock', 'leave_deadlock'):
        return 'Recover'
    if intent in ('social_ready', 'confirming_social', 'await_attention'):
        return 'Join'
    if intent in ('approach_person', 'face_person', 'join_group', 'personal_space', 'social_detour'):
        return 'Approach'
    if intent in ('retain_person', 'social_observe'):
        return 'Observe'
    return 'Explore'


class SocialNavigationEngine:
    def __init__(self):
        self.world = WorldState()
        self.arbiter = BehaviorArbiter()
        self.graph = {'edges': [], 'groups': [], 'scoreMeaning': 'heuristic-not-probability'}
        self.phase = 'Explore'
        self.intent = 'searching'
        self.reason = '等待视觉行为目标'
        self.events = deque(maxlen=48)
        self.execution = {'status': 'idle', 'action': None, 'observedAt': 0., 'detail': ''}
        self.group_route = None
        self.graph_key = None
        self.social_enabled = True

    def observe(self, observation, now):
        self.world.update(observation, now)
        visible, position = self.world.visible(now), self.world.self_state['position']
        key = (self.world.revision, tuple(row['id'] for row in visible), tuple(position) if position else None)
        if key != self.graph_key:
            self.graph = social_graph(visible, position)
            self.graph_key = key

    def step(self, observation, navigation, memory, guidance, controller, now, social_enabled=True):
        self.observe(observation, now)
        controller.aggressive = navigation.get('visualAvoidance') is False
        controller.drive.observe_heading(observation)
        self.social_enabled = social_enabled
        person = self.arbiter.select(self.world, self.graph, now) if social_enabled else None
        if not social_enabled:
            controller.target, controller.ready, controller.confirmed_since, controller.confirmed_frames = None, False, None, 0
        result = self._group_step(person, navigation, memory, controller, now)
        if result is None:
            result = controller.step(observation, navigation, memory, guidance, now,
                                     selected_person=person, managed_target=True)
        phase = phase_for(result.get('intent', controller.phase))
        if phase != self.phase:
            self.events.append({'at': now, 'kind': 'phase', 'detail': f'{self.phase} → {phase}'})
        self.phase, self.intent = phase, result.get('intent', controller.phase)
        self.reason = result.get('reason', self.arbiter.reason)
        result['engine'] = self.snapshot(now)
        return result

    def _group_step(self, person, navigation, memory, controller, now):
        self.group_route = None
        if (person is None or controller.ready or controller.retreat is not None or controller.detour is not None
                or controller.drive.back_heading is not None or controller.blocked_turn or controller.escape_bearing is not None
                or person.get('distance') is None or person['distance'] <= 2.4 or memory.position is None):
            return None
        group = next((row for row in self.graph['groups'] if person['id'] in row['members'] and row['entry']), None)
        if group is None or np.linalg.norm((np.asarray(group['entry'])-memory.position)[[0, 2]]) < .7:
            return None
        key = memory.key(group['entry'])
        if key not in memory.reachable():
            return None
        route = memory.route_to(key, 'join_group')
        if not route.get('available'):
            return None
        controller.target = person
        result = controller.follow_guidance(navigation, memory, route, now)
        if result is not None:
            result.update(targetId=person['id'], targetPosition=person.get('worldPosition'))
            self.group_route = {'groupId': group['id'], 'entry': group['entry'], 'source': 'known-visual-route'}
        return result

    def monitor(self, observation, arguments, controller, now):
        if not self.social_enabled:
            return None
        self.observe(observation, now)
        person = self.arbiter.select(self.world, self.graph, now)
        return controller.monitor(observation, arguments, now, selected_person=person, managed_target=True)

    def feedback(self, action, status, detail, now):
        self.execution = {'action': action, 'status': status, 'detail': detail, 'observedAt': now}
        if status == 'failed':
            self.phase = 'Recover'
            self.reason = detail or '执行失败，等待恢复规划'
        if status in ('completed', 'failed', 'interrupted'):
            self.events.append({'at': now, 'kind': status, 'detail': f'{action} · {detail or status}'})

    def gesture_feedback(self, action, status, detail, now, social_ready):
        self.feedback(action, status, detail, now)
        if action == 'wave' and social_ready:
            if status == 'playing':
                self.phase = 'Interact'
            elif status == 'completed':
                self.phase = 'Maintain'

    def snapshot(self, now):
        return {'version': 1, 'phase': self.phase, 'intent': self.intent, 'reason': self.reason, 'socialEnabled': self.social_enabled,
                'world': self.world.snapshot(now), 'socialGraph': self.graph,
                'arbitration': self.arbiter.snapshot(now), 'groupRoute': self.group_route,
                'execution': self.execution, 'events': list(self.events),
                'layers': {'navigation': 'visual-memory-weighted-search',
                           'behavior': 'phase-utility-hysteresis', 'motion': 'preset-lumen-sparse-generation'}}
