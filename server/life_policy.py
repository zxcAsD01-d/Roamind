"""Demand arbitration and an inspectable, bounded hierarchical intent state."""
from dataclasses import dataclass, field


def resource_operation(label):
    text = label.lower()
    if any(word in text for word in ('sandwich', 'bread', 'apple', 'banana', '三明治', '面包', '苹果', '香蕉')):
        return 'eat'
    if any(word in text for word in ('cup', 'bottle', 'glass of water', '水杯', '水瓶')):
        return 'drink'
    return None


def demand(needs):
    """Hysteresis belongs to Needs; urgency only ranks latched demands."""
    ranked = []
    if needs.seeking_water:
        ranked.append((1.+needs.thirst, 'water'))
    if needs.seeking_food:
        ranked.append((1.+(1-needs.satiety)*.9, 'food'))
    if needs.resting:
        ranked.append((1.+(1-needs.energy)*.75, 'rest'))
    return max(ranked)[1] if ranked else 'curiosity'


@dataclass
class IntentState:
    phase: str = 'idle'
    drive: str = 'curiosity'
    target: str | None = None
    since: float = 0.
    transitions: int = 0
    history: list[dict] = field(default_factory=list)

    def enter(self, phase, drive, target, now):
        if (phase, drive, target) == (self.phase, self.drive, self.target):
            return
        self.phase, self.drive, self.target, self.since = phase, drive, target, now
        self.transitions += 1
        self.history.append({'phase': phase, 'drive': drive, 'target': target, 'at': now})
        self.history = self.history[-24:]

    def follow(self, task, now):
        step = task['steps'][min(task['index'], len(task['steps'])-1)]
        drive = task.get('drive', self.drive)
        phase = {'social': 'search-resource' if drive in ('water', 'food', 'rest') else 'explore-social',
                 'waypoint': 'navigate', 'body': 'body-transition'}.get(step['kind'], 'act')
        if step['kind'] == 'interact':
            awaiting = task.get('objectPlan', {}).get('awaiting')
            phase = 'use-resource' if awaiting == 'node-execute' and step['operation'] in ('eat', 'drink') else (
                'approach' if awaiting == 'approach' else 'interact')
        self.enter(phase, drive, step.get('objectId'), now)

    def snapshot(self):
        return {'phase': self.phase, 'drive': self.drive, 'targetId': self.target,
                'since': self.since, 'transitions': self.transitions, 'history': list(self.history)}
