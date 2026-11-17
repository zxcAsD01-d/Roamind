"""Session-local drives, integrated from actual execution and bounded visual novelty."""
from dataclasses import dataclass, field


@dataclass
class Needs:
    energy: float = .8
    boredom: float = .25
    satiety: float = .78
    thirst: float = .18
    seeking_food: bool = False
    seeking_water: bool = False
    consumed: dict[str, float] = field(default_factory=dict)
    last_at: float | None = None
    resting: bool = False
    rest_since: float | None = None
    novelty: dict[str, float] = field(default_factory=dict)

    def update(self, now, execution, objects):
        dt = 0. if self.last_at is None else max(0., min(2., now-self.last_at))
        self.last_at = now
        body = execution.get('interaction', {})
        supported = body.get('supported') is True and body.get('posture') in ('sit', 'lie')
        computer = body.get('computer') or {}
        gaming = supported and body.get('activity') == 'computer' and computer.get('active') and computer.get('inputs', 0) > 0
        moving = execution.get('playing') and execution.get('locomotion', {}).get('phase') in ('start', 'walk', 'turn')
        active = bool(execution.get('playing'))
        # Units are fractions per second, not animation playback multipliers.
        recovery = .003 if body.get('posture') == 'lie' else .002
        rate = -.00025 if gaming else recovery if supported else (-.0009 if moving else -.0005 if active else .00008)
        self.energy = max(0., min(1., self.energy+rate*dt))
        self.boredom = max(0., min(1., self.boredom+dt*(-.004 if gaming else .0012 if supported or not active else .0002)))
        self.satiety = max(0., self.satiety-dt*(.00028+(.0001 if moving else 0)))
        self.thirst = min(1., self.thirst+dt*(.00045+(.0002 if moving else 0)))
        self._thresholds()
        self._novelty(now, objects)
        if supported and not gaming and self.rest_since is None:
            self.rest_since = now
        elif not supported or gaming:
            self.rest_since = None
        if self.energy <= .3:
            self.resting = True
        elif self.energy >= .68:
            self.resting = False

    def _thresholds(self):
        if self.satiety <= .35:
            self.seeking_food = True
        elif self.satiety >= .72:
            self.seeking_food = False
        if self.thirst >= .6:
            self.seeking_water = True
        elif self.thirst <= .22:
            self.seeking_water = False

    def consume(self, command, operation, receipt):
        """Called only after node verification; retries never replenish needs twice."""
        if command in self.consumed:
            return False
        if (operation not in ('eat', 'drink') or receipt.get('commandId') != command
                or receipt.get('operation') != operation or receipt.get('completed') is not True
                or receipt.get('returned') is not True):
            raise ValueError('缺少完整的实际吃喝回执')
        self.consumed[command] = self.last_at or 0.
        while len(self.consumed) > 128:
            del self.consumed[next(iter(self.consumed))]
        if operation == 'eat':
            self.satiety = min(1., self.satiety+.42)
        else:
            self.thirst = max(0., self.thirst-.55)
        self._thresholds()
        return True

    def _novelty(self, now, objects):
        self.novelty = {key: stamp for key, stamp in self.novelty.items() if now-stamp < 600}
        new = [row for row in objects.values() if row.identity not in self.novelty and row.confidence >= .4 and now-row.observed < 1.5]
        self.boredom = max(0., self.boredom-min(.04, len(new)*.012))
        for row in new:
            self.novelty[row.identity] = now
        self.novelty = dict(sorted(self.novelty.items(), key=lambda item: item[1])[-128:])

    def reward(self, kind):
        self.boredom = max(0., self.boredom-{'social': .25, 'pickup': .15, 'inspect': .07, 'play_computer': .05}.get(kind, .03))

    def snapshot(self):
        initiative = max(.2, min(1., (.3+.7*self.energy)*(.7+.6*self.boredom)*(1-.25*self.thirst)*( .75+.25*self.satiety)))
        return {'energy': round(self.energy*100, 2), 'boredom': round(self.boredom*100, 2),
                'satiety': round(self.satiety*100, 2), 'thirst': round(self.thirst*100, 2),
                'seekingFood': self.seeking_food, 'seekingWater': self.seeking_water,
                'initiative': round(initiative, 3), 'seekingRest': self.resting,
                'restSeconds': round(max(0., (self.last_at or 0)-(self.rest_since or self.last_at or 0)), 1),
                'policy': 'behavior-budget-only; native-motion-speed'}
