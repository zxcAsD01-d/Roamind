"""Utility selects a target; hysteresis preserves the intent across frames."""
class BehaviorArbiter:
    minimum_hold = 3.
    switch_margin = .45
    occlusion_hold = 8.

    def __init__(self):
        self.deferred = {}
        self.target_id = None
        self.since = 0.
        self.reason = '寻找新的探索和社交机会'
        self.candidates = []
        self.switches = 0

    def select(self, world, graph, now):
        group_members = {key for group in graph['groups'] for key in group['members']}
        self.deferred = {key: expiry for key, expiry in self.deferred.items() if expiry > now}
        visible = {row['id']: row for row in world.visible(now) if row['id'] not in self.deferred}
        self.candidates = sorted([self.score(row, group_members) for row in visible.values()],
                                 key=lambda row: (-row['utility'], row['id']))
        best = self.candidates[0] if self.candidates else None
        current = world.tracks.get(self.target_id) if self.target_id not in self.deferred else None
        if current and now-current['observedAt'] < self.occlusion_hold:
            if self.target_id not in visible:
                # Without world localization, a new detector identity at close
                # range is unverified. Let the controller reset its confirmation.
                near_unlocalized = (current.get('worldPosition') is None
                                    and current.get('distance') is not None and current['distance'] <= 1.7)
                if not near_unlocalized or best is None:
                    self.reason = '短暂遮挡，保持同一人物；重新确认前不声明交互成功'
                    return None
            else:
                score = next(row['utility'] for row in self.candidates if row['id'] == self.target_id)
                if (best is None or best['id'] == self.target_id or now-self.since < self.minimum_hold
                        or best['utility'] <= score+self.switch_margin):
                    self.reason = '保持当前目标，候选变化未超过切换阈值'
                    return dict(visible[self.target_id])
        next_id = best['id'] if best else None
        if next_id != self.target_id:
            self.switches += int(self.target_id is not None)
            self.target_id, self.since = next_id, now
        self.reason = '选择视觉社交机会' if best else '没有新鲜人物证据，持续探索'
        return dict(visible[next_id]) if next_id else None

    def score(self, row, group_members):
        distance = row.get('distance')
        components = {'evidence': 1.4*row['confidence'],
                      'travel': .6/(1+(distance or 6)/3),
                      'headFacing': .35 if row.get('attention', {}).get('facingCamera') is True else 0.,
                      'continuity': .35 if row['id'] == self.target_id else 0.,
                      'groupOpportunity': .15 if row['id'] in group_members else 0.}
        return {'id': row['id'], 'utility': round(sum(components.values()), 4),
                'components': components, 'kind': 'person'}

    def snapshot(self, now):
        return {'targetId': self.target_id, 'heldSeconds': round(max(0, now-self.since), 2) if self.target_id else 0,
                'reason': self.reason, 'switches': self.switches, 'candidates': self.candidates,
                'minimumHoldSeconds': self.minimum_hold, 'switchMargin': self.switch_margin,
                'occlusionHoldSeconds': self.occlusion_hold, 'scoreMeaning': 'heuristic-not-probability'}
