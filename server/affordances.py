"""Evidence-backed options, independent of object category or simulation geometry."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from object_precision import position_estimate

MAX_OBJECTS = 64
MAX_AGE = 30.
RESOURCE_AGE = 900.
OPERATIONS = ('inspect', 'approach', 'point', 'touch', 'tap', 'hold', 'stroke', 'sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink')
NODE_OPERATIONS = ('sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink')
LABELS = {'inspect': '观察', 'approach': '接近', 'point': '指向', 'touch': '触碰',
          'tap': '轻拍', 'hold': '保持接触', 'stroke': '轻抚', 'sit': '坐下休息', 'lie': '躺下休息', 'pickup': '拿起并持有',
          'play_computer': '坐下玩电脑', 'eat': '拿起食物并进食', 'drink': '拿起水杯并饮水'}


def vector(raw, size):
    if (not isinstance(raw, list) or len(raw) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)):
        return None
    return list(raw)


@dataclass
class ObjectEvidence:
    identity: str
    label: str
    position: list[float]
    pixel: list[float] | None
    box: list[float]
    frame: int
    observed: float
    confidence: float
    observations: int = 1
    completed: dict[str, float] = field(default_factory=dict)
    nodes: list[dict] = field(default_factory=list)
    remaining: float | None = None
    position_sigma: float = .08
    samples: list[tuple[float, list[float], float]] = field(default_factory=list)


class InteractionOptions:
    def __init__(self):
        self.objects: dict[str, ObjectEvidence] = {}
        self.last_stamp = 0.
        self.visual_client: str | None = None

    def update(self, vision, visual_client, now):
        if self.visual_client != visual_client:
            self.objects.clear()
            self.last_stamp = 0.
            self.visual_client = visual_client
        from life_policy import resource_operation
        self.objects = {key: row for key, row in self.objects.items()
                        if now-row.observed < (RESOURCE_AGE if resource_operation(row.label) else MAX_AGE)}
        if not isinstance(vision, dict):
            return
        fusion, stereo = vision.get('semanticFusion', {}), vision.get('stereo', {})
        if not isinstance(stereo, dict):
            return
        if stereo.get('pose', {}).get('status') == 'initialized' and stereo.get('observedAt', 0) > self.last_stamp:
            self.objects.clear()
        stamp = vision.get('semanticObservedAt')
        if (type(stamp) not in (int, float) or not math.isfinite(stamp) or not -.1 <= now-stamp < 1.5
                or stamp <= self.last_stamp or not isinstance(fusion, dict) or fusion.get('fused') is not True
                or not stereo.get('mapping') or type(fusion.get('frameId')) is not int):
            return
        self.last_stamp = stamp
        associations = fusion.get('associations', [])
        if not isinstance(associations, list):
            return
        for raw in associations[:32]:
            if isinstance(raw, dict):
                self._remember(raw, fusion['frameId'], stamp)
        if len(self.objects) > MAX_OBJECTS:
            self.objects = dict(sorted(self.objects.items(), key=lambda pair: pair[1].observed, reverse=True)[:MAX_OBJECTS])

    def _remember(self, raw, frame, stamp):
        identity, label = raw.get('id'), raw.get('label')
        point, box = vector(raw.get('worldPosition'), 3), vector(raw.get('box'), 4)
        confidence, support = raw.get('confidence'), raw.get('points', 0)
        if (not isinstance(identity, str) or not 1 <= len(identity) <= 100 or not isinstance(label, str)
                or not 1 <= len(label) <= 120 or point is None or box is None
                or type(confidence) not in (int, float) or not .2 <= confidence <= 1
                or type(support) not in (int, float) or not math.isfinite(support) or support < 20
                or box[2] <= box[0] or box[3] <= box[1]):
            return
        old = self.objects.get(identity)
        if old and stamp <= old.observed:
            return
        if old and math.dist(point, old.position) > 1.5:
            return  # Never silently reassign an existing identity to a remote object.
        estimate = position_estimate(raw, stamp, old)
        if estimate is None:
            return  # An outlier cannot refresh freshness or authorize contact.
        position, sigma, samples = estimate
        self.objects[identity] = ObjectEvidence(identity, label, position, vector(raw.get('surfacePixel'), 2),
            box, frame, stamp, confidence,
            (old.observations+1 if old else 1), (old.completed if old else {}), valid_nodes(raw.get('nodes')),
            old.remaining if old else None, sigma, samples)

    def require(self, identity, now, fresh=True):
        row = self.objects.get(identity)
        if row is None:
            raise ValueError('视觉物体不存在或已过期')
        if fresh and not -.1 <= now-row.observed < 1.5:
            raise ValueError('物体视觉证据过期，需重新观察')
        return row

    def snapshot(self, memory, now, body=None):
        rows = [self._public(row, memory, now, body) for row in self.objects.values()]
        rows.sort(key=lambda row: (-row['priority'], row['objectId']))
        return {'objects': rows, 'source': 'same-frame-yolo+stereo-surface',
                'planner': 'evidence-preconditions-v1', 'coordinateFrame': 'visual-odometry',
                'operations': list(OPERATIONS), 'maxAgeSeconds': MAX_AGE, 'resourceMemorySeconds': RESOURCE_AGE}

    def _public(self, row, memory, now, body=None):
        fresh = -.1 <= now-row.observed < 1.5
        position = memory.position
        distance = None if position is None else math.hypot(row.position[0]-position[0], row.position[2]-position[2])
        approach = approach_point(memory, row.position) if fresh and position is not None else None
        from reach import envelope
        reach = envelope(body, row.position[1], float(position[1])) if position is not None else None
        contact_approach = (approach_point(memory, row.position, reach['preferred'], reach['maximum'])
                            if reach and reach['reachableHeight'] and fresh else None)
        options = [self._option(row, operation, fresh, distance, approach, now) for operation in OPERATIONS
                   if operation not in NODE_OPERATIONS or row.nodes]
        for option in options:
            if option['operation'] in NODE_OPERATIONS:
                target = select_node(row, option['operation'], body, self.objects, now)
                remembered_resource = option['operation'] in ('eat', 'drink') and row.remaining != 0
                option['enabled'] = bool((fresh or remembered_resource) and row.confidence >= .4 and target)
                option['reason'] = '' if option['enabled'] else ('需要新鲜键盘表面与附近座椅节点' if option['operation'] == 'play_computer' else '需要新鲜实测支撑面/抓取点及自身尺寸标定')
                option['score'] = .4 if option['enabled'] else -1.
                option['stages'] = ['observe', 'approach', 'align', 'execute', 'verify-support-or-grasp']
                if option['operation'] in ('eat', 'drink'):
                    option['stages'] = ['navigate-memory', 'reobserve-grasp', 'pickup', 'mouth-contact', 'consume', 'return', 'verify-consumption']
                if option['operation'] == 'play_computer':
                    option['stages'] = ['observe-keyboard-and-seat', 'approach', 'sit', 'reach-keys', 'play', 'verify-inputs']
                continue
            if option['operation'] not in ('touch', 'tap', 'hold', 'stroke'):
                continue
            close = reach is not None and distance is not None and distance <= reach['maximum']
            valid = fresh and row.confidence >= .4 and row.pixel is not None
            option['enabled'] = bool(valid and reach and reach['reachableHeight'] and (close or contact_approach))
            if not option['enabled']:
                option['score'] = -1.
                option['reason'] = option['reason'] if not valid else ('等待自身执行部位标定' if reach is None else '当前站姿无法触及目标高度'
                                    if not reach['reachableHeight'] else '需先获得执行部位可达范围内的视觉落脚点')
            else:
                option['reason'] = ''
                option['stages'] = ['observe']+([] if close else ['approach'])+['execute']
            option['reach'] = reach
        return {'objectId': row.identity, 'label': row.label, 'position': row.position, 'frameId': row.frame,
                'observedAt': row.observed, 'confidence': row.confidence, 'observations': row.observations,
                'positionSigmaMeters': round(row.position_sigma, 4), 'positionEvidence': 'robust-last-eight-unique-exposures',
                'distance': distance, 'fresh': fresh, 'options': options, 'remaining': row.remaining,
                'nodes': [{k: v for k, v in node.items() if not k.startswith('control')} for node in row.nodes],
                'priority': max(option['score'] for option in options)}

    @staticmethod
    def _option(row, operation, fresh, distance, approach, now):
        reason, ready = '', fresh
        if not fresh:
            reason = '需要新的同帧语义与双目观测'
        elif operation != 'inspect' and row.confidence < .4:
            ready, reason = False, '检测置信度不足，仅允许复核观察'
        elif operation != 'inspect' and distance is None:
            ready, reason = False, '需要可靠视觉定位'
        elif operation == 'approach' and distance > .8 and approach is None:
            ready, reason = False, '尚无已知安全的接近落脚点'
        elif operation in ('touch', 'tap', 'hold', 'stroke'):
            if row.pixel is None:
                ready, reason = False, '缺少实测物体表面像素'

        novelty = 1./(1.+row.observations*.05)
        cooldown = max(0., 1.-(now-row.completed.get(operation, float('-inf')))/30.)
        cost = {'inspect': .8, 'point': 2., 'approach': 2.+(distance or 0)/.5,
                'touch': 4., 'tap': 4., 'hold': 5., 'stroke': 5., 'sit': 6., 'lie': 8., 'pickup': 5., 'play_computer': 30.,
                'eat': 12., 'drink': 12.}[operation]
        score = row.confidence*novelty/cost-2*cooldown if ready else -1.
        stages = ['observe']
        if operation == 'approach':
            stages.append('approach')
        if operation != 'inspect':
            stages.append('verify' if operation == 'approach' else 'execute')
        return {'id': row.identity+':'+operation, 'operation': operation, 'label': LABELS[operation],
                'enabled': ready, 'reason': reason, 'score': round(score, 4), 'stages': stages,
                'verification': 'fresh-evidence' if operation == 'inspect' else 'actual-playback-and-visual-feedback'}


def approach_point(memory, target, standoff=.8, maximum=None):
    if memory.position is None:
        return None
    outer = standoff+.4 if maximum is None else maximum
    paths = memory.reachable()
    candidates = []
    center = memory.key(target)
    radius = math.ceil(outer/memory.resolution)
    for x in range(center[0]-radius, center[0]+radius+1):
        for z in range(center[1]-radius, center[1]+radius+1):
            key = (x, z)
            if key not in paths.parents or memory.cells.get(key, (0,))[0] >= -.15:
                continue
            point = [(x+.5)*memory.resolution, float(memory.position[1]), (z+.5)*memory.resolution]
            gap = math.hypot(point[0]-target[0], point[2]-target[2])
            if standoff <= gap <= outer:
                candidates.append((paths.lengths[key]+abs(gap-standoff)*2, key, point))
    if maximum is not None:
        # Resolve contact placement inside known-free cells instead of quantizing
        # a short arm's entire reach envelope to 35 cm cell centres.
        for index in range(32):
            angle = index*math.tau/32
            point = [target[0]+standoff*math.sin(angle), float(memory.position[1]), target[2]+standoff*math.cos(angle)]
            key = memory.key(point)
            if key in paths.parents and memory.cells.get(key, (0,))[0] < -.15:
                candidates.append((paths.lengths[key], key, point))
    return min(candidates)[2] if candidates else None


def catalog(args):
    import behavior
    import runtime
    session = runtime.active_session(args.get('clientId'))
    with runtime.LOCK, behavior.POLICY.lock:
        state = behavior.POLICY._state(session['id'])
        options = runtime.INTERACTIONS.setdefault(session['id'], InteractionOptions())
        return copy.deepcopy(options.snapshot(state.memory, time.time(), session['state'].get('body')))


def plan(args):
    import runtime
    identity, operation = args.get('objectId'), args.get('operation', 'inspect')
    if not isinstance(identity, str) or operation not in OPERATIONS:
        raise ValueError('需要有效 objectId 和已支持的 operation')
    snapshot = catalog(args)
    row = next((row for row in snapshot['objects'] if row['objectId'] == identity), None)
    if row is None:
        raise ValueError('视觉物体不存在或已过期')
    selected = next((option for option in row['options'] if option['operation'] == operation), None)
    if selected is None:
        raise ValueError('当前目标尚未观察到该操作所需的支撑/抓取节点')
    if not selected['enabled']:
        raise ValueError(selected['reason'])
    return {'title': LABELS[operation]+'：'+row['label'], 'steps': [
        {'kind': 'interact', 'objectId': identity, 'operation': operation, 'timeout': 180 if operation == 'play_computer' else 120,
         **({'duration': args.get('duration', 30)} if operation == 'play_computer' else {})}],
        'preview': selected, 'evidenceFrameId': row['frameId'], 'revalidateAtExecution': True}


def submit(args):
    import goals
    prepared = plan(args)
    return goals.submit({**prepared, 'clientId': args.get('clientId')})


def valid_nodes(raw):
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:3]:
        if (not isinstance(item, dict) or item.get('kind') not in ('support', 'grasp', 'workstation')
                or item.get('source') != 'measured-stereo-surface' or item.get('frame') != 'control-world-y-up'):
            continue
        if any(vector(item.get(key), 3) is None for key in ('point', 'controlPoint', 'axis', 'controlAxis', 'size')):
            continue
        if any(item['size'][i] < 0 for i in range(3)) or vector(item.get('pixel'), 2) is None:
            continue
        sigma, samples = item.get('sigma'), item.get('samples')
        if type(sigma) not in (int, float) or not 0 <= sigma <= .03 or type(samples) is not int or samples < 40:
            continue
        result.append(copy.deepcopy(item))
    return result


def select_node(row, operation, body, objects=None, now=None):
    if not isinstance(body, dict):
        return None
    stature = body.get('stature')
    if type(stature) not in (int, float) or not .5 < stature < 3:
        return None
    if operation == 'play_computer':
        from computer_nodes import workstation
        return workstation(row, objects, stature, time.time() if now is None else now)
    if operation in ('eat', 'drink'):
        from life_policy import resource_operation
        if resource_operation(row.label) != operation or row.remaining == 0:
            return None
    for target in row.nodes:
        size = target['size']
        if operation in ('pickup', 'eat', 'drink') and target['kind'] == 'grasp' and max(size) <= .3:
            return target
        if operation in ('sit', 'lie') and target['kind'] == 'support':
            if operation == 'sit' and min(size[0], size[2]) >= .28:
                return target
            if operation == 'lie' and max(size[0], size[2]) >= stature*.85 and min(size[0], size[2]) >= .38:
                return target
    return None


def committed_approach(plan, memory, target, preferred, maximum):
    """Keep the same free landing point while the visual target remains consistent."""
    old, source = plan.get('destination'), plan.get('destinationTarget')
    if old is not None and source is not None and math.dist(source, target) < .12:
        key = memory.key(old)
        gap = math.hypot(old[0]-target[0], old[2]-target[2])
        if key in memory.reachable().parents and preferred-.08 <= gap <= maximum:
            return old
    result = approach_point(memory, target, preferred, maximum)
    plan['destination'], plan['destinationTarget'] = result, list(target)
    return result
