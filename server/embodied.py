"""Environment-driven goal ownership and a visual-only high-level LLM boundary."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import behavior
import goals
from needs import Needs
from life_policy import IntentState, demand, resource_operation


@dataclass
class Kernel:
    needs: Needs = field(default_factory=Needs)
    enabled: bool = False
    held_since: float | None = None
    goal_id: str | None = None
    next_at: float = 0.
    failures: int = 0
    reason: str = '自主生活未启用'
    seen: set[str] = field(default_factory=set)
    recent: dict[str, float] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    requests: dict[str, tuple[dict, dict]] = field(default_factory=dict)
    intent: IntentState = field(default_factory=IntentState)


KERNELS: dict[str, Kernel] = {}


def kernel(client):
    return KERNELS.setdefault(client, Kernel())


def record(state, kind, detail):
    state.reason = detail
    state.events.append({'at': time.time(), 'kind': kind, 'detail': detail})
    state.events = state.events[-48:]


def suspend(client, reason='直接控制接管，自主生活已暂停'):
    state = KERNELS.get(client)
    if state and state.enabled:
        state.enabled = False
        record(state, 'paused', reason)


def summary(client):
    state = kernel(client)
    return {'enabled': state.enabled, 'goalId': state.goal_id, 'reason': state.reason,
            'retryAfterSeconds': round(max(0., state.next_at-time.time()), 2),
            'needs': state.needs.snapshot(), 'intent': state.intent.snapshot(),
            'consecutiveFailures': state.failures, 'events': copy.deepcopy(state.events)}


def capabilities():
    return {'version': 3, 'coordinateFrame': 'visual-odometry', 'units': 'meters-seconds',
            'intents': {'socialize': ['timeout'], 'navigate': ['position', 'timeout'],
                        'inspect': ['objectId', 'timeout'], 'interact': ['objectId', 'operation', 'duration', 'timeout'],
                        'contact': ['targetId', 'effector', 'operation', 'duration', 'amplitude', 'timeout'],
                        'body': ['operation', 'timeout'],
                        'action': ['action', 'duration', 'timeout'], 'face': ['heading', 'timeout']},
            'nodeOperations': {'sit': 'observe support -> approach -> sit -> verify support',
                               'lie': 'observe body-length support -> sit -> lie -> verify support',
                               'pickup': 'observe grasp node -> reach -> attach -> carry',
                               'eat': 'observe food grasp -> approach -> pick up -> mouth contact -> bite -> return -> verify',
                               'drink': 'observe water cup grasp -> approach -> pick up -> mouth contact -> sip -> return -> verify',
                               'play_computer': 'observe keyboard + seat -> sit -> touch keys -> play screen game -> verify input/time'},
            'bodyOperations': ['stand', 'release'],
            'needs': {'energy': '0..100; low reduces activity budget and seeks rest',
                      'boredom': '0..100; high increases exploration and node interactions',
                      'satiety': '0..100; seek food <=35; satisfied >=72',
                      'thirst': '0..100; seek water >=60; satisfied <=22'},
            'read': '/api/agent/context', 'submit': '/api/agent/intent', 'autonomy': '/api/agent/autonomy',
            'completion': 'actual-execution-reports; inspect/interact also require fresh post-action semantics',
            'autonomyPolicy': 'hierarchical-needs-FSM; retained-goal; boundary-only-rearbitration; visual-resource-memory; execution-verified-consumption',
            'limits': ['kinematic support/grasp with execution feedback; no force simulation', 'no game text-entry adapter',
                       'structured intents; external LLM supplies reasoning; no online neural training']}


def context(args):
    import runtime
    import affordances
    session = runtime.active_session(args.get('clientId'))
    client = session['id']
    with runtime.LOCK:
        belief = behavior.POLICY.snapshot(client)
        return {'clientId': client, 'observedAt': time.time(), 'coordinateFrame': 'visual-odometry',
                'kernel': summary(client), 'goal': goals.snapshot(client),
                'world': belief['engine']['world'], 'spatialMemory': belief['spatialMemory'],
                'behavior': belief['engine'], 'interactions': affordances.catalog({'clientId': client}),
                'body': copy.deepcopy(session['state'].get('body')),
                'execution': {key: session['state'].get(key) for key in ('playing', 'generating', 'bufferedSeconds', 'interaction')},
                'capabilities': capabilities()}


def intent_plan(raw):
    if not isinstance(raw, dict):
        raise ValueError('intent 必须为结构化目标对象')
    kind = raw.get('kind')
    allowed = capabilities()['intents'].get(kind) if isinstance(kind, str) else None
    if allowed is None or set(raw)-set(allowed)-{'kind'}:
        raise ValueError('目标类型或字段不受支持，请读取 /api/agent/capabilities')
    step = {**raw, 'kind': {'socialize': 'social', 'navigate': 'waypoint', 'inspect': 'interact'}.get(kind, kind)}
    if kind == 'inspect':
        step['operation'] = 'inspect'
    return {'title': '高层目标：'+kind, 'steps': [goals.validate_step(step)]}


def submit(args):
    import runtime
    request_id = args.get('requestId')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
        raise ValueError('需要 requestId（1–100 字符），重试同一目标必须复用此 ID')
    planned = intent_plan(args.get('intent'))
    session = runtime.active_session(args.get('clientId'))
    with runtime.LOCK:
        state = kernel(session['id'])
        old = state.requests.get(request_id)
        if old:
            if old[0] != planned:
                raise ValueError('同一 requestId 不能提交不同目标')
            return {'goal': copy.deepcopy(old[1]), 'replayed': True}
        task = goals.submit({**planned, 'clientId': session['id']})
        state.requests[request_id] = (planned, goals.TASKS[session['id']])
        while len(state.requests) > 32:
            del state.requests[next(iter(state.requests))]
        record(state, 'intent', task['title'])
        return {'goal': task, 'replayed': False}


def configure(args):
    import runtime
    enabled = args.get('enabled')
    if type(enabled) is not bool:
        raise ValueError('enabled 必须为布尔值')
    session = runtime.active_session(args.get('clientId'))
    with runtime.LOCK:
        state = kernel(session['id'])
        if enabled and not state.enabled:
            state.failures, state.next_at = 0, 0.
        state.enabled = enabled
        if not enabled:
            task = goals.TASKS.get(session['id'])
            if task and task['id'] == state.goal_id and task['status'] in goals.ACTIVE:
                goals.cancel({'clientId': session['id']})
        record(state, 'enabled' if enabled else 'paused', '自主生活已开启，等待视觉' if enabled else '自主生活已暂停')
        return summary(session['id'])


def tick(client, vision):
    import runtime
    state = kernel(client)
    session = runtime.SESSIONS.get(client, {}).get('state', {})
    options = runtime.INTERACTIONS.get(client)
    state.needs.update(time.time(), session, options.objects if options else {})
    if not state.enabled:
        return
    now, task = time.time(), goals.TASKS.get(client)
    state.recent = {key: expiry for key, expiry in state.recent.items() if expiry > now}
    if task and task['status'] in goals.ACTIVE:
        state.intent.follow(task, now)
        if not reconsider(client, state, task, session, now):
            state.reason = '保持当前目标：'+task['title']
            return
    if task and task['id'] == state.goal_id and task['id'] not in state.seen:
        finish(client, state, task, now)
    if now < state.next_at:
        return
    body = session.get('interaction', {})
    if body.get('heldObject'):
        state.held_since = state.held_since or now
        if now-state.held_since < 6:
            state.reason = '观察刚拿到的物品'; return
    else:
        state.held_since = None
    if body.get('supported') and state.needs.resting:
        if demand(state.needs) == 'rest':
            state.intent.enter('rest', 'rest', body.get('objectId'), now)
            state.reason = '保持休息，恢复精力'
            return
    stamp = vision.get('observedAt') if isinstance(vision, dict) else None
    if type(stamp) not in (int, float) or not 0 <= now-stamp <= 1.5:
        state.reason = '等待新鲜视觉输入'; return
    if any(c['clientId'] == client and c['status'] not in runtime.TERMINAL_STATUSES for c in runtime.COMMANDS.values()):
        return
    try:
        step, title = choose(client, state, now)
        task = goals.submit({'clientId': client, 'title': title, 'steps': [step], '_autonomous': True})
        state.goal_id = task['id']
        goals.TASKS[client]['drive'] = demand(state.needs)
        state.intent.follow(goals.TASKS[client], now)
        record(state, 'selected', title)
    except (ValueError, RuntimeError) as exc:
        state.failures += 1
        state.next_at = now+min(60, 2**min(6, state.failures))
        record(state, 'failed', str(exc))
        state.intent.enter('recover', demand(state.needs), None, now)


def reconsider(client, state, task, session, now):
    """Re-arbitrate our search only at a settled boundary. User tasks always own control."""
    if task['id'] != state.goal_id or task['steps'][task['index']]['kind'] != 'social':
        return False
    if session.get('playing') or session.get('generating') or session.get('interaction', {}).get('transitioning'):
        return False
    import runtime
    if any(c['clientId'] == client and c['status'] not in runtime.TERMINAL_STATUSES for c in runtime.COMMANDS.values()):
        return False
    wanted = demand(state.needs)
    import affordances
    operation = {'water': 'drink', 'food': 'eat'}.get(wanted)
    options = runtime.INTERACTIONS.get(client)
    found = operation and options and any(resource_operation(row.label) == operation
            and row.identity not in state.recent and row.remaining != 0 and row.confidence >= .4 for row in options.objects.values())
    if wanted == 'rest' and options:
        found = any(row.identity not in state.recent and now-row.observed < 1.5
                    and any(node['kind'] == 'support' for node in row.nodes) for row in options.objects.values())
    if wanted == task.get('drive', 'curiosity') and not found:
        return False
    goals.stop(client, 'superseded', '动作边界重新仲裁需求，保留视觉探索记忆')
    state.seen.add(task['id'])
    state.next_at = now
    return True


def finish(client, state, task, now):
    state.seen = {task['id']}
    successful = task['status'] == 'completed'
    state.failures = 0 if successful else state.failures+1
    state.next_at = now+(0 if successful else min(16, 2**min(4, state.failures)))
    step = task['steps'][0]
    if step['kind'] == 'interact':
        state.recent[step['objectId']] = now+(20 if successful else 90)
    if successful:
        state.needs.reward(step.get('operation', step['kind']))
    if successful and step['kind'] == 'social':
        with behavior.POLICY.lock:
            engine = behavior.POLICY._state(client).engine
            if engine.arbiter.target_id:
                engine.arbiter.deferred[engine.arbiter.target_id] = now+90
    record(state, task['status'], task['detail'])


def choose(client, state, now):
    import affordances
    import runtime
    body = runtime.SESSIONS.get(client, {}).get('state', {}).get('interaction', {})
    catalog = affordances.catalog({'clientId': client})
    state.recent = {key: expiry for key, expiry in state.recent.items() if expiry > now}
    if body.get('supported'):
        return {'kind': 'body', 'operation': 'stand'}, '精力已恢复：起身继续活动'
    if body.get('heldObject'):
        return {'kind': 'body', 'operation': 'release'}, '放回手中物品'
    resource = choose_resource(catalog, state)
    if resource:
        return resource
    rest = choose_node(catalog, state, now)
    if rest:
        return rest
    budget = round(45+135*state.needs.snapshot()['initiative'])
    purpose = {'water': '寻找水杯或水瓶', 'food': '寻找食物', 'rest': '寻找可坐卧支撑面'}.get(demand(state.needs), '寻找新的社交机会')
    return {'kind': 'social', 'timeout': budget}, '持续探索：'+purpose


def choose_resource(catalog, state):
    operation = {'water': 'drink', 'food': 'eat'}.get(demand(state.needs))
    if operation is None:
        return None
    rows = [row for row in catalog['objects'] if resource_operation(row['label']) == operation
            and row['objectId'] not in state.recent and row['confidence'] >= .4 and row.get('remaining') != 0]
    if not rows:
        return None
    row = min(rows, key=lambda item: (item['distance'] if item['distance'] is not None else float('inf'), -item['confidence']))
    return {'kind': 'interact', 'objectId': row['objectId'], 'operation': operation, 'timeout': 180}, (
        '口渴：接近记忆中的饮水目标 ' if operation == 'drink' else '饥饿：接近记忆中的食物 ')+row['label']


def choose_node(catalog, state, now):
    if not state.needs.resting and demand(state.needs) != 'curiosity':
        return None
    operations = ('lie', 'sit') if state.needs.resting and state.needs.energy < .16 else ('sit', 'lie') if state.needs.resting else ('play_computer', 'pickup') if state.needs.boredom > .55 else ()
    for operation in operations:
        rows = [row for row in catalog['objects'] if row['fresh'] and row['objectId'] not in state.recent
                and any(option['operation'] == operation and option['enabled'] for option in row['options'])]
        if rows:
            row = min(rows, key=lambda item: item['distance'] if item['distance'] is not None else float('inf'))
            state.recent[row['objectId']] = now+60
            title = '寻找支撑面休息：' if state.needs.resting else '坐下玩电脑：' if operation == 'play_computer' else '尝试拿取：'
            return {'kind': 'interact', 'objectId': row['objectId'], 'operation': operation, 'timeout': 180,
                    **({'duration': 30} if operation == 'play_computer' else {})}, title+row['label']
    return None
