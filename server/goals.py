"""Bounded, session-owned task runner shared by the browser and MCP."""
from __future__ import annotations

import copy
import math
import time
import uuid
import behavior

TASKS: dict[str, dict] = {}
ACTIVE = {'running', 'waiting'}
KINDS = {'social', 'waypoint', 'face', 'action', 'wait', 'contact', 'text-touch', 'interact', 'body'}


def finite(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} 必须在 {minimum}–{maximum}')
    return value


def validate_step(raw):
    if not isinstance(raw, dict) or raw.get('kind') not in KINDS:
        raise ValueError('未知任务步骤；支持 social/waypoint/face/action/wait/contact/text-touch/interact')
    step = {'kind': raw['kind'], 'timeout': finite(raw.get('timeout', 120), 'timeout', 1, 600)}
    kind = step['kind']
    if kind == 'body':
        if raw.get('operation') not in ('stand', 'release'):
            raise ValueError('body.operation 仅支持 stand/release')
        step['operation'] = raw['operation']
    elif kind == 'interact':
        from affordances import OPERATIONS
        identity, operation = raw.get('objectId'), raw.get('operation', 'inspect')
        if not isinstance(identity, str) or not 1 <= len(identity) <= 100 or operation not in OPERATIONS:
            raise ValueError('interact 需要有效 objectId 和受支持的 operation')
        step.update(objectId=identity, operation=operation)
        if operation == 'play_computer':
            step['duration'] = finite(raw.get('duration', 30), '电脑使用时长', 5, 120)
            if step['timeout'] < step['duration']+10:
                raise ValueError('电脑任务 timeout 需比 duration 至少多 10 秒，供坐下和收手')
    elif kind == 'waypoint':
        position = raw.get('position')
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError('waypoint.position 必须是视觉地图中的 [x,y,z] 米坐标')
        step['position'] = [finite(v, 'position', -1000, 1000) for v in position]
    elif kind == 'face':
        step['heading'] = finite(raw.get('heading'), 'heading', -3600, 3600)
    elif kind in ('wait', 'action'):
        step['duration'] = finite(raw.get('duration', 3), 'duration', .2, 4 if kind == 'wait' else 30)
        if kind == 'action':
            from lumen_actions import supports
            from social_actions import find
            if raw.get('action') not in ('crouch', 'squat', 'sit') and not supports(raw.get('action')) and not find(raw.get('action')):
                raise ValueError('不支持该动作')
            step['action'] = raw['action']
            if supports(step['action']):
                finite(step['duration'], '流明动作时长', 1.5, 7.5)
    elif kind == 'contact':
        from touch_motion import options
        identity = raw.get('targetId')
        if not isinstance(identity, str) or not 1 <= len(identity) <= 100:
            raise ValueError('contact.targetId 需要视觉标注目标令牌')
        step.update(targetId=identity, **options(raw))
    elif kind == 'text-touch':
        text = raw.get('text')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 200:
            raise ValueError('text-touch.text 必须是 1–200 字符')
        step['text'] = text.strip()
        direction = raw.get('direction', 'anywhere')
        if direction not in ('anywhere', 'left', 'right', 'up', 'down'):
            raise ValueError('无效文字方向')
        step['direction'] = direction
        if 'heading' in raw:
            step['heading'] = finite(raw['heading'], 'heading', -3600, 3600)
    return step


def submit(args):
    import runtime
    steps = args.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 24:
        raise ValueError('steps 需要 1–24 个步骤')
    validated = [validate_step(step) for step in steps]
    title = args.get('title', '角色任务')
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
        raise ValueError('title 需要 1–120 字符')
    session = runtime.active_session(args.get('clientId'))
    with runtime.LOCK:
        task = {'id': uuid.uuid4().hex, 'clientId': session['id'], 'title': title.strip(),
                'steps': [{**step, 'status': 'pending'} for step in validated], 'index': 0,
                'status': 'running', 'detail': '准备执行', 'createdAt': time.time(),
                'stepStartedAt': None, 'commandId': None, 'lastVisionAt': 0., 'events': []}
        autonomous = args.get('_autonomous') is True
        busy = session['state'].get('playing') or session['state'].get('generating') or any(
            c['clientId'] == session['id'] and c['status'] not in runtime.TERMINAL_STATUSES
            for c in runtime.COMMANDS.values())
        if not autonomous or busy:
            runtime.enqueue('interrupt', {'clientId': session['id'], 'mode': 'hold', '_autonomous': autonomous})
        TASKS[session['id']] = task
        while len(TASKS) > 128:
            del TASKS[next(iter(TASKS))]
        return copy.deepcopy(task)


def snapshot(client):
    import runtime
    with runtime.LOCK:
        return copy.deepcopy(TASKS.get(client))


def get_state(args):
    import runtime
    client = runtime.active_session(args.get('clientId'))['id']
    return {'goal': snapshot(client), 'behavior': behavior.POLICY.snapshot(client)}


def stop(client, status='cancelled', detail='用户取消任务'):
    task = TASKS.get(client)
    if task and task['status'] in ACTIVE:
        task['status'], task['detail'] = status, detail
        task['steps'][task['index']]['status'] = status
        task['events'].append({'at': time.time(), 'status': status, 'detail': detail})
        task['events'] = task['events'][-48:]


def cancel(args):
    import runtime
    session = runtime.active_session(args.get('clientId'))
    with runtime.LOCK:
        stop(session['id'])
        runtime.enqueue('interrupt', {'clientId': session['id'], 'mode': 'hold'})
        return {'goal': snapshot(session['id'])}


def advance(task, detail):
    task['steps'][task['index']]['status'] = 'completed'
    task['events'].append({'at': time.time(), 'status': 'completed', 'detail': detail})
    task['events'] = task['events'][-48:]
    task['index'] += 1
    task['stepStartedAt'], task['commandId'] = None, None
    task['detail'] = detail
    task['status'] = 'completed' if task['index'] == len(task['steps']) else 'running'


def tick(client, vision):
    import runtime
    task = TASKS.get(client)
    if not task or task['status'] not in ACTIVE:
        return
    try:
        _tick(task, vision)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        stop(client, 'failed', str(exc))
        runtime.enqueue('interrupt', {'clientId': client, 'mode': 'hold', 'goalId': task['id']})


def _tick(task, vision):
    import runtime
    now, client = time.time(), task['clientId']
    step = task['steps'][task['index']]
    if task['stepStartedAt'] is None:
        task['stepStartedAt'], step['status'] = now, 'running'
        if step['kind'] == 'social':
            behavior.POLICY.begin_social(client)
    if now-task['stepStartedAt'] > step['timeout']:
        raise ValueError('步骤超时：'+step['kind'])
    if step['kind'] == 'interact':
        import object_behavior
        object_behavior.tick(task, step, vision)
        return
    if task['commandId']:
        command = runtime.command_status(task['commandId'])
        if command['status'] in ('failed', 'interrupted', 'superseded'):
            if command.get('retainedCommandId'):
                task['commandId'] = command['retainedCommandId']
                task['detail'] = '下一动作准备被拒绝，当前动作继续完成'
                task['handoffFailed'] = True
                return
            if retry_navigation(task, step, command, now):
                return
            raise ValueError(command.get('detail') or '任务动作被中断')
        if command['status'] != 'completed':
            if monitor_social(task, step, command, vision, now):
                return
            state = runtime.active_session(client)['state']
            if (not task.get('handoffFailed') and step['kind'] in ('social', 'waypoint') and command['kind'] == 'visual'
                    and runtime._visual_continuation({'afterCommand': command['id']},
                                                     runtime.active_session(client), [command])):
                visual_step(task, step, vision)
            return
        if step.get('greetingCommand') == command['id']:
            advance(task, '已接近人物、停稳并完成挥手'); return
        task['commandId'] = None
        task['navigationFailures'] = 0
        task.pop('handoffFailed', None)
        if step['kind'] == 'body':
            import node_behavior
            node_behavior.verify_body(task, step, command)
        if step['kind'] not in ('social', 'waypoint'):
            advance(task, '动作已实际播放完成'); return
    session = runtime.active_session(client)
    if session['state'].get('playing') or session['state'].get('generating'):
        return
    if any(c['clientId'] == client and c['status'] in ('queued', 'received', 'playing', 'generating')
           for c in runtime.COMMANDS.values()):
        return
    if step['kind'] in ('social', 'waypoint'):
        return visual_step(task, step, vision)
    args = {key: value for key, value in step.items() if key not in ('kind', 'status', 'timeout')}
    args.update(clientId=client, goalId=task['id'])
    if step['kind'] == 'body':
        command = runtime.enqueue('interaction', args)
    elif step['kind'] == 'face':
        command = runtime.face(args)
    elif step['kind'] == 'action':
        command = runtime.action(args)
    elif step['kind'] == 'text-touch':
        command = runtime.text_contact(args)
    elif step['kind'] == 'contact':
        import touch
        command = touch.enqueue({**args, 'selection': {'targetId': args['targetId']}})
    else:
        command = runtime.enqueue('visual', {**args, 'action': 'observe'})
    task['commandId'], task['detail'], task['status'] = command['id'], '执行 '+step['kind'], 'running'


def monitor_social(task, step, command, vision, now):
    import runtime
    if (step['kind'] != 'social' or command['kind'] != 'visual' or command['status'] != 'playing'
            or now-command.get('created', now) < 1.2 or not isinstance(vision, dict)):
        return False
    reason = behavior.POLICY.monitor(task['clientId'], vision, command['arguments'], now)
    if reason is None:
        return False
    decision = command['arguments'].get('behaviorDecisionId')
    if decision:
        behavior.POLICY.feedback_for(task['clientId'], decision, 'interrupted', reason, stale_ok=True)
    stop = runtime.enqueue('interrupt', {'clientId': task['clientId'], 'goalId': task['id'], 'mode': 'hold'})
    task.update(commandId=stop['id'], detail=reason, status='running')
    task['events'].append({'at': now, 'status': 'person-route-update', 'detail': reason})
    task['events'] = task['events'][-48:]
    return True


def retry_navigation(task, step, command, now):
    detail = command.get('detail', '')
    if (step['kind'] not in ('social', 'waypoint') or command['kind'] != 'visual'
            or command['status'] != 'failed'
            or not detail.startswith(('预设根运动无法安全通过', '播放碰撞：', '移动死锁：', '视觉避障：', '移动距离不足一个安全落脚步',
                                      '基础步态质量未通过：', '规划准备超过保留的动作前缀',
                                      '规划期间动作源已改变'))):
        return False
    failures = task.get('navigationFailures', 0)+1
    if failures > 12:
        return False
    if detail.startswith('移动距离不足一个安全落脚步'):
        behavior.POLICY.reject_short_step(task['clientId'], command['arguments'].get('distance', 0))
    task.update(commandId=None, navigationFailures=failures, status='waiting',
                lastVisionAt=now, detail='局部移动受阻，停稳后重新观察并规划：'+detail)
    task['events'].append({'at': now, 'status': 'replanning', 'detail': detail})
    task['events'] = task['events'][-48:]
    return True


def visual_step(task, step, vision):
    import runtime
    import behavior
    if not isinstance(vision, dict) or time.time()-vision.get('observedAt', 0) > 1.5:
        task['status'], task['detail'] = 'waiting', '等待新鲜双目视觉'; return
    if not task['commandId'] and not runtime.fresh_after_motion(task['clientId'], vision):
        task['status'], task['detail'] = 'waiting', '等待动作完成后的新双目帧'; return
    stamp = vision.get('observedAt', 0)
    if stamp <= task['lastVisionAt']:
        return
    task['lastVisionAt'] = stamp
    args = {'clientId': task['clientId'], 'vision': vision, 'goalId': task['id'],
            'afterCommand': task['commandId'], 'exploreOnly': task.get('drive') in ('water', 'food', 'rest')}
    if step['kind'] == 'social':
        result = runtime.autonomy_decide(args)
        if result.get('completed'):
            if task['commandId']:
                task['detail'] = '社交条件已确认，等待当前脚步结束'; return
            command = runtime.action({'clientId': task['clientId'], 'goalId': task['id'], 'action': 'wave', 'duration': 3})
            step['greetingCommand'] = command['id']
            task['commandId'], task['status'], task['detail'] = command['id'], 'running', '人物距离与面向已确认，停稳后挥手'
            return
        task['detail'] = result.get('reason', result.get('decision', {}).get('reason', '视觉探索中'))
        command = result.get('command') if not result.get('busy') else None
    else:
        result = behavior.POLICY.waypoint(task['clientId'], vision, step['position'])
        task['detail'] = result['reason']
        if result.get('completed'):
            if task['commandId']:
                task['detail'] = '已进入目标附近，等待当前脚步结束'; return
            advance(task, result['reason']); return
        command = runtime.enqueue('visual', {**result, 'clientId': task['clientId'], 'goalId': task['id'],
                                  **({'viewHeading': vision['viewHeading']} if 'viewHeading' in vision else {}),
                                  'continuationOf': task['commandId']}) if result.get('action') else None
    task['status'] = 'running' if command else 'waiting'
    if command:
        task['commandId'] = command['id']
