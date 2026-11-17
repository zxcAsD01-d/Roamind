"""Persistent observe -> approach -> operate plans, revalidated at every boundary."""
from __future__ import annotations

import math
import time

import behavior
from affordances import committed_approach
from visual_navigation import body_bearing
from social_navigation import angle_delta


def event(task, phase, detail):
    task['detail'], task['status'] = detail, 'running'
    task['events'].append({'at': time.time(), 'status': phase, 'detail': detail})
    task['events'] = task['events'][-48:]


def fresh_object(task, step, vision):
    import runtime
    session = runtime.active_session(task['clientId'])
    options = runtime.INTERACTIONS.get(task['clientId'])
    if options is None:
        raise ValueError('尚无同帧双目物体记忆')
    stereo = vision.get('stereo', {}) if isinstance(vision, dict) else {}
    if not stereo.get('mapping') or not -.1 <= time.time()-stereo.get('observedAt', 0) < 1.5:
        task.update(status='waiting', detail='等待视觉重定位'); return None
    row = options.require(step['objectId'], time.time(), fresh=False)
    if options.visual_client != session['state'].get('perceptionClientId'):
        raise ValueError('视觉会话改变，需重新选择物体')
    if step['operation'] != 'inspect' and row.confidence < .4:
        task.update(status='waiting', detail='目标置信度不足，等待复核后再操作'); return None
    return row


def tick(task, step, vision):
    import goals
    import runtime
    plan = task.setdefault('objectPlan', {'phase': 'observe', 'awaiting': None, 'observedAfter': 0.})
    if task['commandId']:
        command = runtime.command_status(task['commandId'])
        if command['status'] in ('failed', 'interrupted', 'superseded'):
            if command.get('retainedCommandId'):
                task['commandId'] = command['retainedCommandId']
                plan['handoffFailed'] = True
                return
            if plan['awaiting'] == 'approach' and goals.retry_navigation(task, {'kind': 'waypoint'}, command, time.time()):
                plan.pop('destination', None)
                return
            raise ValueError(command.get('detail') or '物体交互动作被中断')
        if command['status'] != 'completed':
            continue_approach(task, step, vision, command)
            return
        task['commandId'] = None
        plan.pop('handoffFailed', None)
        if plan['awaiting'] == 'node-execute':
            import node_behavior
            node_behavior.verify(task, step, command)
            finish(task, step, goals); return
        if plan['awaiting'] == 'execute':
            plan['phase'], plan['observedAfter'] = 'verify-result', command.get('completedAt', time.time())
        elif plan['awaiting'] == 'observe':
            plan['phase'], plan['observedAfter'] = 'verify', command.get('completedAt', time.time())
        else:
            plan['phase'], plan['observedAfter'] = 'verify', command.get('completedAt', time.time())
    row = fresh_object(task, step, vision)
    if row is None:
        return
    if time.time()-row.observed >= 1.5:
        seek_memory(task, row, vision)
        return
    if row.observed <= plan['observedAfter']:
        task.update(status='waiting', detail='等待动作完成后的新语义证据'); return
    if plan['phase'] == 'verify-result':
        finish(task, step, goals); return
    session = runtime.active_session(task['clientId'])
    if session['state'].get('playing') or session['state'].get('generating'):
        return
    if any(command['clientId'] == task['clientId'] and command['status'] in ('queued', 'received', 'playing', 'generating')
           for command in runtime.COMMANDS.values()):
        return
    with behavior.POLICY.lock:
        memory = behavior.POLICY._state(task['clientId']).memory
        if memory.position is None or memory.rotation is None:
            task.update(status='waiting', detail='等待可靠身体视觉定位'); return
        if plan['phase'] == 'observe' and step['operation'] == 'inspect':
            observe(task, row, memory, vision); return
        if step['operation'] in ('sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink'):
            import node_behavior
            node_behavior.execute(task, step, row, memory, vision); return
        if step['operation'] == 'inspect':
            finish(task, step, goals); return
        if step['operation'] in ('approach', 'touch', 'tap', 'hold', 'stroke'):
            if approach(task, step, row, memory, vision):
                return
        operate(task, step, row, memory, vision)


def continue_approach(task, step, vision, command):
    import runtime
    plan = task['objectPlan']
    session = runtime.active_session(task['clientId'])
    if (plan['awaiting'] != 'approach' or plan.get('handoffFailed')
            or not runtime._visual_continuation({'afterCommand': command['id']}, session, [command])):
        return
    row = fresh_object(task, step, vision)
    if row is None or time.time()-row.observed >= 1.5:
        return
    with behavior.POLICY.lock:
        memory = behavior.POLICY._state(task['clientId']).memory
        if memory.position is None or memory.rotation is None:
            return
        plan['continuationOf'] = command['id']
        try:
            if step['operation'] in ('sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink'):
                from affordances import select_node
                import node_behavior
                options = runtime.INTERACTIONS.get(task['clientId'])
                target = select_node(row, step['operation'], session['state'].get('body'), options.objects if options else None)
                if target:
                    node_behavior.navigate(task, step, target, memory, vision)
            else:
                approach(task, step, row, memory, vision)
        finally:
            plan.pop('continuationOf', None)


def seek_memory(task, row, vision):
    """Remembered VO locations guide travel; only fresh surfaces may trigger contact."""
    import runtime
    state = runtime.active_session(task['clientId'])['state']
    if state.get('playing') or state.get('generating') or any(
            c['clientId'] == task['clientId'] and c['status'] not in runtime.TERMINAL_STATUSES for c in runtime.COMMANDS.values()):
        return
    with behavior.POLICY.lock:
        memory = behavior.POLICY._state(task['clientId']).memory
        if memory.position is None or memory.rotation is None:
            task.update(status='waiting', detail='等待视觉重定位后沿记忆接近'); return
        distance = math.hypot(row.position[0]-memory.position[0], row.position[2]-memory.position[2])
        if distance <= 1.4:
            plan = task['objectPlan']
            if time.time() < plan.get('nextReobserveAt', 0):
                task.update(status='waiting', detail='等待新语义帧重新确认记忆目标'); return
            if plan.get('reobservations', 0) >= 3:
                raise ValueError('到达记忆位置后仍无法重新识别目标，暂时放弃并继续探索')
            plan['reobservations'] = plan.get('reobservations', 0)+1
            plan['nextReobserveAt'] = time.time()+2.
            observe(task, row, memory, vision)
            return
        destination = committed_approach(task['objectPlan'], memory, row.position, 1., 1.4)
        if destination is None:
            raise ValueError('记忆目标尚无连通的视觉路径，继续探索其他出口')
        result = behavior.POLICY.waypoint(task['clientId'], vision, destination, tolerance=.15)
        if not result.get('action'):
            task.update(status='waiting', detail=result.get('reason', '等待记忆路径更新')); return
        command = runtime.enqueue('visual', {**result, 'clientId': task['clientId'], 'goalId': task['id'],
                                            'viewHeading': vision.get('viewHeading', 0)})
        task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'approach'
        event(task, 'approach-memory', '沿视觉记忆前往目标附近，接触前重新识别：'+row.label)


def observe(task, row, memory, vision):
    import runtime
    bearing = memory.bearing_to([row.position[i]-memory.position[i] for i in range(3)])
    body_delta = angle_delta(bearing-body_bearing(vision))
    action = 'look' if abs(body_delta) <= 45 else ('turn_right' if body_delta > 0 else 'turn_left')
    state = runtime.active_session(task['clientId'])['state']
    height = (state.get('body') or {}).get('stature', 1.5)
    distance = math.hypot(row.position[0]-memory.position[0], row.position[2]-memory.position[2])
    pitch = math.degrees(math.atan2(row.position[1]-memory.position[1]-height*.9, max(.1, distance)))
    pitch += (state.get('body') or {}).get('cameraPitch', 0)
    args = {'viewPitch': max(-30, min(30, pitch)), 'clientId': task['clientId'], 'goalId': task['id'], 'action': action,
            'relativeBearing': bearing, 'duration': .4 if action == 'look' else 2,
            'viewHeading': vision.get('viewHeading', 0)}
    command = runtime.enqueue('visual', args)
    task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'observe'
    event(task, 'observe', '朝向并重新观察：'+row.label)


def approach(task, step, row, memory, vision):
    import runtime
    distance = math.hypot(row.position[0]-memory.position[0], row.position[2]-memory.position[2])
    from reach import envelope
    operation = step.get('operation', 'approach')
    contact = operation in ('touch', 'tap', 'hold', 'stroke')
    reach = envelope(runtime.active_session(task['clientId'])['state'].get('body'),
                     row.position[1], float(memory.position[1])) if contact else None
    if contact and reach is None:
        task.update(status='waiting', detail='等待角色自身手臂标定'); return True
    if reach and not reach['reachableHeight']:
        raise ValueError('当前站姿无法触及目标高度，需要可支撑的蹲下或抬身姿态')
    maximum = reach['maximum'] if reach else .8
    if distance <= maximum:
        return False
    preferred = reach['preferred'] if reach else .6
    destination = committed_approach(task['objectPlan'], memory, row.position, preferred, maximum)
    if destination is None:
        task.update(status='waiting', detail='保持目标，等待可达范围内的视觉落脚点'); return True
    result = behavior.POLICY.waypoint(task['clientId'], vision, destination,
                                      tolerance=(maximum-preferred)*.45 if contact else None)
    if result.get('completed'):
        raise ValueError('视觉路径已结束，但实际目标仍不在执行部位可达范围，需要调整落脚点')
    if not result.get('action'):
        task.update(status='waiting', detail=result['reason']); return True
    from node_behavior import continuation
    command = runtime.enqueue('visual', {**result, **continuation(task), 'clientId': task['clientId'], 'goalId': task['id'],
        'viewHeading': vision.get('viewHeading', 0)})
    task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'approach'
    event(task, 'approach', '沿视觉安全路径接近：'+row.label)
    return True


def operate(task, step, row, memory, vision):
    import goals
    import runtime
    import touch
    operation = step['operation']
    if operation == 'approach':
        finish(task, step, goals); return
    if operation == 'point':
        bearing = memory.bearing_to([row.position[i]-memory.position[i] for i in range(3)])
        delta = angle_delta(bearing-body_bearing(vision))
        if abs(delta) > 12:
            command = runtime.enqueue('visual', {'clientId': task['clientId'], 'goalId': task['id'],
                'action': 'turn_right' if delta > 0 else 'turn_left',
                'relativeBearing': bearing, 'viewHeading': vision.get('viewHeading', 0), 'duration': 2})
            task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'observe'
            event(task, 'align', '先转身对准指示方向'); return
        command = runtime.action({'clientId': task['clientId'], 'goalId': task['id'], 'action': 'point', 'duration': 3})
    else:
        if row.pixel is None:
            raise ValueError('当前物体没有实测表面接触点')
        session = runtime.active_session(task['clientId'])
        target = touch.resolve({'clientId': session['state'].get('perceptionClientId'),
            'frameId': row.frame, 'pixel': row.pixel, 'label': row.label})
        command = touch.enqueue({'clientId': task['clientId'], 'goalId': task['id'],
            'selection': {'targetId': target['targetId']}, 'effector': 'right_hand',
            'operation': operation, 'duration': .6, 'amplitude': .02})
    task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'execute'
    event(task, 'execute', '执行视觉物体操作：'+operation)


def finish(task, step, goals):
    import runtime
    options = runtime.INTERACTIONS.get(task['clientId'])
    if options and step['objectId'] in options.objects:
        options.objects[step['objectId']].completed[step['operation']] = time.time()
    task.pop('objectPlan', None)
    goals.advance(task, '物体交互阶段已按实际反馈完成：'+step['operation'])
