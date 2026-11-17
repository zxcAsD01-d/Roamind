"""Visual-node execution: approach uses VO; control points are only execution targets."""
import copy
import math

import behavior
from affordances import committed_approach, select_node

OPERATIONS = ('sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink')


def verify(task, step, command):
    import runtime
    state = runtime.active_session(task['clientId'])['state'].get('interaction', {})
    if state.get('lastCommand') != command['id']:
        raise ValueError('交互完成回执缺少对应的执行状态')
    operation = step['operation']
    if operation in ('sit', 'lie'):
        if state.get('posture') != operation or not state.get('supported') or state.get('objectId') != step['objectId']:
            raise ValueError('未确认角色在指定支撑面坐稳/躺稳')
    elif operation == 'pickup' and (not state.get('heldObject') or state.get('objectId') != step['objectId']):
        raise ValueError('尚未形成目标物体持有状态')
    elif operation in ('eat', 'drink'):
        verify_consumption(task, step, command, state)
    elif operation == 'play_computer':
        computer = state.get('computer') or {}
        if (state.get('posture') != 'sit' or not state.get('supported') or state.get('objectId') != step['objectId']
                or not computer.get('completed') or computer.get('active') or computer.get('inputs', 0) < 1
                or computer.get('elapsed', 0)+.1 < step.get('duration', 30)):
            raise ValueError('未确认坐稳、键盘输入与完整游戏时长')


def verify_consumption(task, step, command, state):
    import embodied
    import runtime
    receipt = state.get('consumption') or {}
    if (state.get('heldObject') or state.get('objectId') != step['objectId']
            or receipt.get('objectId') != step['objectId'] or receipt.get('commandId') != command['id']
            or receipt.get('operation') != step['operation'] or receipt.get('completed') is not True
            or receipt.get('returned') is not True):
        raise ValueError('缺少拿取、口部接触和放回的完整吃喝回执')
    for key, limit in (('mouthError', .035), ('graspError', .045)):
        value = receipt.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('吃喝接触误差超限：'+key)
    remaining = receipt.get('remaining')
    if type(remaining) not in (int, float) or not math.isfinite(remaining) or not 0 <= remaining <= 1:
        raise ValueError('缺少实际剩余份量')
    embodied.kernel(task['clientId']).needs.consume(command['id'], step['operation'], receipt)
    options = runtime.INTERACTIONS.get(task['clientId'])
    if options and step['objectId'] in options.objects:
        options.objects[step['objectId']].remaining = remaining


def execute(task, step, row, memory, vision):
    import runtime
    from object_behavior import event
    session = runtime.active_session(task['clientId'])
    options = runtime.INTERACTIONS.get(task['clientId'])
    target = select_node(row, step['operation'], session['state'].get('body'), options.objects if options else None)
    if target is None:
        raise ValueError('缺少满足角色尺寸的实测交互节点，请换角度观察支撑面或抓取点')
    if navigate(task, step, target, memory, vision):
        return
    command = runtime.enqueue('interaction', {'clientId': task['clientId'], 'goalId': task['id'],
        'operation': step['operation'], 'objectId': row.identity,
        **({'duration': step.get('duration', 30)} if step['operation'] == 'play_computer' else {}),
        'node': {**copy.deepcopy(target), 'observedAt': row.observed, 'frameId': row.frame}})
    task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'node-execute'
    event(task, 'execute', '执行支撑/抓取并等待物理结果：'+step['operation'])


def navigate(task, step, target, memory, vision):
    import runtime
    from object_behavior import event
    computer = step['operation'] == 'play_computer'
    support_target = target['seat'] if computer else target
    position = support_target['point']
    distance = math.hypot(position[0]-memory.position[0], position[2]-memory.position[2])
    if step['operation'] in ('pickup', 'eat', 'drink'):
        from reach import envelope
        body = copy.deepcopy(runtime.active_session(task['clientId'])['state'].get('body'))
        if body and 'right_hand' in body.get('effectors', {}):
            shoulder = body['effectors']['right_hand']
            shoulder['height'] -= min(.38, max(0., shoulder['height']-(position[1]-float(memory.position[1]))-.24))
        reach = envelope(body, position[1], float(memory.position[1]))
        if reach is None or not reach['reachableHeight']:
            raise ValueError('抓取点超出当前站姿手臂高度范围；需调整姿态或换目标')
        preferred, maximum = reach['preferred'], reach['maximum']
    else:
        # Approach a free edge of the observed patch, not the occupied furniture centre.
        half = min(support_target['size'][0], support_target['size'][2])/2
        support = (runtime.active_session(task['clientId'])['state'].get('body') or {}).get('support')
        if not support:
            raise ValueError('等待自身腿长和脚部标定')
        vertical = position[1]-float(memory.position[1])+.09-support['footHeight']
        horizontal = max(0., (support['legReach']*.98)**2-vertical**2)**.5
        maximum = half-.10+horizontal-support['footOffset']
        if maximum <= half+.12:
            raise ValueError('当前支撑高度与角色腿长不匹配')
        preferred = maximum-.025
    from computer_nodes import facing_seat, approach as computer_approach
    if distance <= maximum and (not computer or facing_seat(memory.position, target)):
        return False
    destination = computer_approach(memory, target, preferred) if computer else committed_approach(
        task['objectPlan'], memory, position, preferred, maximum)
    if destination is None:
        task.update(status='waiting', detail='寻找实测交互节点旁的可达地板'); return True
    result = behavior.POLICY.waypoint(task['clientId'], vision, destination, tolerance=.015)
    if result.get('completed'):
        raise ValueError('已到视觉落脚点但节点仍超出触达范围，需重新估计深度')
    if not result.get('action'):
        task.update(status='waiting', detail=result['reason']); return True
    command = runtime.enqueue('visual', {**result, **continuation(task), 'clientId': task['clientId'], 'goalId': task['id'],
                                        'viewHeading': vision.get('viewHeading', 0)})
    task['commandId'], task['objectPlan']['awaiting'] = command['id'], 'approach'
    event(task, 'approach', '接近视觉支撑/抓取节点')
    return True


def continuation(task):
    identity = task.get('objectPlan', {}).get('continuationOf')
    return {'continuationOf': identity} if identity else {}


def verify_body(task, step, command):
    import runtime
    state = runtime.active_session(task['clientId'])['state'].get('interaction', {})
    if state.get('lastCommand') != command['id']:
        raise ValueError('缺少对应命令的身体状态回执')
    if step['operation'] == 'stand' and (state.get('posture') != 'stand' or state.get('supported')):
        raise ValueError('起身尚未完成')
    if step['operation'] == 'release' and state.get('heldObject') is not None:
        raise ValueError('物品仍在手中')
