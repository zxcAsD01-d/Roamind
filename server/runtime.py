"""Thread-safe plans and commands shared by HTTP, MCP and the browser."""
from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from collections import deque

from planning import navigation, timed_plan
from world import get_world, point
import tracking
import behavior
import goals
import embodied
import lumen_actions
from visual_step import visual_step
from affordances import InteractionOptions

LOCK = threading.RLock()
PLANS = {}
SESSIONS = {}
COMMANDS = {}
METRICS = deque(maxlen=256)
INTERACTIONS: dict[str, InteractionOptions] = {}
TERMINAL_STATUSES = frozenset(('completed', 'failed', 'interrupted', 'superseded'))
REPORT_STATUSES = frozenset(('received', 'generating', 'playing', 'holding', 'tracking',
                             'completed', 'failed', 'interrupted'))


def save_plan(plan):
    with LOCK:
        PLANS[plan["id"]] = plan
        while len(PLANS) > 128:
            del PLANS[next(iter(PLANS))]
    return plan


def find_plan(plan_id):
    with LOCK:
        if plan_id not in PLANS:
            raise ValueError("规划不存在或已过期，请重新规划")
        return copy.deepcopy(PLANS[plan_id])


def summary(plan):
    result = {key: value for key, value in plan.items() if key not in ("rootTrack", "headingTrack", "promptTrack")}
    steps = plan.get('specification', {}).get('steps', [])
    timeline = []
    for index, step in enumerate(plan['timeline']):
        end = min(len(plan['headingTrack'])-1, max(0, round(step['end']*plan['fps'])-1))
        raw = steps[index] if index < len(steps) else {}
        path = step.get('path', [])
        if not path and step['distance'] > .001:
            path = [plan['rootTrack'][0], plan['rootTrack'][-1]]
        timeline.append({**step, 'path': path, 'heading': plan['headingTrack'][end],
                         'fixedHeading': 'heading' in raw or 'face' in raw})
    result['timeline'] = timeline
    return result


def plan_navigation(args):
    return summary(save_plan(navigation(args)))


def plan_timeline(args):
    return summary(save_plan(timed_plan(args)))


def rebase(args):
    old = find_plan(args.get("planId"))
    steps = copy.deepcopy(old["specification"]["steps"])
    return summary(save_plan(timed_plan({"start": point(args.get("start")), "steps": steps,
                                        'timing': old.get('timing', 'strict'),
                                        "startHeading":args.get("startHeading", 0)})))


def apply_report(client, report):
    command = COMMANDS.get(report.get('id'))
    status = report.get('status', 'received')
    if (not command or command['clientId'] != client or status not in REPORT_STATUSES
            or command['status'] in TERMINAL_STATUSES or status == command['status']):
        return
    if status == 'received' and command['status'] != 'queued':
        return
    if command['status'] == 'blending' and status not in TERMINAL_STATUSES:
        return
    command.update(status=status, detail=report.get('detail', ''))
    if command['kind'] == 'gesture':
        behavior.POLICY.gesture_feedback(client, command['arguments'].get('action', 'gesture'),
                                        status, command['detail'])
    retained = report.get('retainedCommandId')
    previous = COMMANDS.get(retained)
    if (status == 'failed' and retained == command['arguments'].get('continuationOf')
            and previous and previous['clientId'] == client and previous['status'] == 'blending'):
        command['retainedCommandId'] = retained
        previous['status'] = 'playing'
    if status == 'completed':
        command['completedAt'] = time.time()
    decision_id = command['arguments'].get('behaviorDecisionId')
    if decision_id and status in ('completed', 'failed', 'interrupted'):
        behavior.POLICY.feedback_for(client, decision_id, status, command['detail'], stale_ok=True)


def sync(args):
    client = args.get("clientId")
    if not isinstance(client, str) or not 1 <= len(client) <= 100:
        raise ValueError("clientId 无效")
    state = args.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("state 必须是对象")
    with LOCK:
        now = time.monotonic()
        SESSIONS[client] = {"id": client, "updated": now, "state": state}
        for key in list(SESSIONS):
            if now-SESSIONS[key]["updated"] > 120:
                del SESSIONS[key]
                INTERACTIONS.pop(key, None)
                embodied.KERNELS.pop(key, None)
        for report in args.get("reports", []):
            apply_report(client, report)
        vision = args.get('vision')
        if isinstance(vision, dict):
            behavior.POLICY.observe(client, vision)
        options = INTERACTIONS.setdefault(client, InteractionOptions())
        options.update(vision, state.get('perceptionClientId'), time.time())
        embodied.tick(client, vision)
        goals.tick(client, vision)
        pending = [c.copy() for c in COMMANDS.values() if c["clientId"] == client and c["status"] == "queued"]
        return {"commands": pending, "world": get_world()["id"], "targets": tracking.snapshots(),
                'kernel': embodied.summary(client), 'goal': goals.snapshot(client), 'behavior': behavior.POLICY.snapshot(client),
                'interactions': options.snapshot(behavior.POLICY._state(client).memory, time.time(), state.get('body'))}


def state():
    with LOCK:
        now = time.monotonic()
        sessions = [{**s, "ageSeconds": round(now-s["updated"], 2)} for s in SESSIONS.values()]
        return {"sessions": sessions, "commands": list(COMMANDS.values())[-20:],
                "world": get_world()["id"], "targets":tracking.snapshots(),
                'goals': [goals.snapshot(s['id']) for s in sessions if goals.snapshot(s['id'])]}


def active_session(client=None):
    with LOCK:
        candidates = [s for s in SESSIONS.values() if time.monotonic()-s["updated"] < 5 and (client is None or s["id"] == client)]
        if not candidates:
            raise ValueError("没有在线网页会话；请打开 http://127.0.0.1:5173/")
        return max(candidates, key=lambda s: s["updated"])


def enqueue(kind, args):
    session = active_session(args.get("clientId"))
    command = {"id": uuid.uuid4().hex, "clientId": session["id"], "kind": kind,
               "status": "queued", "created": time.time(), "arguments": args}
    if kind == "execute":
        find_plan(args.get("planId"))
    with LOCK:
        if not args.get('goalId'):
            if not args.get('_autonomous'):
                embodied.suspend(session['id'])
            goals.stop(session['id'], 'cancelled', '已切换到直接控制')
        for old in COMMANDS.values():
            if kind == 'gesture' and old['kind'] != 'gesture':
                continue
            if (kind == 'gesture' and old['kind'] == 'gesture'
                    and lumen_actions.side(args.get('action')) != lumen_actions.side(old['arguments'].get('action'))):
                continue
            if old['kind'] == 'gesture' and kind not in ('gesture', 'interrupt', 'touch', 'text-contact', 'interaction'):
                continue
            if old["clientId"] == session["id"] and old["status"] in ("queued", "received", "generating", "playing", "holding", "tracking"):
                old["status"] = "blending" if kind == "visual" and args.get("continuationOf") == old["id"] else "superseded"
        COMMANDS[command["id"]] = command
        while len(COMMANDS) > 128:
            del COMMANDS[next(iter(COMMANDS))]
    return command.copy()


def move(args):
    session = active_session(args.get("clientId"))
    planned = navigation({**args, "start": session["state"]["position"]})
    save_plan(planned)
    result = enqueue("navigate", {**args, "clientId": session["id"]})
    result["preview"] = summary(planned)
    return result


def action(args):
    from action_library import LIBRARY
    session = active_session(args.get('clientId'))
    selected = LIBRARY.describe(args['clipId']) if args.get('clipId') else args
    action_name, duration = selected.get('action'), selected.get('duration', 3)
    from social_actions import find
    preset = find(action_name)
    if preset and not args.get('clipId'):
        return enqueue('preset-action', {**args, 'action': action_name, 'prompt': preset['prompt'], 'clientId': session['id']})
    if (lumen_actions.supports(action_name) and not args.get('clipId')
            and not any(key in args for key in ('destination', 'heading', 'face'))):
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 1.5 <= duration <= 7.5:
            raise ValueError('流明动作时长应为 1.5–7.5 秒')
        return enqueue('gesture', {**args, 'action': action_name, 'duration': duration, 'clientId': session['id']})
    step = {'action': selected.get('action'), 'start': 0, 'end': selected.get('duration', 3),
            **{key: args[key] for key in ('destination', 'heading', 'face') if key in args}}
    plan = timed_plan({'start': session['state']['position'], 'startHeading': session['state']['heading'], 'steps': [step]})
    save_plan(plan)
    return enqueue('execute', {'planId': plan['id'], 'clientId': session['id'],
                               **({'goalId': args['goalId']} if args.get('goalId') else {}),
                               **({'clipId': args['clipId']} if args.get('clipId') else {})})


def plan_tracking(args):
    return summary(save_plan(tracking.tracking_plan(args)))


def face(args):
    session = active_session(args.get("clientId"))
    if args.get("targetId"):
        target = tracking.latest(args["targetId"])
        args = {**args, **({"face":target["position"]} if "position" in target else {"heading":target["bearing"]})}
    if "heading" not in args and "face" not in args:
        raise ValueError("指定 heading、face 或 targetId")
    return enqueue("face", {**args, "clientId":session["id"]})


def text_contact(args):
    import math
    text = args.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 200:
        raise ValueError('text 需要 1–200 字符的目标文字')
    if args.get('direction', 'anywhere') not in ('anywhere', 'left', 'right', 'up', 'down',
                                                'upper_left', 'upper_right', 'lower_left', 'lower_right'):
        raise ValueError('无效的文字方向筛选')
    if 'heading' in args and (type(args['heading']) not in (int, float) or not math.isfinite(args['heading'])):
        raise ValueError('heading 需要有限的朝向角度')
    if 'occurrence' in args and (type(args['occurrence']) is not int or args['occurrence'] < 0):
        raise ValueError('occurrence 需要从 0 开始的序号')
    return enqueue('text-contact', args)


def follow(args):
    tracking.latest(args.get("targetId"), True)
    session = active_session(args.get("clientId"))
    tracking.tracking_plan({**args, "start":session["state"]["position"]})
    return enqueue("follow", args)


def look(args):
    session = active_session(args.get('clientId'))
    if session['state'].get('playing') and session['state'].get('locomotion', {}).get('phase') != 'look':
        raise ValueError('独立转头命令需要站定；导航中的小幅转头由并行观察层控制')
    from head_actions import quantize, YAW_LIMIT, PITCH_LIMIT
    bearing = quantize(args.get('relativeBearing', 0), YAW_LIMIT)
    pitch = quantize(args.get('viewPitch', 0), PITCH_LIMIT)
    duration = args.get('duration', 1.2)
    if type(duration) not in (int, float) or not math.isfinite(duration) or not .2 <= duration <= 4:
        raise ValueError('转头时长必须在 0.2–4 秒之间')
    return enqueue('visual', {**args, 'clientId': session['id'], 'action': 'look',
                             'relativeBearing': bearing, 'duration': duration,
                             'viewPitch': pitch, 'lookAction': f'look_yaw_{-bearing:+d}',
                             'viewHeading': session['state'].get('heading', 0)})


def autonomy_decide(args):
    session = active_session(args.get("clientId"))
    with LOCK:
        goal = goals.TASKS.get(session['id'])
        if goal and goal['status'] in goals.ACTIVE and args.get('goalId') != goal['id']:
            return {'busy': True, 'reason': '目标任务正在控制角色'}
        busy = [command for command in COMMANDS.values()
                if command["clientId"] == session["id"]
                and command["status"] in ("queued", "received", "generating", "playing", "holding", "tracking")]
    continuation = _visual_continuation(args, session, busy)
    if busy and continuation is None:
        return {"busy": True, "command": busy[-1].copy()}
    if not continuation and not fresh_after_motion(session['id'], args.get('vision')):
        return {'busy': True, 'reason': '等待动作完成后的新双目帧'}
    decision = behavior.POLICY.decide({**args, "clientId": session["id"]})
    if decision.get('wait'):
        return {'busy': True, 'reason': decision['reason']}
    if decision.get('mission', {}).get('ready'):
        behavior.POLICY.feedback_for(session['id'], decision['decisionId'], 'completed')
        if not args.get('goalId'):
            with behavior.POLICY.lock:
                navigator = behavior.POLICY._state(session['id']).navigator
                if navigator.greeting_command is None:
                    navigator.greeting_command = action({'clientId': session['id'], 'action': 'wave', 'duration': 3})['id']
            return {'busy': False, 'completed': True, 'decision': decision, 'reason': '已接近并停稳，执行一次挥手'}
        return {'busy': False, 'completed': True, 'decision': decision, 'reason': decision['reason']}
    command_args = {"clientId": session["id"], "behaviorDecisionId": decision["decisionId"],
                    "action": decision["action"], "duration": decision["duration"],
                    "relativeBearing": decision["relativeBearing"]}
    command_args['distance'] = decision['distance']
    command_args['continuous'] = decision.get('continuous', False)
    if decision.get('relativePath'):
        command_args['relativePath'] = decision['relativePath']
    for key in ('targetId', 'targetPosition'):
        if key in decision:
            command_args[key] = decision[key]
    command_args['intent'] = decision['intent']
    if 'viewHeading' in args.get('vision', {}):
        command_args['viewHeading'] = args['vision']['viewHeading']
    if args.get('goalId'):
        command_args['goalId'] = args['goalId']
    if continuation:
        command_args["continuationOf"] = continuation
    command = enqueue("visual", command_args)
    return {"busy": False, "decision": decision, "command": command}


def _visual_continuation(args, session, busy):
    if len(busy) != 1:
        return None
    command = busy[0]
    if str(command['arguments'].get('intent', '')).startswith('recovering'):
        return None
    if command['arguments'].get('relativePath'):
        return None
    state = session["state"]
    remaining = state.get("bufferedSeconds", float("inf"))
    if type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining <= 0:
        return None
    locomotion = state.get('locomotion', {})
    handoff = locomotion.get('handoff', {})
    action = command['arguments'].get('action')
    look_ready = action == 'look' and handoff.get('gazeReady') is True
    turn_ready = (action in ('turn_left', 'turn_right') and locomotion.get('phase') == 'turn'
                  and 0 < remaining <= .65
                  and type(handoff.get('headingError')) in (int, float) and handoff['headingError'] <= 8
                  and type(handoff.get('tailDistance')) in (int, float) and handoff['tailDistance'] <= .08)
    brake_in = locomotion.get('brakeIn')
    prefetch = (command['arguments'].get('continuous') is True
                and locomotion.get('phase') in ('start', 'walk')
                and type(brake_in) in (int, float) and math.isfinite(brake_in) and 0 < brake_in <= 1.6)
    if (command["kind"] == "visual" and command["status"] == "playing"
            and (action == 'forward' or turn_ready or look_ready)
            and args.get("afterCommand") == command["id"] and not state.get("generating", True)
            and state.get("playing") is True
            and isinstance(remaining, (int, float)) and math.isfinite(remaining)
            and 0 < remaining and (prefetch or turn_ready or look_ready or remaining <= .25)):
        return command["id"]
    return None


def fresh_after_motion(client, vision):
    finished = max((command.get('completedAt', 0) for command in COMMANDS.values()
                    if command['clientId'] == client and command['kind'] == 'visual'), default=0)
    return not finished or (isinstance(vision, dict) and vision.get('observedAt', 0) > finished)


def autonomy_feedback(args):
    client = args.get("clientId")
    if not isinstance(client, str):
        raise ValueError("clientId 无效")
    return behavior.POLICY.feedback_for(client, args.get("decisionId"), args.get("status", "completed"),
                                        str(args.get("detail", "")))


def plan_visual_step(args):
    return summary(save_plan(visual_step(args)))


def approach_target(args):
    session = active_session(args.get("clientId"))
    target = tracking.latest(args.get("targetId"), True)
    distance = float(args.get("distance",1.2))
    if not .65 <= distance <= 5:
        raise ValueError("distance 必须在 0.65–5 米")
    path = tracking.approach(session["state"]["position"], target["position"],distance)
    return move({"destination":path[-1],"face":target["position"],
                 "clientId":session["id"],"via":path[1:-1]})


def command_status(command_id):
    with LOCK:
        if command_id not in COMMANDS:
            raise ValueError("命令不存在")
        return COMMANDS[command_id].copy()


def record_metric(event):
    with LOCK:
        METRICS.append({key: event[key] for key in ("generationMs", "frames", "constraintStats", "planId", "engine", "overlapFrames") if key in event})


def performance():
    with LOCK:
        samples = list(METRICS)
    values = sorted(s["generationMs"] for s in samples if "generationMs" in s)
    return {"windows": len(values), "p50Ms": values[len(values)//2] if values else 0,
            "p95Ms": values[min(len(values)-1, int(len(values)*.95))] if values else 0,
            "samples": samples}
