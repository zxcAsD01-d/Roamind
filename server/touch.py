"""Visual touch job API. A whole validated operation is published atomically."""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

from touch_motion import build_plan, options, target_unchanged
from contact_tracking import ContactTracker, check_observation

LOCK = threading.RLock()
JOBS = {}
BASE = os.environ.get('MOTION_PERCEPTION_URL', 'http://127.0.0.1:8200')


def resolve(selection):
    request = urllib.request.Request(BASE+'/api/touch/resolve', json.dumps(selection).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'视觉触摸定位失败：{exc.read().decode()}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError('无法连接双目感知服务') from exc


def validate_live(plan):
    target = plan['touch']['target']
    current = resolve({'clientId': plan['touch']['clientId'], 'targetId': target['targetId']})
    check_observation(target, current)
    with LOCK:
        job = JOBS.get(plan['id'])
        if job is None:
            return current
        tracker = job['tracker']
        if job['status'] == 'generating' and current['observedAt']-tracker.last['observedAt'] > 1.5:
            # No motion is playing yet. Re-observe a still-compatible target,
            # never extrapolate velocity across a slow generation interval.
            target_unchanged(target, current)
            job['tracker'] = ContactTracker(current)
            return job['tracker'].snapshot()
        return tracker.update(current)


def track(body):
    with LOCK:
        job = JOBS.get(body.get('id'))
        if job is None or job['cancelled'] or job['status'] not in ('generating', 'ready'):
            raise ValueError('接触任务不存在、未就绪或已取消')
        plan = copy.deepcopy(job['plan'])
    return validate_live(plan)


def runner_instance():
    module = sys.modules['__main__']
    lock = getattr(module, 'REALTIME_RUNNER_LOCK', None)
    if lock is None:
        raise RuntimeError('触摸生成需要驻留 ARDY HTTP 服务')
    with lock:
        if module.REALTIME_RUNNER is None:
            module.REALTIME_RUNNER = module.DualRunner(module.RealtimeRunner)
        return module.REALTIME_RUNNER


def start(body):
    config = options(body)
    selection = body.get('selection')
    if not isinstance(selection, dict) or not isinstance(selection.get('clientId'), str):
        raise ValueError('需要视觉 selection.clientId 和 targetId 或 personId/bodyPart')
    target = resolve(selection)
    history = body.get('history')
    if not isinstance(history, dict) or not np.allclose(history.get('rootOrigin', []), [0, .92, 0]):
        raise ValueError('触摸历史需要标准控制坐标 rootOrigin=[0,0.92,0]')
    identity = uuid.uuid4().hex
    runner = runner_instance()
    source_history = {**history, 'restJoints': runner.primary.model.motion_rep.skeleton.neutral_joints.cpu().numpy().ravel().tolist()}
    plan = build_plan(source_history, target, config, identity, body.get('selfEffector', body.get('selfArm')))
    plan['touch']['clientId'] = selection['clientId']
    job = {'id': identity, 'status': 'generating', 'cancelled': False, 'error': '', 'created': time.time(),
           'plan': plan, 'motion': None, 'metrics': [], 'tracker': ContactTracker(target)}
    with LOCK:
        if any(j['status'] == 'generating' for j in JOBS.values()):
            raise ValueError('已有触摸任务正在生成，请先停止')
        JOBS[identity] = job
        while len(JOBS) > 4:
            del JOBS[next(iter(JOBS))]
    threading.Thread(target=run, args=(job, copy.deepcopy(history)), daemon=True, name='visual-touch').start()
    return {'id': identity, 'status': 'generating', 'target': target, 'plan': public_plan(plan)}


def public_plan(plan):
    return {key: copy.deepcopy(plan[key]) for key in ('id', 'duration', 'frames', 'handJoint', 'handTrack', 'handStart', 'effector', 'contactTrack', 'contactStart', 'touch')}


def run(job, history):
    try:
        runner = runner_instance()
        plan = job['plan']
        def progress(event):
            with LOCK:
                job['metrics'].append({key: event[key] for key in ('generationMs', 'constraintStats') if key in event})
        result = runner.generate(plan['promptTrack'][0], plan['duration'], progress, history,
                                 lambda: job['cancelled'], plan)
        validate_live(plan)
        with LOCK:
            if job['cancelled']:
                job['status'] = 'cancelled'
            else:
                job['motion'], job['status'] = result, 'ready'
    except Exception as exc:
        logging.exception('视觉触摸生成失败')
        with LOCK:
            job['status'] = 'cancelled' if job['cancelled'] else 'failed'
            job['error'] = str(exc)


def snapshot(identity):
    with LOCK:
        if identity not in JOBS:
            raise ValueError('触摸任务不存在或已过期')
        job = JOBS[identity]
        result = {key: copy.deepcopy(job[key]) for key in ('id', 'status', 'motion', 'metrics')}
        if job['error']:
            result['error'] = job['error']
        return result | {'plan': public_plan(job['plan'])}


def catalog():
    with LOCK:
        return {'jobs': [{key: job[key] for key in ('id', 'status', 'created', 'error')} for job in JOBS.values()]}


def cancel(body):
    with LOCK:
        job = JOBS.get(body.get('id'))
        if job is None:
            raise ValueError('触摸任务不存在')
        job['cancelled'] = True
        if job['status'] == 'ready':
            job['motion'], job['status'] = None, 'cancelled'
    return {'cancelled': True}


def validate_playback(body):
    from library_motion import arrays, joints, quality
    from collision import body_collisions
    with LOCK:
        job = JOBS.get(body.get('id'))
        if job is None or job['status'] != 'ready' or job['cancelled']:
            raise ValueError('触摸任务尚未就绪或已取消')
        original, plan = copy.deepcopy(job['motion']), copy.deepcopy(job['plan'])
    motion = body.get('motion')
    if not isinstance(motion, dict) or motion.get('parents') != original['parents'] or motion.get('bones') != original['bones']:
        raise ValueError('触摸播放动作需要相同的 Core27 骨架')
    if motion.get('restJoints') != original['restJoints']:
        raise ValueError('手臂修正不得改变骨架尺寸')
    local, root = arrays(motion)
    old_local, old_root = arrays(original)
    if local.shape != old_local.shape or not np.allclose(root, old_root, atol=1e-6):
        raise ValueError('手臂修正改变了根轨迹或动作帧数')
    prefix = len(root)-plan['frames']
    arm = plan['effector']['chain']
    others = [joint for joint in range(27) if joint not in arm]
    if not np.allclose(local[:prefix], old_local[:prefix], atol=1e-6) or not np.allclose(local[:, others], old_local[:, others], atol=1e-6):
        raise ValueError('接触修正不得改变历史或所选执行链之外的旋转通道')
    report = quality(motion)
    if body_collisions(joints(motion)[prefix:], motion['parents']):
        raise ValueError('VRM 手臂修正后的骨段碰撞复检失败')
    validate_live(plan)
    return {'valid': True, 'quality': report}


def enqueue(body):
    import runtime
    session = runtime.active_session(body.get('clientId'))
    selection = {**body.get('selection', {}), 'clientId': session['state'].get('perceptionClientId')}
    target = resolve(selection)
    return runtime.enqueue('touch', {'clientId': session['id'], **options(body),
                           **({'goalId': body['goalId']} if body.get('goalId') else {}),
                           'selection': {'clientId': selection['clientId'], 'targetId': target['targetId']}})
