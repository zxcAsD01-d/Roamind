"""Two independent ARDY lanes, visual hand targets, immutable paired commits."""
from __future__ import annotations

import copy
import sys
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from library_motion import arrays, joints, quality

LOCK = threading.RLock()
JOBS = {}
PROMPT = 'A person touches something with their right hand.'


def evidence(raw):
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError('需要两个角色各自的视觉手腕观测')
    for actor in raw:
        point = np.asarray(actor.get('target', []), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('手部目标需要三个有限坐标')
        if actor.get('source') != 'yolo11-pose+stereo-sgbm-lr':
            raise ValueError('握手目标必须来自 YOLO 关键点与双目测距')
        if not 0 <= time.time()-float(actor.get('observedAt', 0)) < 2:
            raise ValueError('握手视觉目标过期')
        if not .4 <= float(actor.get('confidence', 0)) <= 1 or not 0 < float(actor.get('sigmaMeters', 1)) <= .18:
            raise ValueError('手腕识别或测距置信度不足')
    return copy.deepcopy(raw)


def runner_instance():
    module = sys.modules.get('__main__')
    runner = getattr(module, 'REALTIME_RUNNER', None)
    if runner is None:
        raise RuntimeError('ARDY 驻留模型尚未就绪')
    return runner


def start(body):
    observations = evidence(body.get('observations'))
    histories = body.get('histories', [])
    if len(histories) != 2:
        raise ValueError('需要两个角色的当前历史')
    for history in histories:
        arrays(history)
    duration = float(body.get('duration', 4))
    if not np.isfinite(duration) or not 2 <= duration <= 8:
        raise ValueError('握手时长必须在 2–8 秒')
    runner = runner_instance()
    identity = uuid.uuid4().hex
    job = {'id': identity, 'status': 'generating', 'observations': observations, 'chunks': [],
           'error': '', 'cancelled': False, 'created': time.time(), 'metrics': []}
    with LOCK:
        if any(j['status'] == 'generating' for j in JOBS.values()):
            raise ValueError('已有双角色生成任务，请先停止')
        JOBS[identity] = job
        while len(JOBS) > 4:
            del JOBS[next(iter(JOBS))]
    threading.Thread(target=run, args=(runner, job, histories, duration), daemon=True, name='visual-handshake').start()
    return {'id': identity, 'status': 'generating'}


def update(body):
    observed = evidence(body.get('observations'))
    with LOCK:
        job = find(body.get('id'))
        if job['status'] != 'generating':
            return {'accepted': False, 'status': job['status']}
        job['observations'] = observed
    return {'accepted': True}


def find(identity):
    if identity not in JOBS:
        raise ValueError('双角色任务不存在或已过期')
    return JOBS[identity]


def cancel(body):
    with LOCK:
        job = find(body.get('id'))
        job['cancelled'] = True
    return {'cancelled': True}


def snapshot(identity, after=-1):
    with LOCK:
        job = find(identity)
        return {key: copy.deepcopy(job[key]) for key in ('id', 'status', 'error', 'metrics')} | {
            'chunks': copy.deepcopy(job['chunks'][after+1:])}


def plan_for(history, target, total, frame, horizon, last_target):
    local, root = arrays(history)
    start = root[-1]+[0, .92, 0]
    forward = Rotation.from_rotvec(local[-1, 0]).apply([0, 0, 1])
    heading = float(np.arctan2(forward[0], forward[2]))
    posed = joints(history)[-1]
    shoulder, hand = posed[8], posed[10]
    target = np.asarray(target, dtype=float)
    reach = np.linalg.norm(posed[9]-shoulder)+np.linalg.norm(hand-posed[9])
    if np.linalg.norm(target-shoulder) > reach*1.12:
        raise ValueError('视觉握手目标超出手臂可达范围；需先重新接近')
    previous = hand if last_target is None else last_target
    delta = target-previous
    target = previous+delta*min(1., .20/max(np.linalg.norm(delta), 1e-6))
    count = min(horizon, total-frame)
    t = np.clip((np.arange(count)+1)/20, 0, 1)
    weights = t*t*t*(10+t*(-15+6*t))
    hands = previous+(target-previous)*weights[:, None]
    return {'id': 'visual-handshake', 'fps': 20, 'frames': count, 'start': start.tolist(),
            'timeline': [{'action': 'handshake', 'start': 0, 'end': count/20}],
            'rootTrack': np.tile(start, (count, 1)).tolist(), 'headingTrack': [heading]*count,
            'promptTrack': [PROMPT]*count, 'handTrack': hands.tolist()}, target


def generate_pair(runner, histories, observations, frame, total, targets):
    tasks, plans, next_targets = [], [], []
    for lane in range(2):
        history = histories[lane]
        local, root = arrays(history)
        origin = np.array([0, .92, 0], dtype=np.float32)
        plan, target = plan_for(history, observations[lane]['target'], total, frame, runner.primary.horizon, targets[lane])
        plans.append(plan); next_targets.append(target)
        task = runner.executor.submit(runner._infer, lane, local.reshape(len(root), -1).astype(np.float32),
                                      (root+origin).astype(np.float32), PROMPT, 0, origin, plan, lambda: False)
        tasks.append(task)
    windows = [task.result(timeout=30) for task in tasks]
    return windows, next_targets, plans


def commit(runner, histories, windows, count, plans):
    result, metrics = [], []
    for lane, (history, window) in enumerate(zip(histories, windows)):
        window = window.slice(0, count)
        local, root = arrays(history)
        window = runner._prepare_commit(window, local.reshape(len(root), -1), root+[0, .92, 0], None,
                                        np.array([0, .92, 0]), plans[lane])
        motion = {**history, 'frames': count, 'duration': count/20,
                  'localRot': window.local.ravel().tolist(), 'rootDelta': (window.root-[0, .92, 0]).ravel().tolist(),
                  'footContacts': window.contacts.astype(int).ravel().tolist()}
        report = quality(motion)
        prefix = {**motion, 'frames': history['frames']+count,
                  'localRot': history['localRot']+motion['localRot'], 'rootDelta': history['rootDelta']+motion['rootDelta'],
                  'footContacts': history.get('footContacts', [0]*(history['frames']*4))+motion['footContacts']}
        prefix['duration'] = prefix['frames']/20
        histories[lane] = prefix
        result.append(motion); metrics.append(report)
    return result, metrics


def run(runner, job, histories, duration):
    try:
        total, frame, targets = round(duration*20), 0, [None, None]
        with runner.session(lambda: job['cancelled']):
            while frame < total:
                if job['cancelled']:
                    raise RuntimeError('双角色生成已停止')
                started = time.perf_counter()
                with LOCK:
                    observed = evidence(job['observations'])
                windows, targets, plans = generate_pair(runner, histories, observed, frame, total, targets)
                count = min(20, total-frame)
                motions, metrics = commit(runner, histories, windows, count, plans)
                with LOCK:
                    job['chunks'].append({'index': len(job['chunks']), 'startFrame': frame, 'motions': motions})
                    job['metrics'].append({'frame': frame, 'generationMs': round((time.perf_counter()-started)*1000), 'quality': metrics})
                frame += count
        with LOCK:
            job['status'] = 'completed'
    except Exception as exc:
        import logging
        logging.exception('双角色握手生成失败')
        with LOCK:
            job['status'], job['error'] = 'failed', str(exc)
