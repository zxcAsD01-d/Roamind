"""Bounded visual touch operations; all positions are meters in control-world Y-up."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from library_motion import arrays, joints
from contact_model import effector, calibration, check_support

HANDS = {'right': (8, 9, 10), 'left': (14, 15, 16)}
OPERATIONS = ('touch', 'tap', 'hold', 'stroke')
PROMPT = 'A person touches something with their right hand.'  # Cached text prior; explicit joint conditions select either hand.


def options(body):
    hand, operation = body.get('hand', 'right'), body.get('operation', 'touch')
    if hand not in HANDS or operation not in OPERATIONS:
        raise ValueError('hand 需要 left/right；operation 需要 touch/tap/hold/stroke')
    duration, amplitude = float(body.get('duration', .6)), float(body.get('amplitude', .01))
    if not np.isfinite([duration, amplitude]).all() or not .2 <= duration <= 3 or not 0 <= amplitude <= .04:
        raise ValueError('操作停留时间需要 0.2–3 秒；滑动幅度需要 0–4 cm')
    name, descriptor = effector(body)
    return {'hand': hand, 'effector': name, 'operation': operation, 'duration': duration,
            'amplitude': amplitude, 'clearance': descriptor['radius']}


def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*t*(10+t*(-15+6*t))


def target_unchanged(initial, current):
    if (current.get('targetId') != initial.get('targetId')
            or current.get('personId') != initial.get('personId') or current['frame'] != initial['frame']):
        raise ValueError('目标身份或坐标系改变，触摸已停止')
    if np.linalg.norm(np.asarray(current['position'])-initial['position']) > .06:
        raise ValueError('视觉目标移动超过 6 cm，请重新定位并规划')
    if np.dot(current['normal'], initial['normal']) < .8:
        raise ValueError('目标表面朝向改变，触摸已停止')


def operation_track(start, target, config, approach_frames=None):
    contact = np.asarray(target['position'])+config.get('clearance', .035)*np.asarray(target['normal'])
    hold = round(config['duration']*20)
    approach = max(24, int(np.ceil(np.linalg.norm(contact-start)/.35*20)))
    approach = max(approach, approach_frames or 0)
    if approach > 60:
        raise ValueError('执行部位接近距离过大，请先移动到目标附近')
    t = np.arange(1, approach+1)/approach
    reach = start+(contact-start)*smooth(t)[:, None]
    middle = np.tile(contact, (hold, 1))
    phase = np.linspace(0, 1, hold)
    if config['operation'] == 'tap':
        middle += (.025*np.sin(np.pi*phase)**4)[:, None]*np.asarray(target['normal'])
    elif config['operation'] == 'stroke':
        if config['amplitude'] > target['localSurfaceRadius']:
            raise ValueError('滑动幅度超过已观测表面范围，请缩小幅度或重新选择')
        middle += (config['amplitude']*np.sin(2*np.pi*phase)**3)[:, None]*np.asarray(target['tangent'])
    withdraw = contact+(start-contact)*smooth(t)[:, None]
    track = np.concatenate((reach, middle, withdraw))
    return track, [approach, approach+hold]


def build_plan(history, target, config, identity, self_arm=None):
    local, root = arrays(history)
    posed = joints(history)[-1]
    name, descriptor = effector(config)
    shoulder, elbow, hand = descriptor['chain'][0], descriptor['chain'][-1], descriptor['joint']
    chain = descriptor['chain']+[hand]
    reach = np.linalg.norm(np.diff(posed[chain], axis=0), axis=1).sum()
    check_support(posed, descriptor)
    source_target = target
    if self_arm is not None:
        points, actual_reach = calibration(self_arm, descriptor)
        # Map the anchor center, including its clearance, through body proportions.
        center = np.asarray(target['position'])+config['clearance']*np.asarray(target['normal'])
        position = posed[shoulder]+(center-points[0])*reach/actual_reach
        source_target = {**target, 'position': position.tolist()}
    else:
        raise ValueError('通用接触必须提供自身执行部位运动学校准')
    distance = np.linalg.norm(np.asarray(source_target['position'])-posed[shoulder])
    if distance > reach*.995:
        raise ValueError(f'目标超出所选执行部位可达范围（归一距离 {distance/reach:.2f}）；请先移动/转向后重新观测')
    source_config = {**config, 'clearance': 0}
    track, interval = operation_track(posed[hand], source_target, source_config)
    contact_track = None
    if self_arm is not None:
        contact_track, _ = operation_track(points[-1], target, config, interval[0])
        track, interval = operation_track(posed[hand], source_target, source_config, (len(contact_track)-round(config['duration']*20))//2)
    count = len(track)
    forward = Rotation.from_rotvec(local[-1, 0]).apply([0, 0, 1])
    heading = float(np.arctan2(forward[0], forward[2]))
    start = root[-1]+[0, .92, 0]
    prompt = PROMPT
    return {'id': identity, 'fps': 20, 'frames': count, 'duration': count/20, 'start': start.tolist(),
            'rootTrack': np.tile(start, (count, 1)).tolist(), 'headingTrack': [heading]*count,
            'promptTrack': [prompt]*count, 'handTrack': track.tolist(), 'handJoint': hand, 'handStart': posed[hand].tolist(),
            'effector': {**descriptor, 'name': name},
            'sourceShoulder': posed[shoulder].tolist(),
            'stanceLocal': local[-1].ravel().tolist(),
            'heightTrack': [True]*count, 'world': 'visual-touch', 'path': [start.tolist()],
            'timeline': [{'action': config['operation'], 'start': 0, 'end': count/20, 'distance': 0}],
            'contactTrack': None if contact_track is None else contact_track.tolist(),
            'contactStart': points[-1].tolist(),
            'touch': {'target': target, 'options': config, 'contactFrames': interval}}


def map_to_source(target, self_arm, source_shoulder, source_reach):
    arm = np.asarray([self_arm.get(key, []) for key in ('shoulder', 'elbow', 'hand')], dtype=float)
    if arm.shape != (3, 3) or not np.isfinite(arm).all():
        raise ValueError('自身手臂标定需要 shoulder/elbow/hand 三维有限坐标')
    reach = np.linalg.norm(arm[1]-arm[0])+np.linalg.norm(arm[2]-arm[1])
    if not .2 <= reach <= 1.2:
        raise ValueError('自身手臂标定长度无效')
    ratio = source_reach/reach
    position = source_shoulder+(np.asarray(target['position'])-arm[0])*ratio
    return {**target, 'position': position.tolist()}


def validate_hand(actual, plan, frame, shoulders=None):
    expected = np.asarray(plan['handTrack'])[frame:frame+len(actual)]
    if shoulders is not None:
        expected = expected+np.asarray(shoulders)-plan['sourceShoulder']
    error = np.linalg.norm(np.asarray(actual)-expected, axis=1)
    if not np.isfinite(error).all() or np.max(error) > .08:
        raise ValueError(f'最终执行部位轨迹误差 {np.max(error)*100:.1f} cm 超过 8 cm，拒绝播放')
    return {'handErrorMaxMeters': float(np.max(error)), 'handErrorMeanMeters': float(np.mean(error))}
