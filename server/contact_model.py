"""Kinematic contact vocabulary. Names describe constraints, never imply physical success."""
from __future__ import annotations

import copy
import numpy as np

EFFECTORS = {
    'right_hand': {'joint': 10, 'chain': [8, 9], 'bone': 'rightHand', 'radius': .035, 'kind': 'arm'},
    'left_hand': {'joint': 16, 'chain': [14, 15], 'bone': 'leftHand', 'radius': .035, 'kind': 'arm'},
    'right_foot': {'joint': 21, 'chain': [19, 20], 'bone': 'rightFoot', 'radius': .065, 'kind': 'leg'},
    'left_foot': {'joint': 25, 'chain': [23, 24], 'bone': 'leftFoot', 'radius': .065, 'kind': 'leg'},
    'head': {'joint': 6, 'chain': [3, 4, 5], 'bone': 'head', 'radius': .105, 'kind': 'axial'},
}

# Declarative prerequisites are deliberately separate from the executable primitives.
CONCEPTS = {
    'head_pat': {'effector': 'hand', 'target': 'scalp_surface', 'operation': 'tap',
                 'missing': ['scalp_surface_detection', 'palm_surface_calibration']},
    'lap_pillow': {'effector': 'head', 'target': 'thigh_surface', 'operation': 'hold',
                   'missing': ['supported_recline_pose', 'partner_seated_pose', 'load_bearing_contact']},
    'ear_tug': {'effector': 'hand', 'target': 'ear_surface', 'operation': 'grasp_pull',
                'missing': ['ear_surface_segmentation', 'finger_grasp', 'compliant_force_control']},
    'clap': {'effector': 'both_hands', 'target': 'palms', 'operation': 'synchronized_tap',
             'missing': ['palm_surface_calibration', 'paired_contact_timing']},
}


def capabilities():
    return {'effectors': copy.deepcopy(EFFECTORS), 'operations': ['touch', 'tap', 'hold', 'stroke'],
            'concepts': copy.deepcopy(CONCEPTS), 'targetInput': 'visual-token-only',
            'tracking': {'maxDisplacementMeters': .18, 'maxSpeedMetersPerSecond': .45,
                         'maxObservationAgeSeconds': 1.5},
            'learning': 'pretrained-pose-estimation; no online model training',
            'contactModel': 'kinematic-sphere-anchor; no force or finger simulation'}


def effector(config):
    name = config.get('effector', config.get('hand', 'right')+'_hand')
    if name not in EFFECTORS:
        raise ValueError('effector 需要 left_hand/right_hand/left_foot/right_foot/head')
    return name, EFFECTORS[name]


def calibration(raw, descriptor):
    legacy = descriptor['kind'] == 'arm' and isinstance(raw, dict) and 'shoulder' in raw
    keys = ('shoulder', 'elbow', 'hand') if legacy else ('base', 'middle', 'end')
    points = np.asarray([raw.get(key, []) for key in keys], dtype=float) if isinstance(raw, dict) else np.empty(0)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError('自身执行部位标定需要三个有限的三维坐标')
    reach = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    if not .04 <= reach <= 1.4:
        raise ValueError('自身执行部位标定长度无效')
    return points, float(reach)


def check_support(posed, descriptor):
    """Conservative pelvis-projection proxy, not a full dynamics/COM certificate."""
    if descriptor['kind'] != 'leg':
        return
    support = 25 if descriptor['joint'] == 21 else 21
    foot, toe = posed[support], posed[support+1]
    if foot[1] > .16 or toe[1] > .12:
        raise ValueError('脚部接触缺少落地的另一只支撑脚，请先进入稳定支撑姿态')
    delta = toe[[0, 2]]-foot[[0, 2]]
    fraction = np.clip((posed[0, [0, 2]]-foot[[0, 2]])@delta/max(delta@delta, 1e-8), 0, 1)
    distance = np.linalg.norm(posed[0, [0, 2]]-foot[[0, 2]]-fraction*delta)
    if distance > .075:
        raise ValueError('脚部接触的骨盆投影不在另一只脚的支撑范围；需要先生成重心转移姿态')
