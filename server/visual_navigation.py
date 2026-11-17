"""Conservative action bounds from stereo evidence; never imports scene geometry."""
from __future__ import annotations

import math
import time


def depth_navigation(raw, lead_seconds=0.):
    empty = {'valid': False, 'tracking': False, 'reason': '缺少双目深度', 'maxDistance': 0.,
             'front': 0., 'left': 0., 'right': 0., 'bearing': 35.}
    if not isinstance(raw, dict) or raw.get('method') != 'stereo-sgbm-lr':
        return empty
    timestamp = raw.get('observedAt')
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        return {**empty, 'reason': '双目时间戳无效'}
    age = time.time()-timestamp
    if not -.1 <= age <= 1:
        return {**empty, 'reason': '双目深度过期'}
    if raw.get('mapping') is not True:
        return {**empty, 'reason': '视觉定位尚未建立或已丢失'}
    ground = raw.get('groundInfo')
    if 'groundInfo' in raw and (not isinstance(ground, dict) or ground.get('frame') != 'gravity-aligned-y-up'
            or ground.get('support', 0) < 40 or ground.get('tiltDeg', 90) > 12 or ground.get('residual', 1) > .04):
        return {**empty, 'tracking': True, 'reason': '等待可落脚的重力对齐地板证据'}
    sectors = raw.get('sectors')
    if not isinstance(sectors, list) or len(sectors) > 31:
        return {**empty, 'reason': '双目方向净空无效'}
    valid = [item for item in sectors if _valid_sector(item)]
    front = next((item['clearance'] for item in valid if abs(item['angle']) < 8), 0.)
    left = max((item['clearance'] for item in valid if item['angle'] < -8), default=0.)
    right = max((item['clearance'] for item in valid if item['angle'] > 8), default=0.)
    best = max(valid, key=lambda item: item['clearance'], default={'angle': 35})
    # Account for body clearance, observation age, generation delay, and already buffered travel.
    travel = max(0., front-.5-.65*(max(0., age)+.35+lead_seconds))
    return {'valid': bool(front), 'tracking': True, 'reason': '双目可通行净空' if front else '前方深度未知',
            'maxDistance': round(min(3., travel), 3), 'front': front, 'left': left,
            'sectors': valid, 'safetyMargin': .5+.65*(max(0., age)+.35+lead_seconds),
            'right': right, 'bearing': max(25., abs(best['angle']))}


def safe_travel(navigation, bearing, known_distance=None):
    if navigation.get('visualAvoidance') is False:
        return min(3., known_distance) if known_distance is not None else 3.
    sectors = navigation.get('sectors')
    if not sectors:
        return navigation['maxDistance'] if not navigation.get('tracking') else 0.
    nearest = min(sectors, key=lambda sector: abs(sector['angle']-bearing))
    if abs(nearest['angle']-bearing) > 8:
        return 0.
    fresh = max(0., min(3., nearest['clearance']-navigation['safetyMargin']))
    # Floor coverage ends before many already mapped corridors do. Remembered
    # free space may extend the plan, but not a measured obstacle or stale view.
    if known_distance is None or nearest['clearance'] < .3 or 'obstacleDistance' not in nearest:
        return fresh
    obstacle = nearest['obstacleDistance']
    if obstacle is not None and (type(obstacle) not in (int, float) or not math.isfinite(obstacle) or obstacle <= 0):
        return fresh
    limit = 3. if obstacle is None else max(0., obstacle-navigation['safetyMargin'])
    return min(3., known_distance, limit)


def exploratory_navigation(raw):
    """Retain measured diagnostics without turning uncertainty into immobility."""
    return {**depth_navigation(raw), 'visualAvoidance': False, 'maxDistance': 3.}


def body_bearing(observation):
    values = [observation.get(key, 0.) for key in ('viewHeading', 'bodyHeading')]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise ValueError('相机与身体朝向必须是有限数值')
    if 'bodyHeading' not in observation:
        return 0.
    angle = math.radians(values[0]-values[1])
    return math.degrees(math.atan2(math.sin(angle), math.cos(angle)))


def _valid_sector(item):
    if not isinstance(item, dict):
        return False
    values = [item.get(key) for key in ('angle', 'clearance', 'confidence')]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return False
    angle, clearance, confidence = values
    return -75 <= angle <= 75 and 0 <= clearance <= 12 and .25 <= confidence <= 1


def restrict_scores(scores, navigation):
    scores = scores.copy()
    if navigation['maxDistance'] < .15:
        scores[0] = -1e6
    scores[1] += min(2., navigation['left'])*.8
    scores[2] += min(2., navigation['right'])*.8
    if navigation['front'] > 2:
        scores[0] += 1.
    return scores
