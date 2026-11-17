"""Standing reach envelopes from self calibration and visual target height."""
import math


def envelope(body, target_height, floor_height, effector='right_hand'):
    profile = body.get('effectors', {}).get(effector) if isinstance(body, dict) else None
    if not isinstance(profile, dict):
        return None
    reach, height, lateral = (profile.get(key) for key in ('reach', 'height', 'lateral'))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (reach, height, lateral)):
        raise ValueError('自身接触标定无效')
    if not .04 <= reach <= 1.4 or not 0 <= height <= 3 or not 0 <= lateral <= 1:
        raise ValueError('自身接触标定超出人体范围')
    vertical = target_height-floor_height-height
    horizontal_squared = (reach*.97)**2-vertical**2-lateral**2
    if horizontal_squared <= 0:
        return {'reachableHeight': False, 'maximum': 0., 'preferred': 0.}
    maximum = math.sqrt(horizontal_squared)
    return {'reachableHeight': True, 'maximum': maximum, 'preferred': maximum*.82}
