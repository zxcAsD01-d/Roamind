"""Independent five-degree gaze actions; motor limits shared by HTTP and policy."""
import math

YAW_LIMIT = 75
PITCH_LIMIT = 30
STEP = 5


def quantize(value, limit):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f'观察角度必须在 ±{limit}° 内')
    return int(math.copysign(math.floor(abs(value)/STEP+.5)*STEP, value))


def catalog():
    return {'stepDegrees': STEP, 'yawLimit': YAW_LIMIT, 'pitchLimit': PITCH_LIMIT,
            'actions': [{'id': f'look_{axis}_{angle:+d}', 'axis': axis, 'degrees': angle}
                        for axis, limit in [('yaw', YAW_LIMIT), ('pitch', PITCH_LIMIT)]
                        for angle in range(-limit, limit+1, STEP)],
            'transition': 'bounded-acceleration; no idle barrier'}
