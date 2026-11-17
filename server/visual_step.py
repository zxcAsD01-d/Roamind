"""Short relative motion compiled without a map or world geometry."""
from __future__ import annotations

import math
import uuid

import numpy as np


FPS = 20
PROMPT = "A person walks forward."


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 必须是有限数值")
    return number


def _point(value: object) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("start 必须是 [x,y,z]")
    point = np.asarray([_finite(item, "start") for item in value], dtype=np.float32)
    return point


def relative_path(raw, start, view_heading):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 32:
        raise ValueError('视觉路线需要 1–32 个相对路径点')
    points, length = [start.tolist()], 0.
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError('相对路径点必须是 [bearingDegrees, distanceMeters]')
        bearing, distance = [_finite(value, 'relativePath') for value in item]
        if not -180 <= bearing <= 180 or not .05 <= distance <= 12:
            raise ValueError('相对路径方向或距离超出范围')
        heading = view_heading-math.radians(bearing)
        point = start+np.array([math.sin(heading)*distance, 0, math.cos(heading)*distance])
        length += float(np.linalg.norm(point-np.asarray(points[-1])))
        points.append(point.tolist())
    if length > 12.05:
        raise ValueError('一次视觉路线不能超过 12 米')
    return points, length


def visual_step(args: dict) -> dict:
    start = _point(args.get("start"))
    heading = math.radians(_finite(args.get("startHeading", 0), "startHeading"))
    view_heading = math.radians(_finite(args.get('viewHeading', args.get('startHeading', 0)), 'viewHeading'))
    relative_bearing = math.radians(_finite(args.get("relativeBearing", 0), "relativeBearing"))
    action = args.get("action")
    if action not in ("forward", "turn_left", "turn_right", "observe"):
        raise ValueError("视觉动作必须是 forward/turn_left/turn_right/observe")
    duration = _finite(args.get("duration", 1.2), "duration")
    if not .2 <= duration <= 30:
        raise ValueError("视觉动作时长必须在 0.2–30 秒")
    path, route_length = (relative_path(args['relativePath'], start, view_heading)
                          if action == 'forward' and args.get('relativePath') else ([], None))
    if route_length is not None:
        duration = max(duration, route_length/.5)
    count = max(1, round(duration * FPS))
    if action in ('turn_left', 'turn_right'):
        # The signed visual bearing remains relative to the exposure camera,
        # even when the head looks to the opposite side of the body turn.
        if abs(relative_bearing) < 1e-6:
            relative_bearing = math.radians(-5 if action == 'turn_left' else 5)
        final_heading = view_heading-math.atan2(math.sin(relative_bearing), math.cos(relative_bearing))
    elif action == 'forward':
        final_heading = view_heading-relative_bearing
    else:
        final_heading = heading
    headings = np.linspace(heading, final_heading, count, dtype=np.float32)
    tracks = np.repeat(start[None, :], count, axis=0)
    if action == "forward":
        distance = _finite(args.get('distance', .65*duration), 'distance')
        if not 0 <= distance <= .65*duration+.001:
            raise ValueError('视觉前进距离必须在时长允许的安全范围内')
        progress = np.linspace(0, 1, count, dtype=np.float32)
        tracks[:, 0] += np.sin(final_heading) * distance * progress
        tracks[:, 2] += np.cos(final_heading) * distance * progress
    if path:
        source = np.asarray(path)
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
        progress = np.linspace(0, cumulative[-1], count)
        tracks = np.column_stack([np.interp(progress, cumulative, source[:, axis]) for axis in range(3)])
        headings = np.arctan2(np.gradient(tracks[:, 0]), np.gradient(tracks[:, 2]))
    return {"id": f"visual_{uuid.uuid4().hex[:12]}", "world": "visual-only", "fps": FPS, 'timing': 'flexible',
            "specification": {"mode": "ego-vision-only", "action": action, "duration": duration},
            "frames": count, "duration": duration, "start": start.tolist(),
            "timeline": [{"action": action, "start": 0, "end": duration, "path": path,
                          "continuous": action == 'forward' and args.get('continuous') is True,
                          "distance": route_length if path else float(np.linalg.norm(tracks[-1] - start)), "prompt": PROMPT}],
            "path": path, "rootTrack": tracks.tolist(), "headingTrack": headings.tolist(),
            "promptTrack": [PROMPT] * count,
            "collision": {"mode": "frontend-visual-feedback", "onFailure": "stop-and-replan"},
            "planningMs": 0}
