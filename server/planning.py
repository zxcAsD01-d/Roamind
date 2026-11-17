"""Timed plans are compiled to per-frame collision-safe ARDY root conditions."""
from __future__ import annotations

import math
import time
import uuid

import numpy as np

from world import RADIUS, SKIN, get_world, point, resolve_target, route, route_length, segment_clear

FPS = 20
ACTIONS = {
    "walk": ("A person walks forward.", 1.6, .92),
    "run": ("A person runs.", 3.5, .92),
    "wave": ("A person waves their hand.", 0, .92),
    "raise": ("A person raises their right arm.", 0, .92),
    "point": ("A person points forward with their right arm.", 0, .92),
    "crouch": ("A person crouches down and stays low.", .7, .45),
    "sit": ("A person sits down.", 0, .55),
    "squat": ("A person squats down.", 0, .55),
    "crawl": ("A person crawls forward on hands and knees.", .4, .32),
}


def finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 必须是有限数值")
    return number


def navigation(args, start=None):
    action = args.get("action", "walk")
    if action not in ACTIONS or ACTIONS[action][1] <= 0:
        raise ValueError("移动动作必须是 walk/run/crouch/crawl")
    beginning = point(start or args.get("start", get_world()["spawn"]))
    destination = resolve_target(args.get("destination"))
    via = args.get("via", [])
    if not isinstance(via, list) or len(via) > 32:
        raise ValueError("via 必须是至多 32 个房间名或坐标的数组")
    points = [beginning] + [resolve_target(p) for p in via] + [destination]
    path = [points[0]]
    for a, b in zip(points, points[1:]):
        path.extend(route(a, b)[1:])
    speed = finite(args.get("speed", .5 if action == 'walk' else ACTIONS[action][1]), "speed")
    if not .1 <= speed <= ACTIONS[action][1]:
        raise ValueError(f"{action} 的 speed 必须在 0.1–{ACTIONS[action][1]} 米/秒")
    minimum = max(1., 1.5*route_length(path)/speed)
    duration = finite(args.get("duration", math.ceil(minimum*20)/20), "duration")
    if duration + .05 < minimum:
        raise ValueError(f"路线过长：至少需要 {minimum:.2f} 秒；不会加速穿过障碍物")
    return timed_plan({"start": beginning, "timing": 'flexible', "steps": [{"action": action, "start": 0,
        "end": duration, "destination": destination, "via": via,
        **{key: args[key] for key in ("heading", "face") if key in args}}],
        "startHeading": args.get("startHeading", 0)})


def validate_steps(args):
    start = point(args.get("start", get_world()["spawn"]))
    steps = args.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        raise ValueError("steps 必须有 1–32 段，包含 action/start/end 和可选 destination/via")
    return start, steps


def timed_plan(args):
    started = time.perf_counter()
    position, steps = validate_steps(args)
    origin = list(position)
    tracks, headings, prompts, timeline, paths, height_track = [], [], [], [], [], []
    previous_end, last_heading = 0., math.radians(finite(args.get("startHeading", 0), "startHeading"))
    for raw in steps:
        if not isinstance(raw, dict):
            raise ValueError("每段动作必须是包含 action/start/end 的对象")
        action = raw.get("action")
        if action not in ACTIONS:
            raise ValueError(f"不支持动作 {action}；可用：{', '.join(ACTIONS)}")
        begin, end = finite(raw.get("start", previous_end), "start"), finite(raw.get("end"), "end")
        if begin < previous_end - 1e-6 or end <= begin or end > 180:
            raise ValueError("时间必须顺序、不重叠，end > start，总时长 ≤180 秒")
        if abs(begin*FPS-round(begin*FPS)) > 1e-5 or abs(end*FPS-round(end*FPS)) > 1e-5:
            raise ValueError("时间必须按 0.05 秒对齐（20 fps）")
        gap = round((begin-previous_end)*FPS)
        tracks.extend([position.copy() for _ in range(gap)])
        headings.extend([last_heading]*gap)
        prompts.extend([ACTIONS["walk"][0]]*gap)
        height_track.extend([False]*gap)
        path = step_path(position, {**raw, "start": begin}, action)
        track, angle = sample_path(path, round((end-begin)*FPS), last_heading)
        angle = heading_track(track, angle, raw, last_heading)
        height = ACTIONS[action][2]
        ramp = np.linspace(0, 1, min(20, len(track)))
        track[:, 1] = height
        track[:len(ramp), 1] = position[1]+(height-position[1])*(ramp*ramp*(3-2*ramp))
        tracks.extend(track.tolist())
        headings.extend(angle.tolist())
        prompts.extend([ACTIONS[action][0]]*len(track))
        height_track.extend([height != ACTIONS['walk'][2]]*len(track))
        timeline.append({"action": action, "start": begin, "end": end, "path": path,
                         "distance": round(route_length(path), 3), "prompt": ACTIONS[action][0]})
        paths.extend(path if not paths else path[1:])
        position, previous_end, last_heading = track[-1].tolist(), end, float(angle[-1])
    return {"id": f"plan_{uuid.uuid4().hex[:12]}", "world": get_world()["id"], "fps": FPS,
            'timing': args.get('timing', 'strict'),
            "specification": {"steps": steps},
            "frames": len(tracks), "duration": previous_end, "start": origin, "timeline": timeline,
            "path": paths, "rootTrack": tracks, "headingTrack": headings, "promptTrack": prompts,
            'heightTrack': height_track,
            "collision": {"radius": RADIUS, "skin": SKIN, "rootConditions": len(tracks),
                          "bodyRefinementPasses": 2, "onFailure": "reject"},
            "planningMs": round((time.perf_counter()-started)*1000, 2)}


def heading_track(track, angles, raw, previous):
    """Degrees at the API boundary; shortest yaw turns, capped at 120 deg/s."""
    if "heading" in raw and "face" in raw:
        raise ValueError("heading 和 face 只能指定一个")
    if "heading" in raw:
        angles = np.full(len(track), math.radians(finite(raw["heading"], "heading")))
    elif "face" in raw:
        target = resolve_target(raw["face"])
        angles = np.arctan2(target[0]-track[:,0], target[2]-track[:,2])
    result = np.empty(len(track))
    for i, desired in enumerate(angles):
        difference = (desired-previous+math.pi) % (2*math.pi)-math.pi
        previous += float(np.clip(difference, -math.pi/1.5/FPS, math.pi/1.5/FPS))
        result[i] = previous
    return result


def step_path(position, raw, action):
    destination = resolve_target(raw["destination"]) if "destination" in raw else position.copy()
    via = raw.get("via", [])
    if not isinstance(via, list) or len(via) > 32:
        raise ValueError("via 必须是至多 32 个途经点的数组")
    waypoints = [position] + [resolve_target(p) for p in via] + [destination]
    path = [position.copy()]
    for a, b in zip(waypoints, waypoints[1:]):
        path.extend(route(a, b)[1:])
    distance = route_length(path)
    speed = ACTIONS[action][1]
    if distance > .01 and speed == 0:
        raise ValueError(f"{action} 是原地动作，不能带移动目的地")
    duration = finite(raw["end"], "end")-finite(raw.get("start", 0), "start")
    if speed and distance*1.5 > duration*speed+.08:
        raise ValueError(f"{action} 路线 {distance:.2f} 米，时间不足；需要至少 {distance*1.5/speed:.2f} 秒")
    return path


def sample_path(path, frames, last_heading):
    points = np.asarray(path, dtype=np.float32)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] < 1e-6:
        return np.repeat(points[:1], frames, axis=0), np.full(frames, last_heading)
    if frames < 2:
        raise ValueError("移动动作至少需要 0.1 秒（两帧）")
    t = np.linspace(0, 1, frames)
    progress = (t*t*(3-2*t))*distances[-1]
    # Preserve every turn as a frame: interpolating across a missed corner
    # can otherwise cut through an inflated wall despite a valid A* path.
    if frames < len(points):
        raise ValueError("动作帧数不足以经过全部路线转角，请增加时长")
    previous = 0
    for vertex in range(1, len(points)-1):
        nearest = int(np.argmin(np.abs(progress-distances[vertex])))
        index = max(previous+1, min(nearest, frames-len(points)+vertex))
        progress[index] = distances[vertex]
        previous = index
    track = np.stack([np.interp(progress, distances, points[:, axis]) for axis in range(3)], axis=1)
    for a, b in zip(track, track[1:]):
        if not segment_clear((a[0],a[2]), (b[0],b[2])):
            raise ValueError("逐帧路线验证失败；增加动作时长后重试")
    delta = np.gradient(track, axis=0)
    angle = np.unwrap(np.arctan2(delta[:, 0], delta[:, 2]))
    return track, angle
