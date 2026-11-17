"""Observed targets and receding-horizon navigation; never consume a future path."""
import copy
import math
import threading
import time

import numpy as np

from planning import finite, timed_plan
from world import RADIUS, SKIN, inside, point, route, route_length

LOCK = threading.Lock()
TARGETS = {}


def observe(args):
    identifier = args.get("id")
    if not isinstance(identifier, str) or not 1 <= len(identifier) <= 100:
        raise ValueError("目标 id 必须为 1–100 个字符")
    stamp = finite(args.get("observedAt", time.time()), "observedAt")
    ttl = finite(args.get("ttl", 2), "ttl")
    if not .2 <= ttl <= 10 or abs(time.time()-stamp) > 10:
        raise ValueError("观测过期或 ttl 不在 0.2–10 秒之间")
    source = args.get("source", "manual")
    if source not in ("manual", "simulation", "ego-yolo", "audio"):
        raise ValueError("source 必须为 manual/simulation/ego-yolo/audio")
    target = {"id": identifier, "source": source, "observedAt": stamp, "ttl": ttl,
              "label": str(args.get("label", identifier))[:100]}
    if args.get("position") is not None:
        target["position"] = point(args["position"])
    if args.get("bearing") is not None:
        target["bearing"] = finite(args["bearing"], "bearing")
    if "position" not in target and "bearing" not in target:
        raise ValueError("至少提供世界 position 或绝对 bearing（度）")
    with LOCK:
        old = TARGETS.get(identifier)
        if old and old["observedAt"] > stamp:
            raise ValueError("拒绝倒序目标观测")
        TARGETS[identifier] = target
        if len(TARGETS) > 256:
            del TARGETS[min(TARGETS, key=lambda key: TARGETS[key]["observedAt"])]
    return target


def snapshots():
    with LOCK:
        return [{**copy.deepcopy(t), "ageSeconds": max(0, time.time()-t["observedAt"]),
                 "stale": time.time()-t["observedAt"] > t["ttl"]} for t in TARGETS.values()]


def latest(identifier, require_position=False):
    target = next((t for t in snapshots() if t["id"] == identifier), None)
    if target is None or target["stale"]:
        raise ValueError("目标未观测到或已丢失；停止跟随，不猜测未来轨迹")
    if require_position and "position" not in target:
        raise ValueError("目标只有方向、没有深度；可面向，不能安全移动/跟随")
    return target


def approach(start, target, distance):
    """Choose a reachable standoff, including objects whose centres are obstacles."""
    direction = math.atan2(start[0]-target[0], start[2]-target[2])
    candidates = []
    for offset in (0, .35, -.35, .7, -.7, 1.1, -1.1, math.pi):
        candidate = [target[0]+distance*math.sin(direction+offset), start[1],
                     target[2]+distance*math.cos(direction+offset)]
        if inside(candidate[0], candidate[2], RADIUS+SKIN):
            continue
        try:
            path = route(start, candidate)
        except ValueError:
            continue
        candidates.append((route_length(path), path))
    if not candidates:
        raise ValueError("目标附近没有安全跟随位置")
    return min(candidates, key=lambda pair: pair[0])[1]


def truncate(path, budget):
    result = [path[0]]
    for a, b in zip(path, path[1:]):
        length = math.hypot(b[0]-a[0], b[2]-a[2])
        if length > budget:
            result.append([a[i]+(b[i]-a[i])*budget/length for i in range(3)])
            break
        result.append(b)
        budget -= length
    return result


def tracking_plan(args):
    target = latest(args.get("targetId"), True)
    start = point(args.get("start"))
    distance = finite(args.get("distance", 1.2), "distance")
    speed = finite(args.get("speed", 1.4), "speed")
    if not .65 <= distance <= 5 or not .1 <= speed <= 1.6:
        raise ValueError("跟随距离必须在 0.65–5 米；speed 在 0.1–1.6 米/秒")
    separation = np.linalg.norm(np.array(start)[[0,2]]-np.array(target["position"])[[0,2]])
    path = [start] if abs(separation-distance) <= .15 else approach(start, target["position"], distance)
    path = truncate(path, speed*2/1.55)
    end = path[-1]
    step = {"action":"walk", "start":0, "end":2, "destination":end, "via":path[1:-1]}
    if args.get("faceTarget", True) or separation <= distance+.15:
        step["face"] = target["position"]
    planned = timed_plan({"start":start, "startHeading":args.get("startHeading", 0), "steps":[step], 'timing': 'flexible'})
    planned["tracking"] = {"targetId":target["id"], "observedAt":target["observedAt"],
                           "position":target["position"], "horizon":2, "usesFutureTrajectory":False}
    return planned
