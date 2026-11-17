"""Turn a retained visual target into a route, without importing scene truth."""
import math

import numpy as np


def route_to_target(memory, target, intent):
    if memory.position is None or memory.rotation is None or not memory.cells:
        return None
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('探索目标需要三个有限坐标')
    direct = float(np.linalg.norm((target-memory.position)[[0, 2]]))
    if direct < .45:
        return None
    paths = memory.reachable()
    key = memory.key(target)
    if key not in paths:
        key = route_prefix(memory, target, paths, direct)
    if key is None:
        return None
    route = memory.route_to(key, intent, paths)
    if not route or not route.get('relativePath'):
        return None
    # A route to an observed prefix remains a prefix. Never append a straight
    # shortcut through unobserved cells to make it appear to reach the target.
    return {**route, 'targetPosition': target.tolist(), 'partial': memory.key(target) != key}


def route_prefix(memory, target, paths, direct):
    ranked = []
    for key, length in paths.lengths.items():
        gap = float(np.linalg.norm(memory.cell_point(key)-target[[0, 2]]))
        progress = direct-gap
        if progress < .65 or length*memory.resolution > direct*3+3:
            continue
        ranked.append((gap+.04*length*memory.resolution, key))
    return min(ranked, default=(math.inf, None))[1]
