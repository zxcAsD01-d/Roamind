"""Shared house geometry and conservative capsule navigation; no model imports."""
from __future__ import annotations

import heapq
import math
from functools import lru_cache
from furniture_geometry import detailed_furnishings
from architecture_geometry import window_wall
from geometry_clearance import box_distance_squared, segment_box_squared

BOUNDS = [-10.0, -8.0, 10.0, 8.0]
RADIUS = 0.32
SKIN = 0.08
CELL = 0.2


def box(name, x, z, width, depth, height, material, kind="furniture", y=None):
    return {"id": name, "center": [x, height / 2 if y is None else y, z],
            "size": [width, height, depth], "material": material, "kind": kind}


def wall(axis, fixed, start, end, name, door=None):
    spans = [(start, end)] if door is None else [(start, door - 0.8), (door + 0.8, end)]
    return [box(f"{name}-{i}", (a+b)/2 if axis == "x" else fixed,
                fixed if axis == "x" else (a+b)/2,
                b-a if axis == "x" else 0.16, 0.16 if axis == "x" else b-a,
                2.8, "wall", "wall") for i, (a, b) in enumerate(spans) if b > a]


@lru_cache(maxsize=1)
def get_world():
    rooms = []
    for i, name in enumerate(["主卧", "卧室二", "卧室三", "卧室四"]):
        rooms.append({"id": f"bedroom{i+1}", "name": name, "kind": "bedroom",
                      "bounds": [-10+i*5, -8, -5+i*5, -1.2], "target": [-6.2+i*5, 0.92, -3], "material": "wood"})
    rooms[0]["target"] = [-6.2, 0.92, -3]
    rooms[1]["target"] = [-1.2, 0.92, -3]
    rooms[3].update(name="电脑房", kind="computer", target=[8, .92, -3.8])
    for key, name, bounds, target, kind in [
        ("living", "客厅", [-10, 1.2, -3, 8], [-5, 0.92, 3.2], "living"),
        ("dining", "餐厅", [-3, 1.2, 3, 8], [0, 0.92, 3.2], "living"),
        ("kitchen", "厨房", [3, 1.2, 7, 8], [5, 0.92, 3], "kitchen"),
        ("bathroom3", "公共卫生间", [7, 1.2, 10, 8], [8.5, 0.92, 3.2], "bathroom"),
        ("bathroom1", "主卧卫生间", [-10, -8, -7, -5.2], [-8.5, 0.92, -6.1], "bathroom"),
        ("bathroom2", "次卧卫生间", [-5, -8, -2, -5.2], [-3.5, 0.92, -6.1], "bathroom"),
        ("hallway", "中央走廊", [-10, -1.2, 10, 1.2], [0, 0.92, 0], "hallway"),
    ]:
        rooms.append({"id": key, "name": name, "kind": kind, "bounds": bounds,
                      "target": target, "material": "tile" if kind in ("bathroom", "kitchen") else "wood"})
    objects = window_wall(-8, [(-6.3, 1.7), (-1.3, 1.7), (3.7, 1.7), (8.7, 1.7)], 'north')
    objects += window_wall(8, [(-6.8, 2.8), (0, 2.1), (5, 1.5), (8.5, 1.1)], 'south')
    objects += wall("z", -10, -8, 8, "west") + wall("z", 10, -8, 8, "east")
    for i in range(4):
        objects += wall("x", -1.2, -10+i*5, -5+i*5, f"bed-door-{i}", -7.5+i*5)
        if i:
            objects += wall("z", -10+i*5, -8, -1.2, f"bed-divider-{i}")
    for i, (a, b, door) in enumerate([(-10,-3,-5), (-3,3,0), (3,7,5), (7,10,8.5)]):
        objects += wall("x", 1.2, a, b, f"south-door-{i}", door)
    for x in [-3, 3, 7]:
        objects += wall("z", x, 1.2, 8, f"south-divider-{x}")
    for offset in [0, 5]:
        objects += wall("x", -5.2, -10+offset, -7+offset, f"ensuite-door-{offset}", -8.5+offset)
        objects += wall("z", -7+offset, -8, -5.2, f"ensuite-wall-{offset}")
    objects += detailed_furnishings(furnishings())
    return {"id": "house-4b2l3b1k", "name": "三卧一电脑房 · 两厅三卫一厨", "units": "meters", "up": "yup",
            "bounds": BOUNDS, "spawn": [-5, 0.92, 3.2], "rooms": rooms, "objects": objects,
            "props": [{"id": "remote-control", "label": "remote control", "center": [-6.08, .468, 5.1],
                       "size": [.055, .036, .18]},
                      *[{"id": f"sandwich-{i}", "label": "sandwich", "kind": "food", "portions": 2,
                         "center": [-.58, .795, 5.1+i*.32], "size": [.16, .07, .14]} for i in range(3)],
                      {"id": "water-cup", "label": "cup of water", "kind": "water", "portions": 3,
                       "center": [2.35, .895, 3.10], "size": [.08, .12, .075]}],
            "computers": [{"id": "study-pc", "screen": [8, 1.04, -7.138], "screenSize": [.94, .53],
                           "keyboard": [7.9, .696, -6.56], "mouse": [8.36, .696, -6.59]}],
            "agent": {"radius": RADIUS, "height": 1.7, "skin": SKIN},
            "simulation": {"step": 1/60, "gravity": 0, "mode": "kinematic-capsule", "dynamicRigidBodies": False}}


def furnishings():
    items = [box("sofa", -8.8, 4.7, 1.2, 3.2, .85, "fabric"),
             box("coffee-table", -6.5, 5.4, 1.1, 1.4, .45, "oak"),
             box("tv-console", -3.6, 5.7, .55, 2.5, .55, "oak"),
             box("dining-table", 0, 5.6, 1.5, 2.1, .75, "oak"),
             box("kitchen-counter", 6.5, 5.4, .7, 4.5, .9, "stone"),
             box("kitchen-counter-back", 5, 7.5, 2.5, .7, .9, "stone"),
             box("fridge", 3.6, 7.2, .7, 1, 1.9, "metal"),
             box("water-dispenser", 2.35, 3.4, .5, .6, 1.35, "ceramic"),
             *[box(f"sandwich-plate-{i}", -.58, 5.1+i*.32, .23, .21, .012, "ceramic", y=.756) for i in range(3)],
             box("bath3-shower", 8.5, 7, 2, 1.5, 1.95, "glass"),
             box("bath3-basin", 9.5, 4.8, .6, 1, .85, "ceramic")]
    for i in range(4):
        x = -6.2+i*5
        if i != 3:
            items += [box(f"bed-{i}", x, -6.3, 1.8, 2.4, .52, "fabric"),
                      box(f"wardrobe-{i}", -9.6+i*5, -3.3, .6, 2, 2.2, "oak")]
    items += computer_furnishings()
    for offset in [0, 5]:
        items += [box(f"toilet-{offset}", -9.4+offset, -7.3, .7, .9, .7, "ceramic"),
                  box(f"basin-{offset}", -7.5+offset, -7.5, .6, .65, .85, "ceramic")]
    for x in [-1.25, 1.25]:
        for z in [5, 6.2]:
            items.append(box(f"chair-{x}-{z}", x, z, .5, .5, .85, "oak"))
    return items


def computer_furnishings():
    # Visible geometry and collision solids share dimensions; these are execution adapters, not observations.
    items = [box('computer-desk', 8, -6.98, 1.8, .85, .68, 'oak'),
             box('computer-chair', 8, -5.81, .56, .58, .88, 'fabric'),
             box('computer-tower', 9.25, -7.02, .3, .55, .58, 'metal'),
             box('computer-shelf', 5.5, -5.5, .6, 1.8, 1.7, 'oak')]
    items += [box('computer-monitor', 8, -7.18, 1.02, .07, .61, 'metal', y=1.04),
              box('computer-monitor-neck', 8, -7.18, .06, .06, .18, 'metal', y=.76),
              box('computer-monitor-base', 8, -7.12, .32, .22, .025, 'metal', y=.6925),
              box('computer-keyboard', 7.96, -6.59, .44, .18, .025, 'metal', y=.6875),
              box('computer-mouse', 8.36, -6.59, .062, .105, .028, 'metal', y=.689)]
    return items


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("坐标必须是 [x, y, z]，单位米")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("坐标必须是有限数值")
    if not 0.1 <= result[1] <= 2:
        raise ValueError("根节点高度必须在 0.1–2 米内")
    return result


def resolve_target(value):
    if isinstance(value, str):
        room = next((r for r in get_world()["rooms"] if value in (r["id"], r["name"])), None)
        if room is None:
            raise ValueError(f"未找到房间：{value}")
        return list(room["target"])
    return point(value)


def inside(x, z, radius=RADIUS+SKIN):
    if not (-10+radius < x < 10-radius and -8+radius < z < 8-radius):
        return True
    for obj in get_world()["objects"]:
        cx, _, cz = obj["center"]
        width, height, depth = obj["size"]
        if obj["center"][1]-height/2 < 1.7 and obj["center"][1]+height/2 > .03 and box_distance_squared(x, z, obj["center"], obj["size"]) < radius*radius-1e-10:
            return True
    return False


def segment_clear(a, b, radius=RADIUS+SKIN):
    if inside(*a, radius) or inside(*b, radius):
        return False
    for obj in get_world()["objects"]:
        if obj["center"][1]-obj["size"][1]/2 >= 1.7 or obj["center"][1]+obj["size"][1]/2 <= .03:
            continue
        center = (obj["center"][0], obj["center"][2])
        half = (obj["size"][0]/2, obj["size"][2]/2)
        if segment_box_squared(a, b, center, half) < radius*radius-1e-10:
            return False
    return True


def intersects_box(a, b, center, half):
    """Exact segment/AABB slabs; sampling can miss millimetre corner cuts."""
    enter, leave = 0., 1.
    for axis in range(2):
        low, high = center[axis]-half[axis], center[axis]+half[axis]
        delta = b[axis]-a[axis]
        if abs(delta) < 1e-10:
            if a[axis] < low or a[axis] > high:
                return False
            continue
        first, last = sorted(((low-a[axis])/delta, (high-a[axis])/delta))
        enter, leave = max(enter, first), min(leave, last)
        if enter > leave:
            return False
    return True


@lru_cache(maxsize=4)
def free_grid(radius):
    return {(i, j) for i in range(101) for j in range(81)
            if not inside(-10+i*CELL, -8+j*CELL, radius)}


def route(start, end, radius=RADIUS+SKIN):
    a, b = (start[0], start[2]), (end[0], end[2])
    if inside(*a, radius) or inside(*b, radius):
        raise ValueError("起点或终点位于障碍物/安全边距内；请先在网页中重置到房屋出生点")
    if segment_clear(a, b, radius):
        return [start, end]
    free = free_grid(radius)
    source = nearest_visible_node(a, free, radius)
    target = nearest_visible_node(b, free, radius)
    path = astar(source, target, free)
    points = [a] + [(-10+i*CELL, -8+j*CELL) for i, j in path] + [b]
    simplified = [points[0]]
    index = 0
    while index < len(points)-1:
        following = len(points)-1
        while following > index+1 and not segment_clear(points[index], points[following], radius):
            following -= 1
        simplified.append(points[following])
        index = following
    return [[x, start[1]+(end[1]-start[1])*i/(len(simplified)-1), z]
            for i, (x, z) in enumerate(simplified)]


def nearest_visible_node(position, free, radius):
    candidates = sorted(free, key=lambda node: math.hypot(-10+node[0]*CELL-position[0], -8+node[1]*CELL-position[1]))
    for node in candidates[:40]:
        if segment_clear(position, (-10+node[0]*CELL, -8+node[1]*CELL), radius):
            return node
    raise ValueError("起终点无法连接到导航网格")


def astar(source, target, free):
    opened = [(0., source)]
    score, came = {source: 0.}, {}
    while opened:
        _, current = heapq.heappop(opened)
        if current == target:
            path = [target]
            while path[-1] != source:
                path.append(came[path[-1]])
            return path[::-1]
        for dx, dz in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            node = (current[0]+dx, current[1]+dz)
            if node not in free:
                continue
            if dx and dz and ((current[0]+dx,current[1]) not in free or (current[0],current[1]+dz) not in free):
                continue
            cost = score[current] + math.hypot(dx, dz)
            if cost < score.get(node, math.inf):
                score[node], came[node] = cost, current
                heapq.heappush(opened, (cost+math.dist(node, target), node))
    raise ValueError("没有可通行路线；不会生成穿墙直线")


def route_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))
