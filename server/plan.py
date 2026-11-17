#!/usr/bin/env python3
"""High-level motion authoring: navigation, actions, intents, keyframes.

This is the layer the MCP tools call.  It turns intent ("go there, via here,
at this speed, avoid that box") and keyframes ("at frame 40 put the right hand
at (x,y,z)") into ARDY constraint JSON via `core`.

Everything resolves to ARDY's four constraint kinds:
    root2d / end-effector (root or a limb) / fullbody (positions only)
so the "solver" is really a uniform keyframe -> constraint compiler, not a
separate IK stack.  Single points use `core.single_point_constraints`' analytic
two-bone IK; full-body points use `core.fullbody_constraints`.
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

FPS = core._FPS
_STANDING_Y = 0.92  # hip height of a standing ARDY character

# --- action library: name -> (prompt, optional root-height override) ---------
# `root_y` overrides the hip height (crouch/crawl/lie live low to the ground).
ACTIONS: dict[str, tuple[str, float | None]] = {
    "walk": ("A person walks forward.", None),
    "run": ("A person runs.", None),
    "crouch": ("A person crouches down and stays low.", 0.45),
    "squat": ("A person squats down.", 0.55),
    "crawl": ("A person crawls forward on hands and knees.", 0.32),
    "lie": ("A person lies down on the ground.", 0.15),
    "side_lie": ("A person lies on their side.", 0.16),
    "sit": ("A person sits down.", 0.55),
    "roll": ("A person rolls on the ground.", 0.16),
    "wave": ("A person waves their hand.", None),
    "point": ("A person points forward with their right arm.", None),
    "touch": ("A person touches something with their right hand.", None),
    "raise": ("A person raises their right arm.", None),
}


# --- obstacle-aware navigation -------------------------------------------------

def plan_path(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]] | None,
    cell: float = 0.25,
    margin: float = 0.45,
) -> list[tuple[float, float]] | None:
    """A* over a 2D grid, avoiding axis-aligned box obstacles.

    ``obstacles`` is a list of ``(cx, cz, half_x, half_z)`` boxes (XZ plane).
    Returns a list of ``(x, z)`` waypoints from start to end, or ``None``.
    """
    obs = obstacles or []
    xs = [start[0], end[0]] + [o[0] - o[2] - margin for o in obs] + [o[0] + o[2] + margin for o in obs]
    zs = [start[1], end[1]] + [o[1] - o[3] - margin for o in obs] + [o[1] + o[3] + margin for o in obs]
    x0, x1 = min(xs) - 1.0, max(xs) + 1.0
    z0, z1 = min(zs) - 1.0, max(zs) + 1.0
    W = int((x1 - x0) // cell) + 1
    H = int((z1 - z0) // cell) + 1
    if W > 400 or H > 400:
        # refuse pathological grids
        return None

    blocked = np.zeros((H, W), dtype=bool)
    for (cx, cz, hx, hz) in obs:
        gi0 = max(0, int((cx - hx - margin - x0) // cell))
        gi1 = min(W - 1, int((cx + hx + margin - x0) // cell))
        gj0 = max(0, int((cz - hz - margin - z0) // cell))
        gj1 = min(H - 1, int((cz + hz + margin - z0) // cell))
        blocked[gj0 : gj1 + 1, gi0 : gi1 + 1] = True

    def to_grid(x: float, z: float) -> tuple[int, int]:
        return int((x - x0) // cell), int((z - z0) // cell)

    gs = to_grid(start[0], start[1])
    ge = to_grid(end[0], end[1])
    if blocked[gs[1], gs[0]] or blocked[ge[1], ge[0]]:
        return None

    def h(g: tuple[int, int]) -> float:
        return float(np.hypot(g[0] - ge[0], g[1] - ge[1]))

    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, gs)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], float] = {gs: 0.0}
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == ge:
            break
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ng = (cur[0] + dx, cur[1] + dz)
            if not (0 <= ng[0] < W and 0 <= ng[1] < H) or blocked[ng[1], ng[0]]:
                continue
            cost = gscore[cur] + (1.0 if dx == 0 or dz == 0 else 1.414)
            if cost < gscore.get(ng, float("inf")):
                gscore[ng] = cost
                came[ng] = cur
                heapq.heappush(open_heap, (cost + h(ng), ng))
    if ge not in came and ge != gs:
        return None
    path = [ge]
    while path[-1] in came:
        path.append(came[path[-1]])
    path.reverse()
    return [(x0 + g[0] * cell, z0 + g[1] * cell) for g in path]


def navigate_keyframes(
    waypoints: list[tuple[float, float, float]],
    speed: float,
    obstacles: list[tuple[float, float, float, float]] | None = None,
    *,
    fps: int = FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root keyframes following a path through ``waypoints`` at ``speed`` m/s.

    Each leg between consecutive waypoints is routed around ``obstacles`` with
    A*; the root height interpolates linearly between the legs' heights, and the
    heading always faces the local path direction.
    """
    pts: list[tuple[float, float]] = []
    heights: list[float] = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        leg = plan_path((a[0], a[2]), (b[0], b[2]), obstacles) or [(a[0], a[2]), (b[0], b[2])]
        if pts and leg and (abs(leg[0][0] - pts[-1][0]) < 1e-6 and abs(leg[0][1] - pts[-1][1]) < 1e-6):
            leg = leg[1:]
        pts.extend(leg)
        # height per leg point: interpolate a[1] -> b[1]
        heights.extend(np.linspace(a[1], b[1], max(len(leg), 1)).tolist())
    if not pts:
        pts = [(waypoints[0][0], waypoints[0][2])]
        heights = [waypoints[0][1]]

    # arc-length -> frame index at `speed`
    seg = np.asarray([np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    frame = (cum / max(speed, 1e-3) * fps).astype(int)
    frame = np.minimum(frame, frame[-1])

    positions = np.stack([np.asarray(pts)[:, 0], np.asarray(heights), np.asarray(pts)[:, 1]], axis=1)
    headings = np.empty(len(pts))
    for i in range(len(pts)):
        j = min(i + 1, len(pts) - 1)
        k = max(i - 1, 0)
        headings[i] = core._heading_for_direction(pts[j][0] - pts[k][0], pts[j][1] - pts[k][1])
    return frame, positions, headings


# --- intents: face / touch / point -------------------------------------------

def face_keyframe(frame: int, root: tuple[float, float, float], target: tuple[float, float, float]) -> dict:
    """Turn to face a world point (or another character)."""
    heading = core._heading_for_direction(target[0] - root[0], target[2] - root[2])
    return {"frame": frame, "root": list(root), "heading": float(heading)}


def touch_keyframe(
    frame: int, position: tuple[float, float, float], *, joint: str = "right-hand",
    root: tuple[float, float, float] = (0.0, _STANDING_Y, 0.0), heading: float = 0.0,
) -> dict:
    """Reach a limb out to touch a world position."""
    return {"frame": frame, "root": list(root), "heading": float(heading), "joints": [(joint, list(position))]}


def point_keyframe(
    frame: int, direction: tuple[float, float, float], *, joint: str = "right-hand",
    root: tuple[float, float, float] = (0.0, _STANDING_Y, 0.0), heading: float = 0.0,
) -> dict:
    """Point a limb along a direction: target = root + dir * max-reach."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d) or 1.0
    reach = {"right-hand": 0.44, "left-hand": 0.44, "right-foot": 0.85, "left-foot": 0.85}[joint]
    target = np.asarray(root, float) + d * reach
    return {"frame": frame, "root": list(root), "heading": float(heading), "joints": [(joint, target.tolist())]}


# --- the keyframe solver -------------------------------------------------------

def solve_keyframes(
    keyframes: list[dict],
    *,
    default_root: tuple[float, float, float] = (0.0, _STANDING_Y, 0.0),
    default_heading: float = 0.0,
) -> list[dict]:
    """Compile a list of keyframes into ARDY constraint JSON.

    A keyframe is a dict with ``frame`` and any of:

    * ``root`` + ``heading``          -- put the hips at a world point.
    * ``joints``: [(joint, [x,y,z])]  -- partial world points, solved by IK.
    * ``fullbody``                    -- a full SMPL->ARDY pose at that frame.

    Frames of the same kind are merged into one constraint object (ARDY groups
    by type), and the whole thing is the "full" keyframe solve the MCP exposes.
    """
    merged: dict[str, dict] = {}

    def add(type_: str, entry: dict) -> None:
        if type_ not in merged:
            merged[type_] = {"type": type_, "frame_indices": [], "root_positions": [], "local_joints_rot": []}
        m = merged[type_]
        if "joint_names" in entry:
            names = entry["joint_names"]
            if "joint_names" in m and m["joint_names"] != names:
                raise ValueError(f"incompatible joint sets for {type_}: {m['joint_names']} vs {names}")
            m["joint_names"] = names
        m["frame_indices"].extend(entry["frame_indices"])
        m["root_positions"].extend(entry["root_positions"])
        m["local_joints_rot"].extend(entry["local_joints_rot"])

    for kf in keyframes:
        frame = int(kf["frame"])
        root = tuple(kf.get("root", list(default_root)))
        heading = float(kf.get("heading", default_heading))

        if "fullbody" in kf:
            fb = kf["fullbody"]
            add("fullbody", {
                "frame_indices": [frame],
                "root_positions": [list(fb["root_positions"])],
                "local_joints_rot": [fb["local_joints_rot"]],
            })
            continue

        if "joints" in kf:
            for joint, xyz in kf["joints"]:
                c = core.single_point_constraints([(frame, joint, tuple(xyz))], root=root, heading=heading)[0]
                add(c["type"], c)
        elif "root" in kf:
            c = core.root_track_json(np.array([frame]), np.array([list(root)]), np.array([heading]))[0]
            add(c["type"], c)
        elif "heading" in kf:
            # heading only: same as root keyframe at the default root
            c = core.root_track_json(np.array([frame]), np.array([list(root)]), np.array([heading]))[0]
            add(c["type"], c)

    return list(merged.values())
