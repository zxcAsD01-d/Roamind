#!/usr/bin/env python3
"""Constraint-point system shared by the HTTP API and the MCP server.

The whole feature is built on ARDY's *node-level* constraints
(``ardy/ardy/constraints.py``):

* ``root2d``        -- observe the root ``(x, z)`` at chosen frames
* ``end-effector``  -- observe a subset of joints (``joint_names``) at chosen
  frames; with ``["Hips"]`` it pins root position **and root y (height)** **and
  heading**, leaving the limbs free.  That is exactly the "root-height" move.
* ``right-hand`` / ``left-hand`` / ``...`` -- observe an end effector (hand or
  foot) plus the hips; used to make ARDY reach for a point on another character.

Because the constraint files are plain JSON produced *here*, the "system" is
just: a set of ``ConstraintTrack``s (frames + world points + headings), an
SMPL->ARDY pose bridge, and serializers that emit the JSON ``generate.py
--constraints`` understands.

Coordinate frame throughout this module is ARDY's: **Y-up, +Z forward, right
hand at -X**, the same frame three.js / the frontend use.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from functools import lru_cache
from settings import PROJECT as ROOT, MOTION_TOOLS, configured_path
import build_pose_data as _bpd
import ardy_to_frontend as _atf

axis_angle_to_matrix = _bpd.axis_angle_to_matrix
rotmat_to_quat = _atf.rotmat_to_quat
quat_to_axis_angle = _atf.quat_to_axis_angle
ARDY_PARENTS = _atf.PARENTS
ARDY_NEUTRAL = np.asarray(json.loads((ROOT / 'server/assets/core-rest.json').read_text()))


@lru_cache(maxsize=1)
def smpl_rest():
    path = configured_path('SMPL_MODEL_PATH', 'models/smpl/SMPL_NEUTRAL.pkl')
    if not path.is_file():
        raise ValueError('SMPL_MODEL_PATH is required for SMPL/AMASS conversion')
    return _bpd.load_smpl_rest(path)


# AMASS stores SMPL in a Z-up world; ARDY is Y-up. The proper (handedness
# preserving) change of frame is a -90 degree rotation about X.
_A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # Rx(-90deg)
_FPS = 20  # ARDY-Core-RP-20FPS-Horizon40

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# SMPL joint index -> ARDY CoreSkeleton27 joint index (unlisted ARDY joints keep
# the identity local rotation: Spine2, HandEnd, Thumb1 on both sides).
SMPL_TO_ARDY = {
    0: 0,    # pelvis -> Hips
    1: 23,   # left_hip -> LeftUpLeg
    2: 19,   # right_hip -> RightUpLeg
    3: 1,    # spine1 -> Spine
    4: 24,   # left_knee -> LeftLeg
    5: 20,   # right_knee -> RightLeg
    6: 2,    # spine2 -> Spine1
    7: 25,   # left_ankle -> LeftFoot
    8: 21,   # right_ankle -> RightFoot
    9: 4,    # spine3 -> Spine3
    10: 26,  # left_foot -> LeftToeBase
    11: 22,  # right_foot -> RightToeBase
    12: 5,   # neck -> Neck
    13: 13,  # left_collar -> LeftShoulder
    14: 7,   # right_collar -> RightShoulder
    15: 6,   # head -> Head
    16: 14,  # left_shoulder -> LeftArm
    17: 8,   # right_shoulder -> RightArm
    18: 15,  # left_elbow -> LeftForeArm
    19: 9,   # right_elbow -> RightForeArm
    20: 16,  # left_wrist -> LeftHand
    21: 10,  # right_wrist -> RightHand
    # 22 left_hand, 23 right_hand: no ARDY equivalent -> left identity
}
N_ARDY = 27
N_SMPL = 24


def smpl_axis_angle_to_ardy(poses: np.ndarray) -> np.ndarray:
    """SMPL axis-angle ``poses`` (..., 72) -> ARDY local axis-angle (..., 27, 3).

    ``poses`` are AMASS SMPL-H per-joint axis-angle (first 24 joints, 72 dims).
    Returns the ARDY local rotations in ARDY's **Y-up** frame, ready for
    ``EndEffectorConstraintSet.from_dict``.
    """
    _, SMPL_PARENTS = smpl_rest()
    t = poses.shape[0]
    # SMPL world rotations in the Z-up AMASS frame.
    r_local = axis_angle_to_matrix(poses[:, : N_SMPL * 3].reshape(t, N_SMPL, 3))
    w = np.empty_like(r_local)
    w[:, 0] = r_local[:, 0]
    for j in range(1, N_SMPL):
        w[:, j] = w[:, SMPL_PARENTS[j]] @ r_local[:, j]
    # Into ARDY's Y-up frame: premultiply every world rotation by A.
    w_y = _A[None, None] @ w  # (t, 24, 3, 3)

    # ARDY world rotation per joint (identity where unmapped).
    w_ardy = np.broadcast_to(np.eye(3), (t, N_ARDY, 3, 3)).copy()
    for smpl_j, ardy_j in SMPL_TO_ARDY.items():
        w_ardy[:, ardy_j] = w_y[:, smpl_j]

    # World -> local along ARDY's tree (parents always precede children).
    local = np.empty_like(w_ardy)
    local[:, 0] = w_ardy[:, 0]
    for j in range(1, N_ARDY):
        local[:, j] = w_ardy[:, ARDY_PARENTS[j]].swapaxes(-1, -2) @ w_ardy[:, j]
    return quat_to_axis_angle(rotmat_to_quat(local)).reshape(t, N_ARDY * 3)


def smpl_joint_trajectory(
    poses: np.ndarray, trans: np.ndarray, joint: int | str
) -> np.ndarray:
    """World ``(x, y, z)`` of one SMPL joint over time, in ARDY's Y-up frame."""
    SMPL_REST, SMPL_PARENTS = smpl_rest()
    j = SMPL_JOINT_NAMES.index(joint) if isinstance(joint, str) else int(joint)
    r_local = axis_angle_to_matrix(poses[:, : N_SMPL * 3].reshape(poses.shape[0], N_SMPL, 3))
    w = np.empty_like(r_local)
    w[:, 0] = r_local[:, 0]
    for i in range(1, N_SMPL):
        w[:, i] = w[:, SMPL_PARENTS[i]] @ r_local[:, i]
    pos = np.empty((poses.shape[0], N_SMPL, 3))
    rel = SMPL_REST.copy()
    rel[1:] -= SMPL_REST[SMPL_PARENTS[1:]]
    pos[:, 0] = SMPL_REST[0] + trans
    for i in range(1, N_SMPL):
        pos[:, i] = pos[:, SMPL_PARENTS[i]] + (w[:, SMPL_PARENTS[i]] @ rel[i])
    return pos[:, j] @ _A.T  # (x, y, z) -> Y-up


def _heading_for_direction(dx: float, dz: float) -> float:
    """Body yaw (rad) that makes the +Z-forward T-pose face ``(dx, dz)``."""
    return math.atan2(dx, dz)


def _yaw_axis_angle(theta: float) -> np.ndarray:
    # rotation of `theta` about +Y (axis-angle)
    return np.array([0.0, theta, 0.0])


def root_track_json(
    frames: np.ndarray, positions: np.ndarray, headings: np.ndarray | None
) -> list[dict]:
    """Serialize a root track as one ``end-effector`` constraint over ``["Hips"]``.

    Each keyframe carries the root position (x, y, z) and a yawed T-pose, which
    ARDY reads as: put the hips here, at this height, facing this way. The legs
    and arms stay free.
    """
    n = len(frames)
    local = np.zeros((n, N_ARDY, 3), dtype=np.float64)
    if headings is None:
        dx = np.gradient(positions[:, 0])
        dz = np.gradient(positions[:, 2])
        headings = np.arctan2(dx, dz)
    for i, th in enumerate(headings):
        local[i, 0] = _yaw_axis_angle(float(th))
    return [
        {
            "type": "end-effector",
            "joint_names": ["Hips"],
            "frame_indices": [int(f) for f in frames],
            "root_positions": [p.tolist() for p in positions],
            "local_joints_rot": [l.tolist() for l in local],
        }
    ]


def two_point_track(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    duration: float,
    *,
    fps: int = _FPS,
    keyframe_every: int = 8,
    steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A root track walking from ``a`` to ``b`` in ``duration`` seconds.

    ``a``/``b`` are ``(x, y, z)`` world points (y = root height). The path is a
    straight line in XZ; the height ramps between the two floor levels over the
    middle 40% of the clip (or as ``steps`` equal risers), so a jump between
    floors reads as a climb instead of a pop.
    """
    frames = int(round(duration * fps))
    t = np.linspace(0.0, 1.0, frames)
    x = a[0] + (b[0] - a[0]) * t
    z = a[2] + (b[2] - a[2]) * t
    if steps:
        # discrete risers: quantize the climb into `steps` equal jumps
        level = np.floor(t * steps) / max(steps, 1)
        y = a[1] + (b[1] - a[1]) * level
    else:
        # smooth ramp over the whole path (a slope walk)
        y = a[1] + (b[1] - a[1]) * t
    positions = np.stack([x, y, z], axis=1)
    headings = np.full(frames, _heading_for_direction(b[0] - a[0], b[2] - a[2]))
    idx = np.arange(0, frames, keyframe_every)
    if idx[-1] != frames - 1:
        idx = np.append(idx, frames - 1)
    return idx, positions[idx], headings[idx]


def stairs_track(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    duration: float,
    *,
    riser: float = 0.25,
    tread: float = 0.5,
    fps: int = _FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A root track that climbs a staircase by following the *ground line*.

    The hip height is held at ``ground(x) + standing_height`` where the ground
    is a step function of the path: flat approach, then ``n = (b.y-a.y)/riser``
    treads each ``tread`` metres deep, then flat exit.  A keyframe is emitted on
    every riser *and* every ``riser/tread`` worth of travel, so the model gets a
    dense, tread-aligned root to follow instead of a vague ramp.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    frames = int(round(duration * fps))
    s = np.linspace(0.0, 1.0, frames)
    x = a[0] + (b[0] - a[0]) * s
    z = a[2] + (b[2] - a[2]) * s

    n = max(1, int(round((b[1] - a[1]) / riser)))  # number of risers
    c0, c1 = 0.2, 0.8  # climb occupies the middle 60% of the path
    ramp = np.clip((s - c0) / (c1 - c0), 0.0, 1.0)
    level = np.floor(ramp * n) / n  # 0 .. 1 in n steps
    y = a[1] + (b[1] - a[1]) * level
    positions = np.stack([x, y, z], axis=1)

    heading = _heading_for_direction(b[0] - a[0], b[2] - a[2])
    # keyframes: every 4 frames + every riser transition + both ends
    keyframe_every = max(1, int(round(fps * tread / 1.4)))  # ~ tread spacing
    kf = {0, frames - 1}
    kf.update(range(0, frames, keyframe_every))
    kf.update(np.flatnonzero(np.diff(level) != 0) + 1)
    idx = np.array(sorted(kf), dtype=int)
    return idx, positions[idx], np.full(len(idx), heading)


def fullbody_constraints(
    poses: np.ndarray,
    trans: np.ndarray,
    src_fps: int,
    duration: float,
    *,
    keyframe_every: int = 8,
) -> list[dict]:
    """Bind ARDY's *entire* skeleton to the other character's pose.

    Emits a single ``fullbody`` constraint, which conditions on **joint
    positions only** (ARDY ignores the rotations -- see ``FullBodyConstraintSet.
    update_constraints``), so ARDY mirrors the other character's body while
    keeping its own joint orientations.
    """
    step = max(1, int(src_fps / _FPS))
    p, tr = poses[::step], trans[::step]
    frames = min(len(p), int(round(duration * _FPS)))
    p, tr = p[:frames], tr[:frames]
    idx = np.arange(0, frames, keyframe_every)
    if idx[-1] != frames - 1:
        idx = np.append(idx, frames - 1)

    local = smpl_axis_angle_to_ardy(p)
    root = smpl_joint_trajectory(p, tr, 0)
    return [
        {
            "type": "fullbody",
            "frame_indices": [int(i) for i in idx],
            "local_joints_rot": [local[i].reshape(N_ARDY, 3).tolist() for i in idx],
            "root_positions": [root[i].tolist() for i in idx],
            "root_2d": [root[i, [0, 2]].tolist() for i in idx],
        }
    ]


def limb_constraints(
    poses: np.ndarray,
    trans: np.ndarray,
    src_fps: int,
    duration: float,
    *,
    joints: tuple[str, ...] = ("right-hand", "left-hand", "right-foot", "left-foot"),
    keyframe_every: int = 8,
) -> list[dict]:
    """Bind a chosen set of ARDY limbs to the other character's limbs.

    One ``<limb>-hand``/``<limb>-foot`` constraint per joint; each pins that end
    effector (position + rotation) plus the hips, so ARDY's hands *and* feet
    track the other character's, leaving the torso free to stay balanced.
    """
    step = max(1, int(src_fps / _FPS))
    p, tr = poses[::step], trans[::step]
    frames = min(len(p), int(round(duration * _FPS)))
    p, tr = p[:frames], tr[:frames]
    idx = np.arange(0, frames, keyframe_every)
    if idx[-1] != frames - 1:
        idx = np.append(idx, frames - 1)

    local = smpl_axis_angle_to_ardy(p)
    root = smpl_joint_trajectory(p, tr, 0)
    return [
        {
            "type": jt,
            "frame_indices": [int(i) for i in idx],
            "root_positions": [root[i].tolist() for i in idx],
            "local_joints_rot": [local[i].reshape(N_ARDY, 3).tolist() for i in idx],
        }
        for jt in joints
    ]


# --------------------------------------------------------------------------- #
# Single-point keyframes: mark ONE joint at ONE world point, no reference pose.
# --------------------------------------------------------------------------- #

# limb chain = (base, middle, end) joint indices in CoreSkeleton27, plus the
# direction the joint bends naturally.  The IK places `end` (wrist / ankle) at
# the target; a hand/foot extends beyond along the chain direction.
_CHAINS = {
    "right-hand": (7, 8, 9, (0.0, -1.0, 0.0)),
    "left-hand": (13, 14, 15, (0.0, -1.0, 0.0)),
    "right-foot": (19, 20, 21, (0.0, 0.0, 1.0)),
    "left-foot": (23, 24, 25, (0.0, 0.0, 1.0)),
}


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit ``u`` onto unit ``v``."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    d = float(np.dot(u, v))
    if d > 1.0 - 1e-8:
        return np.eye(3)
    if d < -1.0 + 1e-8:
        axis = np.cross(u, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return axis_angle_to_matrix(axis[None] * np.pi)[0]
    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    return axis_angle_to_matrix(axis[None] * np.arccos(d))[0]


def _mat_to_aa(R: np.ndarray) -> np.ndarray:
    return quat_to_axis_angle(rotmat_to_quat(R[None]))[0]


def _ik_limb(joint: str, target_local: np.ndarray) -> np.ndarray:
    """Local axis-angle (27,3) that puts `joint`'s end at `target_local` (body frame)."""
    base, mid, end, bend = _CHAINS[joint]
    base_pos = ARDY_NEUTRAL[base]
    l1 = float(np.linalg.norm(ARDY_NEUTRAL[mid] - ARDY_NEUTRAL[base]))
    l2 = float(np.linalg.norm(ARDY_NEUTRAL[end] - ARDY_NEUTRAL[mid]))

    v = target_local - base_pos
    dist = float(np.linalg.norm(v))
    dist = float(np.clip(dist, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    v = v / max(np.linalg.norm(v), 1e-9)

    # elbow/knee position in the plane spanned by (v, bend direction)
    b = np.asarray(bend, float)
    b -= v * float(np.dot(b, v))
    if np.linalg.norm(b) < 1e-6:
        b = np.array([0.0, 0.0, 1.0])
        b -= v * float(np.dot(b, v))
    b /= np.linalg.norm(b)
    cos_a = float(np.clip((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1.0, 1.0))
    a = np.arccos(cos_a)
    elbow = base_pos + l1 * (v * cos_a + b * np.sin(a))
    end_pos = base_pos + v * dist

    u_rest = (ARDY_NEUTRAL[mid] - ARDY_NEUTRAL[base]) / l1
    f_rest = (ARDY_NEUTRAL[end] - ARDY_NEUTRAL[mid]) / l2
    R_base = _rot_between(u_rest, (elbow - base_pos) / l1)
    # the mid joint's *local* rotation is expressed in the frame R_base already
    # set, so the forearm's target world direction must be pulled back through it
    R_mid = _rot_between(f_rest, R_base.T @ ((end_pos - elbow) / l2))

    local = np.zeros((N_ARDY, 3), dtype=np.float64)
    local[base] = _mat_to_aa(R_base)
    local[mid] = _mat_to_aa(R_mid)
    return local


def single_point_constraints(
    points: list[tuple[int, str, tuple[float, float, float]]],
    *,
    root: tuple[float, float, float],
    heading: float = 0.0,
) -> list[dict]:
    """One keyframe per *single* point: mark a joint at a world position.

    ``points`` is a list of ``(frame, joint, (x, y, z))``; ``joint`` is one of
    ``"right-hand" / "left-hand" / "right-foot" / "left-foot"``.  Each point
    becomes a separate ``end-effector`` constraint built by analytic two-bone
    IK, so no reference pose is needed -- the rest of the body is a yawed
    T-pose and only the marked limb reaches for the point.
    """
    # world -> body frame: undo the root yaw and root translation
    ry = axis_angle_to_matrix(np.array([[0.0, heading, 0.0]]))[0]
    ry_inv = ry.T
    out = []
    for frame, joint, xyz in points:
        xyz = np.asarray(xyz, float)
        target_local = ry_inv @ (xyz - np.asarray(root, float))
        local = _ik_limb(joint, target_local)
        local[0] = np.array([0.0, heading, 0.0])  # root yaw
        out.append(
            {
                "type": joint,
                "frame_indices": [int(frame)],
                "root_positions": [list(root)],
                "local_joints_rot": [local.tolist()],
            }
        )
    return out


def smpl_follow_track(
    poses: np.ndarray,
    trans: np.ndarray,
    src_fps: int,
    joint: str | int,
    duration: float,
    *,
    keyframe_every: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root track that follows the world point of one SMPL joint.

    Resamples the source motion to ``fps``, extracts ``joint``'s world position,
    and heads the avatar along its own velocity.
    """
    src = poses.shape[0]
    fps = _FPS
    step = max(1, int(src_fps / fps))
    pos = smpl_joint_trajectory(poses[::step], trans[::step], joint)
    frames = min(len(pos), int(round(duration * fps)))
    pos = pos[:frames]
    if joint in (0, "pelvis"):
        # following the pelvis: lower the root to standing height so the avatar
        # doesn't float at hip level
        pos = pos.copy()
        pos[:, 1] = pos[:, 1] - float(smpl_rest()[0][0, 1])
    idx = np.arange(0, frames, keyframe_every)
    if idx[-1] != frames - 1:
        idx = np.append(idx, frames - 1)
    return idx, pos[idx], None


def hand_follow_constraints(
    poses: np.ndarray,
    trans: np.ndarray,
    src_fps: int,
    duration: float,
    hand: str = "right",
    *,
    keyframe_every: int = 8,
) -> list[dict]:
    """ARDY's hand follows the other character's moving hand.

    Emits ``right-hand`` / ``left-hand`` end-effector constraints: at each
    keyframe the SMPL pose is re-expressed on ARDY's skeleton, and ARDY is told
    to match that hand + the hips + heading.
    """
    step = max(1, int(src_fps / _FPS))
    p = poses[::step]
    tr = trans[::step]
    frames = min(len(p), int(round(duration * _FPS)))
    p, tr = p[:frames], tr[:frames]
    idx = np.arange(0, frames, keyframe_every)
    if idx[-1] != frames - 1:
        idx = np.append(idx, frames - 1)

    local = smpl_axis_angle_to_ardy(p)  # (frames, 27*3)
    # root world position = pelvis, in Y-up
    root = smpl_joint_trajectory(p, tr, 0)

    kind = f"{hand}-hand"
    return [
        {
            "type": kind,
            "frame_indices": [int(i) for i in idx],
            "root_positions": [root[i].tolist() for i in idx],
            "local_joints_rot": [local[i].reshape(N_ARDY, 3).tolist() for i in idx],
        }
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", nargs=3, type=float, required=True)
    ap.add_argument("--to", nargs=3, type=float, required=True)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", type=Path, default=MOTION_TOOLS / "out" / "two_point_constraints.json")
    args = ap.parse_args()
    idx, pos, head = two_point_track(tuple(args.frm), tuple(args.to), args.duration, steps=args.steps)
    js = root_track_json(idx, pos, head)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(js, indent=2))
    print(f"wrote {args.out}: {len(js)} root keyframes over {args.duration}s")
