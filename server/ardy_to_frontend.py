#!/usr/bin/env python3
"""Convert an ARDY motion ``.npz`` into the frontend's motion JSON.

The AMASS path feeds ``retarget.ts`` SMPL local rotations in a **Z-up** world;
ARDY's CoreSkeleton27 output is already three.js-native:

    * ``Y`` is up  (hip height sits on y, ground at y = 0),
    * ``+Z`` is forward (the neutral pose has the toes at +z, and the neutral
      spine/hip axes give ``left x up = +Z``),
    * the neutral pose **is a T-pose** (shoulder -> elbow -> wrist y is
      constant to 1e-7), which is what a VRM humanoid rest pose is too.

so no change of frame is needed -- the JSON just carries ``up: "yup"`` and the
frontend skips its Z-up -> Y-up premultiply.

Joint order is ARDY's ``CoreSkeleton27.bone_order_names_with_parents``; each
joint is renamed to the VRM humanoid bone it drives so that
``VrmRetargeter``'s existing name lookup resolves.  Joints with no VRM
equivalent (hand end / thumb, the intermediate spine links that get folded into
a deeper joint) map to ``""`` and are skipped.

Usage:
    python motion-tools/ardy_to_frontend.py motion-tools/out/ardy_walk_long.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from settings import PROJECT as ROOT, configured_path
OUT_DIR = configured_path('MOTION_ARDY_ASSETS_DIR', 'models/ardy')

# CoreSkeleton27, in ARDY's joint order (source parents, for the FK self-check).
CSKEL27 = [
    # name,            parent index,  VRM humanoid bone ("" = skip)
    ("Hips", -1, "hips"),
    ("Spine", 0, "spine"),
    ("Spine1", 1, ""),          # folded into `chest` via Spine3
    ("Spine2", 2, "chest"),
    ("Spine3", 3, "chest"),     # deepest joint wins in VrmRetargeter
    ("Neck", 4, "neck"),
    ("Head", 5, "head"),
    ("RightShoulder", 4, "rightShoulder"),
    ("RightArm", 7, "rightUpperArm"),
    ("RightForeArm", 8, "rightLowerArm"),
    ("RightHand", 9, "rightHand"),
    ("RightHandEnd", 10, ""),
    ("RightHandThumb1", 10, ""),
    ("LeftShoulder", 4, "leftShoulder"),
    ("LeftArm", 13, "leftUpperArm"),
    ("LeftForeArm", 14, "leftLowerArm"),
    ("LeftHand", 15, "leftHand"),
    ("LeftHandEnd", 16, ""),
    ("LeftHandThumb1", 16, ""),
    ("RightUpLeg", 0, "rightUpperLeg"),
    ("RightLeg", 19, "rightLowerLeg"),
    ("RightFoot", 20, "rightFoot"),
    ("RightToeBase", 21, "rightToes"),
    ("LeftUpLeg", 0, "leftUpperLeg"),
    ("LeftLeg", 23, "leftLowerLeg"),
    ("LeftFoot", 24, "leftFoot"),
    ("LeftToeBase", 25, "leftToes"),
]

NAMES = [n for n, _, _ in CSKEL27]
PARENTS = [p for _, p, _ in CSKEL27]
VRM_BONES = [b for _, _, b in CSKEL27]
N = len(CSKEL27)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrices -> (...,4) quaternions, w-first (w >= 0)."""
    m = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    trace = m00 + m11 + m22
    q = np.empty((len(m), 4), dtype=np.float64)

    # Shepperd: take whichever of w/x/y/z has the largest magnitude so the
    # square root never loses precision.
    big_w = trace > 0
    big_x = ~big_w & (m00 > m11) & (m00 > m22)
    big_y = ~big_w & ~big_x & (m11 > m22)
    big_z = ~big_w & ~big_x & ~big_y

    s = np.sqrt(np.where(big_w, trace + 1.0, 1.0)) * 2.0
    q[big_w, 0] = 0.25 * s[big_w]
    q[big_w, 1] = (m21 - m12)[big_w] / s[big_w]
    q[big_w, 2] = (m02 - m20)[big_w] / s[big_w]
    q[big_w, 3] = (m10 - m01)[big_w] / s[big_w]

    s = np.sqrt(np.where(big_x, 1.0 + m00 - m11 - m22, 1.0)) * 2.0
    q[big_x, 0] = (m21 - m12)[big_x] / s[big_x]
    q[big_x, 1] = 0.25 * s[big_x]
    q[big_x, 2] = (m01 + m10)[big_x] / s[big_x]
    q[big_x, 3] = (m02 + m20)[big_x] / s[big_x]

    s = np.sqrt(np.where(big_y, 1.0 + m11 - m00 - m22, 1.0)) * 2.0
    q[big_y, 0] = (m02 - m20)[big_y] / s[big_y]
    q[big_y, 1] = (m01 + m10)[big_y] / s[big_y]
    q[big_y, 2] = 0.25 * s[big_y]
    q[big_y, 3] = (m12 + m21)[big_y] / s[big_y]

    s = np.sqrt(np.where(big_z, 1.0 + m22 - m00 - m11, 1.0)) * 2.0
    q[big_z, 0] = (m10 - m01)[big_z] / s[big_z]
    q[big_z, 1] = (m02 + m20)[big_z] / s[big_z]
    q[big_z, 2] = (m12 + m21)[big_z] / s[big_z]
    q[big_z, 3] = 0.25 * s[big_z]

    neg = q[:, 0] < 0
    q[neg] *= -1
    return q.reshape(R.shape[:-2] + (4,))


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """(...,4) unit quaternions (w-first) -> (...,3) axis-angle."""
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arctan2(s, w)
    # small angle: aa = 2v exactly (axis * angle = v/s * 2s)
    scale = np.where(s > 1e-8, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale[..., None]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(...,4) unit quaternions (w-first) -> (...,3,3) rotation matrices."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def fk(local: np.ndarray, parents: list[int]) -> np.ndarray:
    """(...,J,3,3) local rotations -> (...,J,3,3) world rotations."""
    out = np.empty_like(local)
    out[..., 0, :, :] = local[..., 0, :, :]
    for j in range(1, local.shape[-3]):
        out[..., j, :, :] = out[..., parents[j], :, :] @ local[..., j, :, :]
    return out


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc slerp between (...,4) unit quaternions (w-first)."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)  # take the short way
    dot = np.abs(np.sum(a * b, axis=-1, keepdims=True)).clip(1.0 - 1e-7, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    w0 = np.where(s < 1e-6, 1.0 - t, np.sin((1.0 - t) * theta) / s)
    w1 = np.where(s < 1e-6, t, np.sin(t * theta) / s)
    out = a * w0 + b * w1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def find_loop_point(
    local: np.ndarray, root: np.ndarray, root_rot: np.ndarray, fps: int, ref: int
) -> int:
    """Last frame the clip should play before wrapping back to ``ref + 1``.

    The generated path is a circle that drifts a little, so wrapping at the raw
    end would teleport the avatar.  Search the tail of the clip for the frame
    whose position, heading and pose are all closest to the frame that the
    cross-fade lands on -- the fade only has to absorb whatever is left.
    """
    t_len = local.shape[0]
    lo = int(t_len * 0.6)
    fwd = np.einsum("nij,j->ni", root_rot, np.array([0.0, 0.0, 1.0]))[:, [0, 2]]
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)

    def pose_gap(a: np.ndarray, b: np.ndarray) -> float:
        r = a @ np.swapaxes(b, -1, -2)
        tr = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
        return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean())

    best, best_cost = t_len - 1, float("inf")
    for f in range(max(lo, ref + 1), t_len):
        pos = float(np.linalg.norm(root[f, [0, 2]] - root[ref, [0, 2]]))
        heading = float(
            np.degrees(np.arctan2(fwd[ref, 0] * fwd[f, 1] - fwd[ref, 1] * fwd[f, 0],
                                  np.dot(fwd[ref], fwd[f])))
        )
        cost = pos + 0.02 * abs(heading) + 0.02 * pose_gap(local[f], local[ref])
        if cost < best_cost:
            best, best_cost = f, cost
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path, help="ARDY output .npz")
    ap.add_argument("--out", type=Path, default=None, help="output JSON (default public/ardy/<stem>.json)")
    ap.add_argument("--manifest", type=Path, default=OUT_DIR / "manifest.json")
    ap.add_argument("--round", type=int, default=6, help="decimal places")
    ap.add_argument("--loop-at", type=int, default=-1,
                    help="last frame to play before wrapping (-1 = auto, i.e. the frame closest to the start)")
    ap.add_argument("--loop-fade", type=int, default=15, help="frames of cross-fade at the seam (0 = off)")
    ap.add_argument("--no-loop", action="store_true", help="emit the raw clip, no trim / no cross-fade")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    local = np.asarray(data["local_rot_mats"], dtype=np.float64)
    root = np.asarray(data["root_positions"], dtype=np.float64)
    fps = int(data["fps"])
    text = str(data["text"]) if "text" in data.files else args.npz.stem

    if local.shape[1] != N:
        raise SystemExit(f"expected {N} joints (CoreSkeleton27), got {local.shape[1]}")

    # Self-check: the world rotations ARDY saved must be the FK of the locals.
    # A mismatch here would mean the skeleton table above is wrong.
    if "global_rot_mats" in data.files:
        want = np.asarray(data["global_rot_mats"], dtype=np.float64)
        got = fk(local, PARENTS)
        err = float(np.abs(want - got).max())
        if err > 1e-4:
            raise SystemExit(f"FK self-check failed: max |global - FK(local)| = {err:g}")
        print(f"FK self-check ok (max err {err:.2e})")

    total = local.shape[0]
    if not args.no_loop:
        fade = max(0, args.loop_fade)
        # Loop plan: play frames [fade .. last], cross-fade the final `fade`
        # frames into frames [0 .. fade-1], then wrap to `fade`.  The frame that
        # follows the fade's last frame is therefore the wrap target itself, so
        # the seam is continuous by construction and the fade only has to absorb
        # the drift between the tail of the clip and its head.
        last = args.loop_at if args.loop_at >= 0 else find_loop_point(
            local, root, np.asarray(data["global_rot_mats"], dtype=np.float64)[:, 0],
            fps, ref=fade - 1,
        )
        if last < 2 * fade - 1:
            raise SystemExit(f"clip too short to cross-fade: last={last}, fade={fade}")
        if last >= total:
            raise SystemExit(f"--loop-at {last} is past the end of the clip ({total} frames)")

        gap = float(np.linalg.norm(root[last, [0, 2]] - root[fade - 1, [0, 2]]))
        print(f"loop: play frames {fade}..{last} ({(last - fade + 1) / fps:.1f}s), "
              f"tail->head gap {gap:.3f} m, fade {fade} frames ({fade / fps:.2f}s)")

        q = rotmat_to_quat(local)
        for k in range(fade):
            i = last - fade + 1 + k
            a = (k + 1) / fade
            local[i] = quat_to_rotmat(quat_slerp(q[i], q[k], a))
            root[i] = root[i] * (1 - a) + root[k] * a
        local = local[fade : last + 1]
        root = root[fade : last + 1]
        # The seam is now between the last kept frame and index 0 (original
        # frame `fade`), i.e. two adjacent frames of the source clip.
        seam_pos = float(np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]))
        rr = local[-1] @ np.swapaxes(local[0], -1, -2)
        tr = rr[:, 0, 0] + rr[:, 1, 1] + rr[:, 2, 2]
        seam_pose = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean())
        print(f"loop self-check at the seam: root gap {seam_pos:.4f} m, "
              f"mean joint delta {seam_pose:.1f}° (one 1/{fps}s step)")

    frames = local.shape[0]
    aa = quat_to_axis_angle(rotmat_to_quat(local)).reshape(frames * N * 3)
    # root motion relative to the (new) frame 0: the app puts it straight on
    # avatarRoot, whose origin already sits on the VRM's feet.
    root_delta = (root - root[0]).reshape(frames * 3)


    nd = args.round
    motion = {
        "version": 1,
        "name": args.npz.stem,
        "source": "ardy",
        "dataset": f"ardy/{text}",
        "up": "yup",
        "fps": fps,
        "frames": frames,
        "duration": round(frames / fps, 3),
        "bones": VRM_BONES,
        "parents": PARENTS,
        "localRot": [round(float(v), nd) for v in aa],
        "rootDelta": [round(float(v), nd) for v in root_delta],
        "prompt": text,
    }

    out = args.out or (OUT_DIR / f"{args.npz.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(motion, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {frames} frames @ {fps} fps)")

    # Upsert into the ARDY manifest alongside the AMASS one.
    entry = {
        "id": args.npz.stem,
        "file": out.name,
        "name": args.npz.stem,
        "dataset": f"ARDY · {text}",
        "fps": fps,
        "frames": frames,
        "duration": motion["duration"],
        "bytes": out.stat().st_size,
        "prompt": text,
    }
    entries = json.loads(args.manifest.read_text()) if args.manifest.exists() else []
    # Drop entries whose file is gone (e.g. an earlier smoke-test output).
    entries = [e for e in entries if (args.manifest.parent / str(e.get("file", ""))).exists()]
    entries = [e for e in entries if e.get("id") != entry["id"]]
    entries.append(entry)
    entries.sort(key=lambda e: e["id"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"manifest {args.manifest}: {len(entries)} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
