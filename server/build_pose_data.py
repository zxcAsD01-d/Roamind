#!/usr/bin/env python3
"""Turn AMASS SMPL npz into HumanML3D-style ``pose_data`` joints for TMR.

Replaces `raw_pose_processing.ipynb` from EricGuo5513/HumanML3D, which is the
only piece of that repo TMR actually needs -- and which is built on
`human_body_prior.BodyModel` over the *gendered SMPL+H* models (registration
gated). This does the same job with the neutral SMPL we already have:

    1. downsample to 20 fps          (``int(fps / 20)``, as upstream)
    2. SMPL forward kinematics       (rigid chain on the rest joints ``J``)
    3. add the global translation
    4. swap Y and Z                  (AMASS is Z-up, HumanML3D wants Y-up)
    5. save ``pose_data/<relpath>.npy``

Why neutral / betas=0 is acceptable
-----------------------------------
TMR's `joints_to_guofeats` ends in ``uniform_skeleton``, which re-fits every
clip onto the canonical skeleton from ``skeleton_example_h3d.npy`` (leg-length
scaling + IK/FK). Proportions are replaced wholesale, so per-subject shape
parameters barely survive into the 263-dim feature. Only pose directions and
root motion matter -- and those come from ``poses`` / ``trans`` either way.

The pkl is the original SMPL release: everything needed here (``J``,
``kintree_table``) is plain numpy, so we unpickle with a stub that stands in for
the one chumpy field (``shapedirs``) instead of installing the abandoned
``chumpy`` package.

Usage:
    python motion-tools/build_pose_data.py                 # whatever npz exist
    python motion-tools/build_pose_data.py --manifest ...  # only TMR's paths
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from settings import PROJECT as ROOT, MOTION_TOOLS, configured_path
AMASS_RAW = configured_path('MOTION_AMASS_DIR', 'datasets/amass/raw')
POSE_DATA = MOTION_TOOLS / 'TMR/datasets/motions/pose_data'
MANIFEST = MOTION_TOOLS / 'pose_data_manifest.json'
SMPL_PKL = configured_path('SMPL_MODEL_PATH', 'models/smpl/SMPL_NEUTRAL.pkl')

TARGET_FPS = 20
N_JOINTS = 24


class _Stub:
    """Stand-in for chumpy objects: keeps whatever state the pickle carries."""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: ANN001
        if module.startswith("chumpy") or module.startswith("ch."):
            return type(f"{module}.{name}", (_Stub,), {})
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001 - tolerate anything else exotic
            return type(f"{module}.{name}", (_Stub,), {})


def load_smpl_rest(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rest joint positions ``J`` (24,3) and parent index array (24,) with root=-1."""
    return _load_smpl_rest_cached(str(pkl_path))


@lru_cache(maxsize=4)
def _load_smpl_rest_cached(pkl_key: str) -> tuple[np.ndarray, np.ndarray]:
    # Cached per worker process: unpickling the 39 MB pkl once per clip would
    # dominate the runtime (10k+ clips).
    with open(pkl_key, "rb") as fh:
        data = _SMPLUnpickler(fh, encoding="latin1").load()
    joints = np.asarray(data["J"], dtype=np.float64)
    tree = np.asarray(data["kintree_table"], dtype=np.int64)
    parents = tree[0].copy()
    parents[parents >= 0x7FFFFFFF] = -1  # 4294967295 -> "no parent"
    if joints.shape[0] != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} rest joints, got {joints.shape}")
    return joints, parents


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(...,3) axis-angle -> (...,3,3) rotation matrices (Rodrigues)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-8), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), K.shape)
    a = angle[..., 0]
    return eye + np.sin(a)[..., None, None] * K + (1.0 - np.cos(a))[..., None, None] * (K @ K)


def forward_kinematics(rest_j: np.ndarray, parents: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Rigid SMPL chain -> posed joint positions.

    Mirrors ``smplx.lbs.batch_rigid_transform``: relative offsets from the rest
    joints, then ``G_child = G_parent @ [R | J_child - J_parent]``.
    """
    rel = rest_j.copy()
    rel[1:] -= rest_j[parents[1:]]

    t = rot.shape[0]
    m = np.zeros((t, N_JOINTS, 4, 4), dtype=np.float64)
    m[..., :3, :3] = rot
    m[..., :3, 3] = rel[None]
    m[..., 3, 3] = 1.0

    chain = [m[:, 0]]
    for i in range(1, N_JOINTS):
        chain.append(chain[parents[i]] @ m[:, i])
    posed = np.stack(chain, axis=1)[..., :3, 3]
    return posed + trans[:, None, :]


def amass_fps(data) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in data:
            return float(data[key])
    raise KeyError("no framerate field")


def process_one(args: tuple[str, str, str]) -> tuple[str, str, int]:
    npz_rel, npz_root, out_root = args
    src = Path(npz_root) / (npz_rel + ".npz")
    dst = Path(out_root) / (npz_rel + ".npy")
    if dst.exists():
        return npz_rel, "skip", 0
    if not src.exists():
        return npz_rel, "missing", 0
    try:
        data = np.load(src, allow_pickle=True)
        fps = amass_fps(data)
        step = max(1, int(fps / TARGET_FPS))
        poses = data["poses"][::step]
        trans = data["trans"][::step]
        if len(poses) == 0:
            return npz_rel, "empty", 0

        rest_j, parents = load_smpl_rest(SMPL_PKL)  # cheap: cached by OS page cache
        rot = axis_angle_to_matrix(poses[:, : N_JOINTS * 3].reshape(-1, N_JOINTS, 3))
        joints = forward_kinematics(rest_j, parents, rot, trans.astype(np.float64))

        # AMASS is Z-up, HumanML3D/TMR want Y-up: (x, y, z) -> (x, z, y)
        joints = joints[..., [0, 2, 1]]

        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, joints.astype(np.float32))
        return npz_rel, "ok", len(joints)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        return npz_rel, f"error {type(exc).__name__}: {exc}", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="list of relative paths TMR needs")
    ap.add_argument("--raw", type=Path, default=AMASS_RAW)
    ap.add_argument("--out", type=Path, default=POSE_DATA)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rels = json.loads(args.manifest.read_text()) if args.manifest.exists() else []
    if not rels:
        rels = sorted(str(p.relative_to(args.raw))[: -len(".npz")] for p in args.raw.rglob("*.npz"))
    if args.limit:
        rels = rels[: args.limit]

    print(f"{len(rels)} sequence(s) -> {args.out}", flush=True)
    t0 = time.time()
    counts: dict[str, int] = {}
    done = 0
    tasks = [(r, str(args.raw), str(args.out)) for r in rels]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # submit()+as_completed() so progress reflects completion order; the
        # Executor.map() convenience API yields results, not futures.
        futs = [pool.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            rel, status, n = fut.result()
            kind = status.split(" ")[0]
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in ("ok", "skip"):
                print(f"  {kind}: {rel} ({status})", flush=True)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{done}/{len(tasks)}] {counts} {el:.0f}s", flush=True)

    print(f"done in {time.time()-t0:.0f}s: {counts}")
    return 1 if counts.get("missing") else 0


if __name__ == "__main__":
    sys.exit(main())
