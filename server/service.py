#!/usr/bin/env python3
"""ARDY constraint service: HTTP API + MCP interface over ``core.py``.

Two transports, one core:

* ``python service.py http --port 8100``  -- REST API the frontend talks to.
* ``python service.py mcp``               -- an MCP stdio server exposing the
  same operations as tools, so a client (Claude Code) can drive ARDY motion.

Both write generated clips into ``motion-frontend/public/service/``, which the
Vite dev server serves back to the browser as ``/service/*.json`` -- no CORS on
the read path, the browser just fetches the same origin.

Endpoints (HTTP):
    GET  /health
    POST /api/move     {"from":[x,y,z],"to":[x,y,z],"duration":8,"steps":0,"prompt":"..."}
    POST /api/follow   {"npz":"<path>","joint":"pelvis","duration":12,"prompt":"..."}
    POST /api/hand     {"npz":"<path>","hand":"right","duration":12,"prompt":"..."}
    GET  /api/motions
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import traceback
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import plan  # noqa: E402
import prompt  # noqa: E402
from realtime import GenerationCancelled, RealtimeRunner  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "server"))
import control  # noqa: E402
import runtime  # noqa: E402
from dual import DualRunner  # noqa: E402
from ardy_playback import WorldPlayback  # noqa: E402

from settings import PROJECT as ROOT, MOTION_TOOLS, SERVICE_DIR, EMBEDDINGS
VENV_PY = Path(sys.executable)
ARDY_GENERATE = ROOT / 'scripts/ardy_generate.py'
TMR_VENV_PY = Path(os.environ.get('TMR_PYTHON', sys.executable))
RECOGNIZE = MOTION_TOOLS / 'ardy_service/recognize.py'
MANIFEST = SERVICE_DIR / "manifest.json"
GENERATION_LOCK = threading.Lock()
REALTIME_RUNNER_LOCK = threading.Lock()
REALTIME_RUNNER = None
WORLD_PLAYBACK = WorldPlayback()
STREAM_REQUESTS: dict[str, tuple[int, threading.Event]] = {}
STREAM_REQUESTS_LOCK = threading.Lock()


def register_stream(args: dict) -> threading.Event:
    client = args.get("clientId")
    revision = args.get("revision")
    if not isinstance(client, str) or not 1 <= len(client) <= 100:
        raise ValueError("clientId must be a string of 1–100 characters")
    if type(revision) is not int or revision < 1:
        raise ValueError("revision must be a positive integer")
    event = threading.Event()
    with STREAM_REQUESTS_LOCK:
        previous = STREAM_REQUESTS.get(client)
        if previous and revision <= previous[0]:
            event.set()
            return event
        if previous:
            previous[1].set()
        STREAM_REQUESTS[client] = (revision, event)
    return event


def release_stream(args: dict, event: threading.Event) -> None:
    client = args.get("clientId")
    revision = args.get("revision")
    with STREAM_REQUESTS_LOCK:
        current = STREAM_REQUESTS.get(client)
        if current and current[0] == revision and current[1] is event:
            del STREAM_REQUESTS[client]


def get_realtime_runner() -> DualRunner:
    global REALTIME_RUNNER
    with REALTIME_RUNNER_LOCK:
        if REALTIME_RUNNER is None:
            REALTIME_RUNNER = DualRunner(RealtimeRunner)
        return REALTIME_RUNNER


def generate_world_window(plan, start, history, world, cancelled):
    if cancelled():
        raise GenerationCancelled()
    return get_realtime_runner().world_window(plan, start, history, world, cancelled)


def run_stream_generation(args: dict, prompt_text: str, duration: float, emit) -> dict:
    cancelled = register_stream(args)
    try:
        if cancelled.is_set():
            raise GenerationCancelled()
        runner = get_realtime_runner()
        scene_plan = runtime.find_plan(args["planId"]) if args.get("planId") else None
        motion = runner.generate(prompt_text, duration, emit, args.get("history"), cancelled.is_set, scene_plan)
        if cancelled.is_set():
            raise GenerationCancelled()
        return motion
    finally:
        release_stream(args, cancelled)

DEFAULT_PROMPT = "A person walks forward."
CLIMB_PROMPT = "A person climbs stairs."

# --------------------------------------------------------------------------- #
# generation + conversion
# --------------------------------------------------------------------------- #

def convert_npz_to_json(npz: Path, out_json: Path, prompt: str) -> dict:
    """ARDY npz -> frontend motion JSON (raw, no loop cross-fade)."""
    data = np.load(npz, allow_pickle=True)
    local = np.asarray(data["local_rot_mats"], dtype=np.float64)
    root = np.asarray(data["root_positions"], dtype=np.float64)
    fps = int(data["fps"])
    frames = local.shape[0]
    aa = core.quat_to_axis_angle(core.rotmat_to_quat(local)).reshape(frames * core.N_ARDY * 3)
    root_delta = (root - root[0]).reshape(frames * 3)
    motion = {
        "version": 1,
        "name": npz.stem,
        "source": "ardy-service",
        "dataset": f"service · {prompt}",
        "up": "yup",
        "fps": fps,
        "frames": frames,
        "duration": round(frames / fps, 3),
        "bones": core._atf.VRM_BONES,
        "parents": core._atf.PARENTS,
        "localRot": [round(float(v), 6) for v in aa],
        "rootDelta": [round(float(v), 6) for v in root_delta],
        "rootOrigin": root[0].tolist(),
        "prompt": prompt,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(motion, separators=(",", ":")))
    return {
        "id": npz.stem,
        "file": out_json.name,
        "name": npz.stem,
        "dataset": f"service · {prompt}",
        "fps": fps,
        "frames": frames,
        "duration": motion["duration"],
        "bytes": out_json.stat().st_size,
        "prompt": prompt,
    }


def upsert_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    entries = [e for e in entries if (MANIFEST.parent / str(e.get("file", ""))).exists()]
    entries = [e for e in entries if e.get("id") != entry["id"]]
    entries.append(entry)
    MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def run_generate(constraints: list[dict], prompt: str, duration: float, out_npz: Path) -> Path:
    constraints_path = MOTION_TOOLS / "out" / f"{out_npz.stem}_constraints.json"
    constraints_path.parent.mkdir(parents=True, exist_ok=True)
    constraints_path.write_text(json.dumps(constraints, indent=2))

    cmd = [
        str(VENV_PY), str(ARDY_GENERATE), prompt,
        "--model", "core", "--duration", str(duration), "--seed", "0",
        "--device", "auto", "--diffusion_steps", "6",
        "--no-encode", "--constraints", str(constraints_path), "--output", str(out_npz),
    ]
    with GENERATION_LOCK:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"generation failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return out_npz


def ensure_prompt(prompt: str) -> str:
    """Only cached prompts are allowed (the 8B encoder must not load per call)."""
    if not EMBEDDINGS.is_file():
        raise ValueError('ARDY_EMBEDDINGS is missing; configure a trusted prompt cache')
    cache = np.load(EMBEDDINGS, allow_pickle=True)
    known = {str(p) for p in cache["prompts"]}
    if prompt not in known:
        raise ValueError(f"prompt {prompt!r} not cached; cached: {sorted(known)}")
    return prompt


def list_prompts() -> list[str]:
    if not EMBEDDINGS.is_file():
        return []
    cache = np.load(EMBEDDINGS, allow_pickle=True)
    return sorted(str(item) for item in cache["prompts"])


def generate_prompt(args: dict) -> dict:
    """Generate a continuous autoregressive clip for a cached text prompt."""
    prompt = ensure_prompt(str(args.get("prompt", "")))
    duration = float(args.get("duration", 24.0))
    if not math.isfinite(duration) or not 1.0 <= duration <= 60.0:
        raise ValueError("duration must be between 1 and 60 seconds")

    name = f"ardy_{time.time_ns()}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate([], prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", prompt)
    upsert_manifest(entry)
    entry["url"] = f"/files/{entry['file']}"
    return entry


def generate_prompt_stream(args: dict, emit) -> dict:
    """Generate and persist a motion while sending each autoregressive window."""
    prompt_text = ensure_prompt(str(args.get("prompt", "")))
    duration = float(args.get("duration", 24.0))
    if not math.isfinite(duration) or not 1.0 <= duration <= 60.0:
        raise ValueError("duration must be between 1 and 60 seconds")

    if REALTIME_RUNNER is None:
        emit({"type": "status", "message": "首次准备常驻动作模型…"})
    motion = run_stream_generation(args, prompt_text, duration, emit)

    out_json = SERVICE_DIR / f"{motion['id']}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(motion, separators=(",", ":")))
    entry = {
        "id": motion["id"],
        "file": out_json.name,
        "name": motion["name"],
        "dataset": motion["dataset"],
        "fps": motion["fps"],
        "frames": motion["frames"],
        "duration": motion["duration"],
        "bytes": out_json.stat().st_size,
        "prompt": prompt_text,
    }
    upsert_manifest(entry)
    entry["url"] = f"/files/{entry['file']}"
    return entry


def warm_realtime_runner() -> None:
    """Load and warm the autoregressive model before the first browser request."""
    global REALTIME_RUNNER
    try:
        with REALTIME_RUNNER_LOCK:
            if REALTIME_RUNNER is None:
                REALTIME_RUNNER = DualRunner(RealtimeRunner)
            REALTIME_RUNNER.warmup()
        print("ARDY dual overlap models are warm", flush=True)
    except Exception:
        traceback.print_exc()
        with REALTIME_RUNNER_LOCK:
            REALTIME_RUNNER = None


def generate_move(args: dict) -> dict:
    a = tuple(args["from"])
    b = tuple(args["to"])
    duration = float(args.get("duration", 10.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    # Ground-line climb when the two points sit at different heights; a plain
    # walk otherwise.
    if abs(b[1] - a[1]) > 0.1:
        idx, pos, head = core.stairs_track(a, b, duration)
    else:
        idx, pos, head = core.two_point_track(a, b, duration)
    constraints = core.root_track_json(idx, pos, head)
    name = f"move_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", prompt)
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_follow(args: dict) -> dict:
    npz = Path(args["npz"])
    joint = args.get("joint", "pelvis")
    duration = float(args.get("duration", 12.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    d = np.load(npz, allow_pickle=True)
    fps = float(core._bpd.amass_fps(d))
    idx, pos, head = core.smpl_follow_track(d["poses"], d["trans"], fps, joint, duration)
    constraints = core.root_track_json(idx, pos, head)
    name = f"follow_{joint}_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"follow {joint}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_hand(args: dict) -> dict:
    npz = Path(args["npz"])
    hand = args.get("hand", "right")
    duration = float(args.get("duration", 12.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    d = np.load(npz, allow_pickle=True)
    fps = float(core._bpd.amass_fps(d))
    constraints = core.hand_follow_constraints(d["poses"], d["trans"], fps, duration, hand)
    name = f"hand_{hand}_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"{hand}-hand follow")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_bind(args: dict) -> dict:
    """Fine limb interaction: bind ARDY's body/limbs to the other SMPL object.

    ``mode`` is ``fullbody`` (every joint position follows) or ``limbs`` (hands
    and feet follow, torso free).
    """
    npz = Path(args["npz"])
    mode = args.get("mode", "fullbody")
    duration = float(args.get("duration", 10.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    d = np.load(npz, allow_pickle=True)
    fps = float(core._bpd.amass_fps(d))
    if mode == "limbs":
        constraints = core.limb_constraints(d["poses"], d["trans"], fps, duration)
    else:
        constraints = core.fullbody_constraints(d["poses"], d["trans"], fps, duration)
    name = f"bind_{mode}_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"bind {mode}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_point(args: dict) -> dict:
    """Mark a single joint at a single world point (analytic two-bone IK)."""
    points = [(int(p["frame"]), p["joint"], (p["x"], p["y"], p["z"])) for p in args["points"]]
    root = tuple(args.get("root", [0.0, 0.92, 0.0]))
    heading = float(args.get("heading", 0.0))
    duration = float(args.get("duration", 6.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    constraints = core.single_point_constraints(points, root=root, heading=heading)
    name = f"point_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"point {len(points)}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_action(args: dict) -> dict:
    """Play a basic action (crouch / crawl / lie / sit / roll / ...)."""
    name = args["name"]
    prompt, root_y = plan.ACTIONS[name]
    duration = float(args.get("duration", 5.0))
    constraints = []
    if root_y is not None:
        # keep the body at the action's characteristic height
        idx, pos, head = core.two_point_track(
            (0.0, root_y, 0.0), (0.0, root_y, 0.0), duration, steps=0
        )
        constraints = core.root_track_json(idx, pos, head)
    out_npz = MOTION_TOOLS / "out" / f"action_{name}_{int(time.time())}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{out_npz.stem}.json", f"action {name}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_navigate(args: dict) -> dict:
    """Go somewhere, via waypoints, at a speed, routing around obstacles."""
    waypoints = [tuple(w) for w in args["waypoints"]]
    speed = float(args.get("speed", 1.4))
    obstacles = [tuple(o) for o in args.get("obstacles", [])]
    duration = float(args.get("duration", 0))  # 0 = derive from speed
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    idx, pos, head = plan.navigate_keyframes(waypoints, speed, obstacles)
    if duration <= 0:
        duration = float(idx[-1]) / plan.FPS + 1.0
    constraints = core.root_track_json(idx, pos, head)
    name = f"nav_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"navigate {speed}m/s")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_keyframes(args: dict) -> dict:
    """Solve arbitrary keyframes (root / partial points / fullbody) into motion."""
    keyframes = args["keyframes"]
    duration = float(args.get("duration", 10.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    constraints = plan.solve_keyframes(
        keyframes,
        default_root=tuple(args.get("default_root", [0.0, 0.92, 0.0])),
        default_heading=float(args.get("default_heading", 0.0)),
    )
    name = f"kf_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"keyframes x{len(keyframes)}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def generate_intent(args: dict) -> dict:
    """face / touch / point a target from a standing root."""
    kind = args["kind"]  # face | touch | point
    frame = int(args.get("frame", 30))
    duration = float(args.get("duration", 6.0))
    prompt = ensure_prompt(args.get("prompt", DEFAULT_PROMPT))
    root = tuple(args.get("root", [0.0, 0.92, 0.0]))
    if kind == "face":
        kf = plan.face_keyframe(frame, root, tuple(args["target"]))
    elif kind == "touch":
        kf = plan.touch_keyframe(frame, tuple(args["position"]), joint=args.get("joint", "right-hand"), root=root)
    elif kind == "point":
        kf = plan.point_keyframe(frame, tuple(args["direction"]), joint=args.get("joint", "right-hand"), root=root)
    else:
        raise ValueError(f"unknown intent {kind}")
    constraints = plan.solve_keyframes([kf], default_root=root)
    name = f"intent_{kind}_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", f"intent {kind}")
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"
    return entry


def recognize_motion(args: dict) -> dict:
    """Recognise a motion clip with TMR (the motion-recognition model).

    Spawns ``recognize.py`` under the TMR venv and returns the top-K nearest
    text descriptions, so the service itself never imports TMR.
    """
    npz = args["npz"]
    top_k = int(args.get("top_k", 5))
    proc = subprocess.run(
        [str(TMR_VENV_PY), str(RECOGNIZE), "--npz", str(npz), "--top-k", str(top_k)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"recognition failed: {proc.stderr[-800:]}")
    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        score, text = line.split("\t", 1)
        results.append({"text": text, "score": float(score)})
    return {"results": results}


def generate_scene(args: dict) -> dict:
    """Describe a scene in natural language -> structured state -> motion.

    The nearby person's action is resolved from the *cached* action prompts, and
    the direction/distance ground the intent keyframes (face them, touch if they
    are within reach).  The full annotated state is returned for the caller.
    """
    text = args["text"]
    ego_root = tuple(args.get("root", [0.0, 0.92, 0.0]))
    ego_action = args.get("ego_action")  # e.g. "wave" (a plan.ACTIONS key)
    duration = float(args.get("duration", 8.0))
    d = prompt.describe(text, ego_root)
    entities = d["entities"]

    if ego_action:
        gen_prompt = plan.ACTIONS[ego_action][0]
    elif entities:
        gen_prompt = plan.ACTIONS[entities[0]["action_key"]][0]
    else:
        gen_prompt = DEFAULT_PROMPT
    gen_prompt = ensure_prompt(gen_prompt)

    constraints = plan.solve_keyframes(d["keyframes"], default_root=ego_root)
    name = f"scene_{int(time.time())}"
    out_npz = MOTION_TOOLS / "out" / f"{name}.npz"
    run_generate(constraints, gen_prompt, duration, out_npz)
    entry = convert_npz_to_json(out_npz, SERVICE_DIR / f"{name}.json", gen_prompt)
    upsert_manifest(entry)
    entry["url"] = f"/service/{entry['file']}"

    return {
        "state": entities,
        "prompt": prompt.compose_prompt(entities[0], ego_action) if entities else gen_prompt,
        "generation_prompt": gen_prompt,
        "clip": entry,
    }


def list_motions() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not 0 <= n <= 8 * 1024 * 1024:
            raise ValueError('Request body exceeds 8 MiB')
        value = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(value, dict):
            raise ValueError('Request body must be a JSON object')
        return value

    def _stream_json_lines(self, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            if event.get("type") == "chunk":
                runtime.record_metric(event)
            payload = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
            self.wfile.write(payload.encode("utf-8") + b"\n")
            self.wfile.flush()

        try:
            entry = generate_prompt_stream(body, emit)
            emit({"type": "complete", "entry": entry})
        except (GenerationCancelled, BrokenPipeError, ConnectionResetError):
            # Superseded/disconnected streams have no completion and save no clip.
            return
        except Exception as exc:  # noqa: BLE001 - the stream already has HTTP headers
            try:
                emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_OPTIONS(self):  # noqa: N802
        self._send(204, {})

    def do_GET(self):  # noqa: N802
        try:
            result = control.get(self.path)
            if result is not None:
                return self._send(200, result)
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        if self.path == "/health":
            return self._send(200, {"ok": True, "prompt_count": len(list_prompts()),
                                   "engine": REALTIME_RUNNER.info() if REALTIME_RUNNER else {
                                       "mode": "on-demand", "ready": False,
                                       "promptCacheAvailable": EMBEDDINGS.is_file()}})
        if self.path == "/api/prompts":
            return self._send(200, {"prompts": list_prompts()})
        if self.path == "/api/motions":
            return self._send(200, list_motions())
        if self.path.startswith("/files/"):
            # Serve generated clips straight from the service, bypassing the
            # Vite public-dir race (a file written mid-session may not be seen
            # by the dev server on the very next request).
            name = self.path[len("/files/"):]
            if not name.endswith('.json') or Path(name).name != name or '\\' in name:
                return self._send(400, {"error": "Invalid generated clip name"})
            p = SERVICE_DIR / name
            if p.is_symlink():
                return self._send(403, {"error": "Symlink clips are not served"})
            if not p.is_file():
                return self._send(404, {"error": f"no such file: {name}"})
            body = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            body = self._read()
            if self.path.startswith('/api/ardy/world/'):
                operation = self.path.removeprefix('/api/ardy/world/')
                return self._send(200, WORLD_PLAYBACK.dispatch(operation, body, generate_world_window))
            result = control.post(self.path, body)
            if result is not None:
                return self._send(200, result)
            if self.path == "/api/generate/stream":
                return self._stream_json_lines(body)
            if self.path == "/api/generate/cancel":
                event = register_stream(body)
                event.set()
                release_stream(body, event)
                return self._send(200, {"cancelled": True})
            if self.path == "/api/generate":
                return self._send(200, generate_prompt(body))
            if self.path == "/api/move":
                return self._send(200, generate_move(body))
            if self.path == "/api/follow":
                return self._send(200, generate_follow(body))
            if self.path == "/api/hand":
                return self._send(200, generate_hand(body))
            if self.path == "/api/bind":
                return self._send(200, generate_bind(body))
            if self.path == "/api/point":
                return self._send(200, generate_point(body))
            if self.path == "/api/action":
                return self._send(200, generate_action(body))
            if self.path == "/api/navigate":
                return self._send(200, generate_navigate(body))
            if self.path == "/api/keyframes":
                return self._send(200, generate_keyframes(body))
            if self.path == "/api/intent":
                return self._send(200, generate_intent(body))
            if self.path == "/api/scene":
                return self._send(200, generate_scene(body))
            if self.path == "/api/recognize":
                return self._send(200, recognize_motion(body))
            self._send(404, {"error": f"unknown endpoint {self.path}"})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report to the caller
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):
        pass


def serve_http(port: int) -> None:
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    (MOTION_TOOLS / "out").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ARDY service: http://127.0.0.1:{port}  (health /api/move /api/follow /api/hand /api/motions)")
    if os.environ.get("ARDY_WARMUP", "0") == "1":
        threading.Thread(target=warm_realtime_runner, name="ardy-warmup", daemon=True).start()
    srv.serve_forever()


# --------------------------------------------------------------------------- #
# MCP (stdio, newline-delimited JSON-RPC)
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "generate_move",
        "description": "Generate an ARDY motion walking between two world points, with root-height constraint (climb stairs via `steps`).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "start [x,y,z]"},
                "to": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "end [x,y,z]"},
                "duration": {"type": "number", "default": 8},
                "steps": {"type": "integer", "default": 0},
                "prompt": {"type": "string"},
            },
            "required": ["from", "to"],
        },
    },
    {
        "name": "generate_follow",
        "description": "Constrain ARDY's root to follow one joint of an AMASS SMPL clip (interaction).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "npz": {"type": "string", "description": "path to an AMASS .npz"},
                "joint": {"type": "string", "default": "pelvis"},
                "duration": {"type": "number", "default": 12},
            },
            "required": ["npz"],
        },
    },
    {
        "name": "generate_hand",
        "description": "Constrain ARDY's hand to follow the other character's moving hand (interaction).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "npz": {"type": "string"},
                "hand": {"type": "string", "enum": ["left", "right"], "default": "right"},
                "duration": {"type": "number", "default": 12},
            },
            "required": ["npz"],
        },
    },
    {
        "name": "generate_bind",
        "description": "Bind ARDY's skeleton to another SMPL object: fullbody (whole body follows) or limbs (hands+feet follow).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "npz": {"type": "string"},
                "mode": {"type": "string", "enum": ["fullbody", "limbs"], "default": "fullbody"},
                "duration": {"type": "number", "default": 10},
            },
            "required": ["npz"],
        },
    },
    {
        "name": "generate_point",
        "description": "Mark a single joint at a single world point per keyframe (analytic IK, no reference pose).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame": {"type": "integer"},
                            "joint": {"type": "string", "enum": ["right-hand", "left-hand", "right-foot", "left-foot"]},
                            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                        },
                        "required": ["frame", "joint", "x", "y", "z"],
                    },
                },
                "root": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "heading": {"type": "number", "default": 0},
                "duration": {"type": "number", "default": 6},
            },
            "required": ["points"],
        },
    },
    {
        "name": "generate_navigate",
        "description": "Go to a destination via waypoints at a speed, auto-routing around obstacles.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "waypoints": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "description": "[[x,y,z], ...]"},
                "speed": {"type": "number", "default": 1.4},
                "obstacles": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, "description": "[[cx,cz,half_x,half_z], ...]"},
                "duration": {"type": "number", "default": 0},
            },
            "required": ["waypoints"],
        },
    },
    {
        "name": "generate_action",
        "description": "Play a basic action: walk/run/crouch/crawl/lie/sit/side_lie/roll/wave/point/touch/raise.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "duration": {"type": "number", "default": 5},
            },
            "required": ["name"],
        },
    },
    {
        "name": "generate_keyframes",
        "description": "Solve arbitrary keyframes: root points, world partial points, fullbody poses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyframes": {"type": "array"},
                "default_root": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "default_heading": {"type": "number", "default": 0},
                "duration": {"type": "number", "default": 10},
            },
            "required": ["keyframes"],
        },
    },
    {
        "name": "generate_intent",
        "description": "face (toward a target), touch (a position), or point (a direction).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["face", "touch", "point"]},
                "target": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "joint": {"type": "string", "default": "right-hand"},
                "frame": {"type": "integer", "default": 30},
                "root": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "generate_scene",
        "description": "Describe a scene in natural language (e.g. '左边3米有个人在挥手') and author a motion: parses action + direction + distance into an annotated state, then generates a response grounded by intent keyframes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "ego_action": {"type": "string", "description": "the ego's reaction, a plan.ACTIONS key (wave/touch/point/...)"},
                "root": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "duration": {"type": "number", "default": 8},
            },
            "required": ["text"],
        },
    },
    {
        "name": "recognize_motion",
        "description": "Recognise a motion clip with TMR: returns the top-K nearest text descriptions (the motion-recognition model behind scene description).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "npz": {"type": "string", "description": "path to an AMASS .npz"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["npz"],
        },
    },
    {
        "name": "list_motions",
        "description": "List generated service clips.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

HANDLERS = {
    "generate_move": generate_move,
    "generate_follow": generate_follow,
    "generate_hand": generate_hand,
    "generate_bind": generate_bind,
    "generate_point": generate_point,
    "generate_action": generate_action,
    "generate_navigate": generate_navigate,
    "generate_keyframes": generate_keyframes,
    "generate_intent": generate_intent,
    "generate_scene": generate_scene,
    "recognize_motion": recognize_motion,
    "list_motions": lambda args: list_motions(),
}


def _tool_spec(name: str) -> dict:
    return next(t for t in TOOLS if t["name"] == name)


def mcp_loop() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        mid = req.get("id")
        method = req.get("method")

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0", "id": mid, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ardy-constraint", "version": "0.1.0"},
                },
            }
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": mid, "result": {}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            try:
                result = HANDLERS[name](args)
                text = json.dumps(result)
                resp = {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": False}}
            except Exception as exc:  # noqa: BLE001
                resp = {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}}
        elif method.startswith("notifications/"):
            continue
        else:
            resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["http", "mcp"])
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    if args.mode == "http":
        serve_http(args.port)
    else:
        mcp_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
