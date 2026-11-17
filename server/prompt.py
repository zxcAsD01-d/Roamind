#!/usr/bin/env python3
"""Natural-language scene -> structured motion state -> ARDY prompt + intent.

This is the "prompt generator" side of the motion-world interface: a scene is
described in plain language ("左边 3 米有个人在挥手"), parsed into a *complete*
state annotation -- subject action, relative **direction** and **distance** --
and turned into (a) an ARDY text prompt and (b) the intent keyframes (face
toward the person, reach out, etc.).

The nearby person's action itself is recognised by the TMR motion-recognition
model (`recognize.py`), whose text label is then fed in here; so an AI that only
sees language can still author a scene grounded in what the characters are
actually doing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan  # noqa: E402

# Beyond this distance a character's *action* is not recognised -- the LLM still
# knows where it is and which way it faces, but not what it is doing.
RECOGNITION_RADIUS = 5.0  # metres

# direction word -> heading offset in radians (ego faces +Z, left = +X)
DIRECTION_OFFSETS: dict[str, float] = {
    "前": 0.0, "正前": 0.0, "前方": 0.0, "面前": 0.0,
    "后": np.pi, "后方": np.pi, "背后": np.pi, "身后": np.pi,
    "左": np.pi / 2, "左边": np.pi / 2, "左侧": np.pi / 2, "左方": np.pi / 2,
    "右": -np.pi / 2, "右边": -np.pi / 2, "右侧": -np.pi / 2, "右方": -np.pi / 2,
    "左前": np.pi / 4, "右前": -np.pi / 4,
    "左后": 3 * np.pi / 4, "右后": -3 * np.pi / 4,
}

# action verb (zh) -> English prompt fragment + ACTION key (plan.ACTIONS)
ACTION_VERBS: dict[str, tuple[str, str]] = {
    "走": ("walks", "walk"), "走路": ("walks", "walk"), "走": ("walks", "walk"),
    "跑": ("runs", "run"), "跑步": ("runs", "run"),
    "蹲": ("crouches down", "crouch"), "蹲下": ("crouches down", "crouch"),
    "爬": ("crawls forward on hands and knees", "crawl"), "爬行": ("crawls forward on hands and knees", "crawl"),
    "躺": ("lies down", "lie"), "躺下": ("lies down", "lie"),
    "侧躺": ("lies on their side", "side_lie"),
    "坐": ("sits down", "sit"), "坐下": ("sits down", "sit"),
    "滚": ("rolls on the ground", "roll"), "翻滚": ("rolls on the ground", "roll"),
    "挥手": ("waves their hand", "wave"),
    "指": ("points forward with their right arm", "point"), "指向": ("points forward with their right arm", "point"),
    "摸": ("touches something with their right hand", "touch"), "触摸": ("touches something with their right hand", "touch"),
    "举手": ("raises their right arm", "raise"),
}


def parse_scene(text: str) -> list[dict]:
    """Parse a NL description into a list of entity states.

    Each entity: ``{"action": verb_en, "action_key": key, "direction": heading_offset,
    "distance": metres, "subject": "someone"}``.  Unparseable text yields an
    empty list rather than a wrong state.
    """
    out = []
    for clause in re.split(r"[，,;；。]+", text):
        clause = clause.strip()
        if not clause:
            continue
        direction = 0.0
        dir_word = None
        for w, off in DIRECTION_OFFSETS.items():
            if w in clause:
                direction = off
                dir_word = w
                break
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|m|M)", clause)
        distance = float(m.group(1)) if m else 1.5
        action_en = None
        action_key = None
        for w, (en, key) in ACTION_VERBS.items():
            if w in clause:
                action_en, action_key = en, key
                break
        if action_en is None:
            continue
        out.append(
            {
                "subject": "someone",
                "action": action_en,
                "action_key": action_key,
                "direction": float(direction),
                "direction_word": dir_word or "front",
                "distance": distance,
            }
        )
    return out


# direction word -> English, for prompt composition
DIRECTION_EN = {
    "前": "front", "正前": "front", "前方": "front", "面前": "front",
    "后": "behind", "后方": "behind", "背后": "behind", "身后": "behind",
    "左": "left", "左边": "left", "左侧": "left", "左方": "left",
    "右": "right", "右边": "right", "右侧": "right", "右方": "right",
    "左前": "front-left", "右前": "front-right",
    "左后": "back-left", "右后": "back-right",
}


def compose_prompt(entity: dict, ego_action: str | None = None) -> str:
    """An ARDY text prompt from a parsed entity (and the ego's own reaction)."""
    d = DIRECTION_EN.get(entity["direction_word"], entity["direction_word"])
    dist = entity["distance"]
    who = f"someone {dist:g} metres to the {d}"
    if ego_action:
        return f"A person faces {who} who {entity['action']}, and {ego_action}."
    return f"A person faces {who} who {entity['action']}."


def entity_position(entity: dict, ego_root: tuple[float, float, float]) -> tuple[float, float, float]:
    """World position of the entity relative to the ego's root + heading 0."""
    th = entity["direction"]
    d = entity["distance"]
    x = ego_root[0] + np.sin(th) * d
    z = ego_root[2] + np.cos(th) * d
    return (x, ego_root[1], z)


def entity_keyframes(
    entities: list[dict],
    ego_root: tuple[float, float, float],
    frames: list[int] | None = None,
) -> list[dict]:
    """Face each entity in turn (heading keyframes), touching the nearest one.

    Returns keyframes suitable for ``plan.solve_keyframes``.
    """
    if not entities:
        return []
    frames = frames or [int(10 + i * 20) for i in range(len(entities))]
    kfs = []
    for ent, frame in zip(entities, frames):
        target = entity_position(ent, ego_root)
        kf = plan.face_keyframe(frame, ego_root, target)
        # touching distance: reach out toward the entity
        if ent["distance"] <= 1.0:
            kf["joints"] = [("right-hand", list(target))]
        kfs.append(kf)
    return kfs


def describe(text: str, ego_root: tuple[float, float, float] = (0.0, 0.92, 0.0)) -> dict:
    """The one-call interface: NL -> structured state + prompt + keyframes."""
    entities = parse_scene(text)
    for e in entities:
        e["nearby"] = e["distance"] <= RECOGNITION_RADIUS
        if not e["nearby"]:
            e["action"] = None  # distant: position/direction only, no action
    return {
        "entities": entities,
        "prompt": compose_prompt(entities[0]) if entities and entities[0]["action"] else "A person stands.",
        "keyframes": entity_keyframes(entities, ego_root),
    }


def annotate_scene(
    characters: list[dict],
    recognize,
    *,
    ego_root: tuple[float, float, float] = (0.0, 0.92, 0.0),
    radius: float = RECOGNITION_RADIUS,
) -> list[dict]:
    """Annotate observed characters into the motion-world state.

    ``characters`` is a list of ``{"position": [x,y,z], "npz": path}``.  ``recognize``
    is a callable ``npz -> text`` (the TMR recognition model).  Each character is
    reduced to ``{direction, distance, position, action}`` where ``action`` is the
    recognised text **only when within ``radius``** -- farther away it is ``None``.
    """
    entities = []
    for c in characters:
        pos = c["position"]
        dx = pos[0] - ego_root[0]
        dz = pos[2] - ego_root[2]
        dist = float(np.hypot(dx, dz))
        heading_offset = float(np.arctan2(dx, dz))
        direction_word = _direction_word_for(heading_offset)
        entity = {
            "subject": c.get("subject", "someone"),
            "direction": heading_offset,
            "direction_word": direction_word,
            "distance": dist,
            "position": list(pos),
            "nearby": dist <= radius,
            "action": None,
        }
        if entity["nearby"] and c.get("npz") and recognize:
            entity["action"] = recognize(c["npz"])
        entities.append(entity)
    return entities


def _direction_word_for(offset: float) -> str:
    """Nearest compass word for a heading offset (radians)."""
    deg = float(np.degrees(offset)) % 360
    names = [
        (0, "front"), (45, "front-left"), (90, "left"), (135, "back-left"),
        (180, "behind"), (225, "back-right"), (270, "right"), (315, "front-right"),
    ]
    return min(names, key=lambda n: abs((deg - n[0] + 180) % 360 - 180))[1]
