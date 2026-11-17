#!/usr/bin/env python3
"""Run ARDY text-to-motion inference cheaply on a CPU/MPS box.

Wraps the upstream ``ardy/scripts/generate.py`` so the 8 B text encoder is
never loaded unless the prompt is missing from the embedding cache:

* cache miss  -> spawn ``ardy_text_encoder.py`` (loads Llama-3-8B with
  device_map offload, encodes, writes the cache, exits), then continue;
* cache hit   -> serve the embedding straight from disk.

Either way the motion model itself is only 191 M parameters / ~765 MB in fp32,
so generation runs comfortably in RAM.

Usage:
    python motion-tools/ardy_generate.py "A person walks in a circle." --model core
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from settings import PROJECT as ROOT, ARDY_ROOT as ARDY, MOTION_TOOLS, EMBEDDINGS
sys.path.insert(0, str(ARDY))
GENERATE = ARDY / 'scripts/generate.py'
ENCODER = MOTION_TOOLS / 'ardy_text_encoder.py'
DEFAULT_CACHE = EMBEDDINGS
LLM_DIM = 4096


class CachedTextEncoder:
    """Duck-typed stand-in for ``ardy.model.LLM2VecEncoder``.

    ARDY only needs ``(feature, length)`` where the feature is ``[B, 1, 4096]``
    -- a single pooled vector per prompt -- so a lookup table is a drop-in.
    """

    llm_dim = LLM_DIM

    def __init__(self, table: dict[str, np.ndarray], device: str = "cpu") -> None:
        self._table = table
        self._device = device

    def to(self, device=None, dtype=None):  # noqa: ANN001, ANN002
        if device is not None:
            self._device = str(device)
        return self

    def eval(self):
        return self

    def get_device(self):
        return self._device

    def __call__(self, texts: list[str] | str):
        if isinstance(texts, str):
            texts = [texts]
        missing = [t for t in texts if t not in self._table]
        if missing:
            raise KeyError(
                f"prompt(s) not in the embedding cache: {missing!r}. "
                f"Run: python {ENCODER} --prompt <text>"
            )
        emb = np.stack([self._table[t] for t in texts]).astype(np.float32)
        feat = torch.from_numpy(emb).unsqueeze(1)  # [B, 1, 4096]
        return feat, [1] * len(texts)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {str(p): e for p, e in zip(data["prompts"], data["emb"])}


def ensure_embedding(prompt: str, cache: Path, cpu_budget: str) -> None:
    if prompt in load_cache(cache):
        return
    print(f"prompt not cached; encoding with the 8B text encoder -> {cache}", flush=True)
    t = time.time()
    subprocess.run(
        [sys.executable, str(ENCODER), "--prompt", prompt, "--out", str(cache), "--cpu-budget", cpu_budget],
        check=True,
        cwd=str(ROOT),
    )
    print(f"encoded in {time.time() - t:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_CACHE, help="embedding cache path")
    ap.add_argument("--cpu-budget", default="8GiB", help="RAM budget for the 8B encoder")
    ap.add_argument("--no-encode", action="store_true", help="fail instead of encoding a missing prompt")
    known, rest = ap.parse_known_args()

    prompt = next((a for a in rest if not a.startswith("-")), None)
    if prompt is None:
        ap.error("no prompt given; pass it before any --flags")

    if known.no_encode:
        if prompt not in load_cache(known.embeddings):
            sys.exit(f"prompt not in {known.embeddings} and --no-encode was set")
    else:
        ensure_embedding(prompt, known.embeddings, known.cpu_budget)

    table = load_cache(known.embeddings)
    encoder = CachedTextEncoder(table)

    # Swap ARDY's text-encoder factory for the cache-backed one. `load_model`
    # resolves this global at call time, so patching the module is enough.
    # NB: `import ardy.model.load_model as m` would bind the *function*
    # `load_model`, because ardy/model/__init__.py re-exports it over the
    # submodule attribute -- sys.modules is the only reliable handle.
    import importlib

    importlib.import_module("ardy.model.load_model")
    ardy_load_model = sys.modules["ardy.model.load_model"]
    ardy_load_model.load_text_encoder = lambda **kwargs: encoder

    # Hand the remaining argv to upstream's CLI unchanged.
    sys.argv = [str(GENERATE), *rest]
    print(f"generating with cached embeddings ({len(table)} prompt(s) in {known.embeddings})", flush=True)
    t = time.time()
    runpy.run_path(str(GENERATE), run_name="__main__")
    print(f"generation finished in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
