"""Measure actual inference and MLX/CPU parity; not a VRM quality benchmark."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_policy import MODEL_DIRECTORY, SuperMimicPolicy


def statistics(samples: list[float]) -> dict:
    return {'samples': len(samples), 'meanMs': float(np.mean(samples)),
            'p50Ms': float(np.percentile(samples, 50)), 'p95Ms': float(np.percentile(samples, 95)),
            'p99Ms': float(np.percentile(samples, 99)), 'maxMs': max(samples),
            'over20Ms': sum(value > 20 for value in samples)}


def inputs() -> list[tuple[np.ndarray, np.ndarray]]:
    weights = load_file(str(MODEL_DIRECTORY / 'policy.safetensors'))
    random = np.random.default_rng(614)
    samples = []
    for _ in range(16):
        history = random.normal(size=(1, 10, 99)).astype(np.float32)
        goal = random.normal(size=(1, 730)).astype(np.float32)
        history = history * np.sqrt(weights['obs_norm.var'] + 1e-4) + weights['obs_norm.mean']
        goal = goal * np.sqrt(weights['goal_norm.var'] + 1e-4) + weights['goal_norm.mean']
        samples.append((history, goal))
    return samples


def benchmark(backend: str, threads: int, count: int, cases: list) -> tuple[dict, list]:
    policy = SuperMimicPolicy(backend=backend, threads=threads)
    for index in range(30):
        policy.predict(*cases[index % len(cases)])
    actions = [policy.predict(*case) for case in cases]
    times = []
    for index in range(count):
        start = time.perf_counter()
        policy.predict(*cases[index % len(cases)])
        times.append((time.perf_counter() - start) * 1000)
    return {'backend': backend, 'threads': threads, **statistics(times)}, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/supermimic/benchmark.json')
    args = parser.parse_args()
    if not 20 <= args.steps <= 100000:
        parser.error('--steps must be between 20 and 100000')
    cases, results, baseline = inputs(), [], None
    for backend, threads in [('torch', 1), ('torch', 2), ('mlx', 1)]:
        result, actions = benchmark(backend, threads, args.steps, cases)
        if baseline is None:
            baseline = np.asarray(actions)
        result['maxActionDifferenceFromTorch'] = float(np.max(np.abs(np.asarray(actions) - baseline)))
        result['parityPassed'] = bool(np.allclose(actions, baseline, atol=3e-4, rtol=1e-4))
        results.append(result)
    report = {'machine': platform.platform(), 'policy': 'g1-lafan1-040000',
              'inputKind': 'synthetic normalization-scaled; no physical quality claim', 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(row['parityPassed'] for row in results):
        raise SystemExit('Backend parity failed')


if __name__ == '__main__':
    main()
