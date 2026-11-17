"""Portable inference for the pinned, independently trained G1 policy.

Architecture follows Chong Zhang / Pantelis Manouselis' MIT implementation.
See licenses/SuperMimic-reproduction-MIT.txt. No simulator imports or pickle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / 'models/supermimic/g1-lafan1-040000'
POLICY_SHA256 = 'a1c409acf38bc11a216a7decb120e4986c587cdd3aa3d383319dfaed990404ef'
CONFIG_SHA256 = '2dc9e60bb86efc0e35bcac7a063bf6b683ad3d5ef3c6bd853bc814ced46bc2dc'
FloatArray = NDArray[np.float32]


class Inference(Protocol):
    def __call__(self, history: FloatArray, goal: FloatArray) -> FloatArray: ...


def checked_config(directory: Path) -> dict:
    for name, expected in [('policy.safetensors', POLICY_SHA256), ('config.json', CONFIG_SHA256)]:
        with (directory / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'SuperMimic artifact checksum mismatch: {name}')
    config = json.loads((directory / 'config.json').read_text())
    dimensions = config['policy']
    if (dimensions['obs_dim'], dimensions['goal_dim'], dimensions['hist'], dimensions['z_dim']) != (99, 730, 10, 32):
        raise ValueError('Unsupported SuperMimic policy interface')
    return config


def checked_inputs(history: FloatArray, goal: FloatArray) -> tuple[FloatArray, FloatArray]:
    history, goal = np.asarray(history, dtype=np.float32), np.asarray(goal, dtype=np.float32)
    if history.shape != (1, 10, 99) or goal.shape != (1, 730):
        raise ValueError('Expected G1 history [1,10,99] and reference goal [1,730]')
    if not np.isfinite(history).all() or not np.isfinite(goal).all():
        raise ValueError('SuperMimic inputs must be finite')
    if np.max(np.abs(history)) > 1e5 or np.max(np.abs(goal)) > 1e5:
        raise ValueError('SuperMimic inputs exceed supported numerical range')
    return history, goal


def torch_inference(path: Path, threads: int) -> Inference:
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(threads)
    weights = load_file(str(path), device='cpu')

    def norm(value, prefix):
        return (value - weights[prefix + '.mean']) / torch.sqrt(weights[prefix + '.var'] + 1e-4)

    def mlp(value, prefix):
        for layer in (0, 2, 4, 6, 8):
            value = torch.nn.functional.linear(value, weights[f'{prefix}.{layer}.weight'], weights[f'{prefix}.{layer}.bias'])
            if layer < 8:
                value = torch.nn.functional.elu(value)
        return value

    @torch.inference_mode()
    def infer(history: FloatArray, goal: FloatArray) -> FloatArray:
        latent = torch.tanh(mlp(norm(torch.from_numpy(goal), 'goal_norm'), 'enc'))
        observed = norm(torch.from_numpy(history), 'obs_norm').flatten(-2)
        return mlp(torch.cat((latent, observed), -1), 'dec').numpy().copy()
    return infer


def mlx_inference(path: Path) -> Inference:
    import mlx.core as mx
    weights = mx.load(str(path))
    mx.eval(weights)

    def norm(value, prefix):
        return (value - weights[prefix + '.mean']) * mx.rsqrt(weights[prefix + '.var'] + 1e-4)

    def mlp(value, prefix):
        for layer in (0, 2, 4, 6, 8):
            value = value @ weights[f'{prefix}.{layer}.weight'].T + weights[f'{prefix}.{layer}.bias']
            if layer < 8:
                value = mx.where(value > 0, value, mx.expm1(mx.minimum(value, 0)))
        return value

    @mx.compile
    def forward(history, goal):
        latent = mx.tanh(mlp(norm(goal, 'goal_norm'), 'enc'))
        observed = norm(history, 'obs_norm').reshape(1, -1)
        return mlp(mx.concatenate((latent, observed), axis=-1), 'dec')

    def infer(history: FloatArray, goal: FloatArray) -> FloatArray:
        result = forward(mx.array(history), mx.array(goal))
        mx.eval(result)
        return np.array(result)
    return infer


class SuperMimicPolicy:
    def __init__(self, directory: Path = MODEL_DIRECTORY, backend: str = 'torch', threads: int = 1):
        if backend not in ('torch', 'mlx') or threads not in (1, 2, 4):
            raise ValueError('Unsupported backend or thread count')
        self.config = checked_config(directory)
        self.backend = backend
        path = directory / 'policy.safetensors'
        self._infer = mlx_inference(path) if backend == 'mlx' else torch_inference(path, threads)

    def predict(self, history: FloatArray, goal: FloatArray) -> FloatArray:
        history, goal = checked_inputs(history, goal)
        action = self._infer(history, goal)
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise ValueError('Invalid SuperMimic output; retain original motion')
        return action[0]
