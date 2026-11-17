"""Evaluate G1 reference tracking with real contact physics, separate from the VRM player."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_policy import SuperMimicPolicy
from supermimic_physics import G1Physics
from supermimic_reference import DT, G1Reference, observation
from benchmark_supermimic import statistics


def rollout(policy: SuperMimicPolicy, reference: G1Reference, seconds: float, controller: str) -> tuple[dict, dict]:
    sim = G1Physics(policy.config['control']['joint_names'])
    sim.reset(*(value[0] for value in (reference.position, reference.rotation, reference.joints,
                                      reference.velocity, reference.angular_velocity, reference.joint_velocity)))
    history, previous = None, np.zeros(29)
    times, roots, rotations, joints, errors, joint_errors = [], [], [], [], [], []
    failure = None
    count = min(len(reference.joints) - 1, round(seconds / DT))
    for frame in range(count):
        start = time.perf_counter()
        p, r, v, w, q, qd = sim.state()
        current = observation(q, qd, r, w, previous, reference.rotation[frame], sim.default)
        history = np.repeat(current[None, None], 10, axis=1) if history is None else np.concatenate((history[:, 1:], current[None, None]), axis=1)
        action = policy.predict(history, reference.goal(frame)) if controller == 'supermimic' else (reference.joints[frame + 1] - sim.default) / sim.scale
        sim.step(action)
        times.append((time.perf_counter() - start) * 1000)
        previous = action
        p, r, _, _, q, _ = sim.state()
        roots.append(p); rotations.append(r); joints.append(q)
        errors.append(float(np.linalg.norm(p - reference.position[frame + 1])))
        joint_errors.append(float(np.sqrt(np.mean((q - reference.joints[frame + 1])**2))))
        if abs(p[2] - reference.position[frame + 1, 2]) > .25 or abs(r[2, 2] - reference.rotation[frame + 1, 2, 2]) > .8:
            failure = {'frame': frame + 1, 'seconds': (frame + 1) * DT, 'reason': 'reference height/tilt termination'}
            break
    report = {'controller': controller, 'requestedSeconds': count * DT, 'simulatedSeconds': len(times) * DT,
              'completed': failure is None, 'failure': failure, 'meanRootErrorM': float(np.mean(errors)),
              'maxRootErrorM': max(errors), 'meanJointRmseRad': float(np.mean(joint_errors)),
              'stepLatency': statistics(times), 'physics': 'CPU MuJoCo, 4 x 5ms, G1 gains, self-collision off',
              'vrmQualityValidated': False}
    trajectory = {'root': np.array(roots), 'rotation': np.array(rotations), 'joints': np.array(joints)}
    return report, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--backend', choices=('torch', 'mlx'), default='mlx')
    parser.add_argument('--reference', type=Path, default=ROOT / '.local-data/supermimic/g1/walk1_subject1.csv')
    args = parser.parse_args()
    if not .1 <= args.seconds <= 600:
        parser.error('seconds must be between .1 and 600')
    policy, reference = SuperMimicPolicy(backend=args.backend), G1Reference.from_csv(args.reference)
    reports = []
    for controller in ('supermimic', 'reference-pd'):
        result, trajectory = rollout(policy, reference, args.seconds, controller)
        np.savez_compressed(ROOT / f'output/supermimic/{controller}-trajectory.npz', **trajectory)
        reports.append(result)
    report = {'backend': args.backend, 'reference': str(args.reference), 'results': reports,
              'scope': 'Independent simulator transfer test; neither original published evaluation nor VRM improvement'}
    (ROOT / 'output/supermimic/tracking.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
