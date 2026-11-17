"""Wall-clock 50 Hz G1 feedback through the actual local HTTP service."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from supermimic_policy import MODEL_DIRECTORY, checked_config
from supermimic_reference import DT, G1Reference
from validate_supermimic_tracking import rollout


class RemotePolicy:
    def __init__(self, url: str):
        self.config, self.url = checked_config(MODEL_DIRECTORY), url
        self.sequence, self.started = 0, None
        self.roundtrips: list[float] = []
        self.server_times: list[float] = []

    def predict(self, history: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.started is None:
            self.started = time.perf_counter()
        remaining = self.started + self.sequence * DT - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        data = {'profile': 'unitree-g1-29dof', 'sessionId': 'g1-physics-http-test',
                'sequence': self.sequence, 'reset': self.sequence == 0, 'deadlineMs': 20,
                'observation': history[0, -1].tolist(), 'goal': goal[0].tolist()}
        request = Request(self.url + '/infer', data=json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        with urlopen(request, timeout=1) as response:
            result = json.load(response)
        self.roundtrips.append((time.perf_counter() - started) * 1000)
        self.server_times.append(result['latencyMs'])
        if result['sequence'] != self.sequence or result['appliedToVrm']:
            raise ValueError('Invalid realtime acknowledgement')
        self.sequence += 1
        return np.array(result['action'], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--url', default='http://127.0.0.1:8300')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('seconds must be between 1 and 300')
    from benchmark_supermimic import statistics
    policy = RemotePolicy(args.url)
    reference = G1Reference.from_csv(ROOT / '.local-data/supermimic/g1/walk1_subject1.csv')
    result, _ = rollout(policy, reference, args.seconds, 'supermimic')
    result['stepLatencyIncludes50HzPacing'] = True
    report = {'physics': result, 'httpRoundtrip': statistics(policy.roundtrips),
              'serverInference': statistics(policy.server_times), 'vrmEnabled': False}
    (ROOT / 'output/supermimic/http-tracking.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not result['completed']:
        raise SystemExit('G1 HTTP tracking failed')


if __name__ == '__main__':
    main()
