"""Bounded-map planner benchmark; optionally compare a saved implementation."""
import argparse
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from exploration_memory import ExplorationMemory


def make_memory(factory):
    memory = factory()
    memory.position, memory.rotation = np.array([.175, 0., .175]), np.eye(3)
    memory.cells = {(x, z): (-2., 100., 1) for x in range(-50, 50) for z in range(-50, 50)}
    return memory


def benchmark(factory):
    cold = []
    for _ in range(5):
        memory = make_memory(factory)
        started = time.perf_counter()
        reachable = memory.reachable()
        cold.append((time.perf_counter()-started)*1000)
    cached = []
    for _ in range(100):
        started = time.perf_counter()
        memory.reachable()
        cached.append((time.perf_counter()-started)*1000)
    measured = make_memory(factory)
    tracemalloc.start()
    measured.reachable()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    nodes = reachable.parents if hasattr(reachable, 'parents') else reachable
    return {'mapCells': len(memory.cells), 'reachableCells': len(nodes),
            'coldMedianMs': round(statistics.median(cold), 3),
            'repeatP95Ms': round(sorted(cached)[95], 4), 'peakKiB': round(peak/1024, 1)}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('exploration_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ExplorationMemory


def cloud_benchmark():
    from semantic_map import SemanticMap

    class PreviousMap(SemanticMap):
        def _expire(self, timestamp, center):
            expired = [key for key, voxel in self.voxels.items()
                       if timestamp-voxel['lastSeen'] > 45 or np.linalg.norm(voxel['point']-center) > 18]
            for key in expired:
                del self.voxels[key]
            while len(self.voxels) > self.limit:
                self.voxels.popitem(last=False)

    result = {}
    for name, factory in [('baseline', PreviousMap), ('optimized', SemanticMap)]:
        world = factory()
        points = np.random.default_rng(4).uniform(-4, 4, (40000, 3))
        world.voxels.update((index, {'point': point, 'lastSeen': 100.}) for index, point in enumerate(points))
        started = time.perf_counter()
        world._expire(100., np.zeros(3))
        cold = (time.perf_counter()-started)*1000
        repeated = []
        for _ in range(10):
            started = time.perf_counter()
            world._expire(100.1, np.zeros(3))
            repeated.append((time.perf_counter()-started)*1000)
        result[name] = {'voxels': len(world.voxels), 'firstCleanupMs': round(cold, 3),
                        'repeatMedianMs': round(statistics.median(repeated), 4)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--include-cloud', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/playwright/exploration/planner-benchmark.json'))
    args = parser.parse_args()
    result = {'optimized': benchmark(ExplorationMemory)}
    if args.baseline:
        result['baseline'] = benchmark(load_baseline(args.baseline))
        result['repeatSpeedup'] = round(result['baseline']['repeatP95Ms']/result['optimized']['repeatP95Ms'], 1)
        result['peakMemoryReductionPercent'] = round(100*(1-result['optimized']['peakKiB']/result['baseline']['peakKiB']), 1)
    if args.include_cloud:
        result['cloudCleanup'] = cloud_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
