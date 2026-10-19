"""Inference-free benchmark runner backed by recorded samples."""
import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import random
import sqlite3
import statistics
from types import SimpleNamespace

from .benchmark import BenchmarkResult
from .dataset import config_key


class MockBenchmarkRunner:
    """Drop-in runner; recorded elapsed time is counted, never slept.

    median selects the measured trial nearest median throughput (and its time).
    cycle wraps trials per configuration; random samples with replacement.
    """
    def __init__(self, path='benchmark/benchdata.sqlite3', mode='cycle', seed=42):
        if mode not in ('median', 'cycle', 'random'):
            raise ValueError('mode must be median, cycle or random')
        self.mode = mode
        self.seed = seed
        self.records = {}
        uri = Path(path).resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as db:
            for key, repetition, tps, stderr, elapsed in db.execute(
                'SELECT config, repetition, tps, stderr, elapsed_seconds '
                'FROM samples ORDER BY config, repetition'
            ):
                self.records.setdefault(key, []).append((tps, stderr, elapsed))
        if not self.records:
            raise ValueError('Benchmark dataset is empty')
        configs = [json.loads(key) for key in self.records]
        self.cpu_ids = sorted({cpu for config in configs for cpu in config['cpus']})
        self.steps = sorted({config['steps'] for config in configs})
        self.reset()

    def reset(self):
        self.positions = {}
        self.random = random.Random(self.seed)
        self.calls = 0
        self.elapsed_seconds = 0.0
        self.missing_elapsed = 0

    def run(self, config):
        key = config_key(replace(config, cpus=tuple(sorted(config.cpus))))
        rows = self.records.get(key)
        if not rows:
            raise KeyError(f'No recorded measurements for {config}; available steps: {self.steps}')
        if self.mode == 'median':
            median = statistics.median(row[0] for row in rows)
            row = min(rows, key=lambda row: abs(row[0] - median))
        elif self.mode == 'cycle':
            position = self.positions.get(key, 0)
            row = rows[position % len(rows)]
            self.positions[key] = position + 1
        else:
            row = self.random.choice(rows)
        tps, stderr, elapsed = row
        self.calls += 1
        if elapsed is None:
            self.missing_elapsed += 1
        else:
            self.elapsed_seconds += elapsed
        return BenchmarkResult(config, tps, stderr, elapsed_seconds=elapsed)


def main(argv=None):
    from .autotuner import AutoTuner
    from .benchmark import BenchmarkConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='benchmark/benchdata.sqlite3')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--mode', choices=['median', 'cycle', 'random'], default='cycle')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beam-width', type=int, default=3)
    parser.add_argument('--max-depth', type=int, default=4)
    args = parser.parse_args(argv)
    runner = MockBenchmarkRunner(args.data, args.mode, args.seed)
    topology = SimpleNamespace(cpus=runner.cpu_ids, n_logical_cpus=len(runner.cpu_ids))
    tuner = AutoTuner(topology, runner, base_config=BenchmarkConfig((), 1, steps=args.steps))
    try:
        results = tuner.beam_search(n=args.beam_width, max_depth=args.max_depth)
    except KeyError as error:
        parser.exit(1, f'{error}\n')
    if not results:
        parser.error('search produced no candidates')
    best = max((result for _, result in results), key=lambda result: result.tokens_per_second)
    print(f'Best: cpus={best.config.cpus} steps={best.config.steps} tok/s={best.tokens_per_second:.3f}')
    print(f'Calls: {runner.calls}; recorded total time: {runner.elapsed_seconds:.3f}s; '
          f'missing times: {runner.missing_elapsed}')


if __name__ == '__main__':
    main()
