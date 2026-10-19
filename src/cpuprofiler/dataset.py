"""Resumable benchmark collection and inference-free replay."""
import argparse
from contextlib import contextmanager
import fcntl
from dataclasses import asdict, replace
import hashlib
import json
import platform
import sqlite3
import statistics
import time
from pathlib import Path

if __package__ in (None, ""):
    # Support `uv run src/cpuprofiler/dataset.py` as well as package execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cpuprofiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
    from cpuprofiler.topology import CPUTopology
else:
    from .benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
    from .topology import CPUTopology


@contextmanager
def collection_lock(path):
    """Prevent simultaneous collectors from perturbing measurements."""
    with Path(str(Path(path).resolve()) + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Collection already running for {path}') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def config_key(config):
    return json.dumps(asdict(config), sort_keys=True)


class BenchmarkDataset:
    def __init__(self, path, context=None):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS samples (
            config TEXT, repetition INTEGER, tps REAL, stderr TEXT,
            PRIMARY KEY (config, repetition))''')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(samples)')}
        if 'elapsed_seconds' not in columns:
            self.db.execute('ALTER TABLE samples ADD COLUMN elapsed_seconds REAL')
            self.db.commit()
        if context is not None:
            value = json.dumps(context, sort_keys=True)
            old = self.db.execute("SELECT value FROM metadata WHERE key='context'").fetchone()
            if old and old[0] != value:
                raise ValueError('Dataset context differs; use a separate database for this machine/model/settings')
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('context', ?)", (value,))
            self.db.commit()

    def samples(self, config):
        return self.db.execute(
            'SELECT repetition, tps, stderr FROM samples WHERE config=? ORDER BY repetition',
            (config_key(config),),
        ).fetchall()

    def record(self, result, repetition, elapsed_seconds=None):
        self.db.execute('INSERT INTO samples (config, repetition, tps, stderr, elapsed_seconds) VALUES (?, ?, ?, ?, ?) ON CONFLICT(config, repetition) DO NOTHING',
                        (config_key(result.config), repetition, result.tokens_per_second, result.raw_stderr, elapsed_seconds))
        self.db.commit()

    def close(self):
        self.db.close()


class ReplayRunner:
    """BenchmarkRunner replacement. Missing measurements fail without inference."""
    def __init__(self, dataset, mode='median'):
        if mode not in ('median', 'cycle'):
            raise ValueError('mode must be median or cycle')
        self.dataset = dataset
        self.mode = mode
        self.positions = {}

    def run(self, config):
        rows = self.dataset.samples(config)
        if not rows:
            raise KeyError(f'No recorded measurements for {config}')
        if self.mode == 'median':
            tps = statistics.median(row[1] for row in rows)
        else:
            key = config_key(config)
            pos = self.positions.get(key, 0)
            tps = rows[pos % len(rows)][1]
            self.positions[key] = pos + 1
        return BenchmarkResult(config, tps, 'replayed from dataset')


def collect(topology, runner, dataset, steps=(32, 64, 128, 256), repeats=3,
            beam_width=12, max_depth=4):
    """Union expanded beams ranked by each budget, median and individual trials.

    Width >= the replay beam width covers its path for these ranking rules.
    Other budgets/aggregations and noisy future runs are not guaranteed covered.
    """
    steps = tuple(sorted(set(steps)))
    if not steps or min(steps) < 2 or min(repeats, beam_width, max_depth) < 1:
        raise ValueError('steps must be >= 2; repeats, beam_width and max_depth must be positive')
    candidates = [BenchmarkConfig((cpu,), 1) for cpu in sorted(topology.cpus)]
    for depth in range(min(max_depth, topology.n_logical_cpus)):
        candidates = sorted(candidates, key=lambda c: c.cpus)
        print(f'depth={depth + 1}: {len(candidates)} configurations', flush=True)
        # Interleave repetitions across configurations to reduce time-order bias.
        for repetition in range(repeats):
            for base in candidates:
                for budget in steps:
                    config = replace(base, steps=budget)
                    if any(row[0] == repetition for row in dataset.samples(config)):
                        continue
                    started = time.perf_counter()
                    result = runner.run(config)
                    elapsed = time.perf_counter() - started
                    dataset.record(result, repetition, elapsed)
                    print(f'cpus={config.cpus} steps={budget} repeat={repetition + 1} '
                          f'tok/s={result.tokens_per_second:.3f} elapsed={elapsed:.3f}s', flush=True)
        parents = set()
        for budget in steps:
            for trial in [None, *range(repeats)]:
                def score(base):
                    rows = {r: tps for r, tps, _ in dataset.samples(replace(base, steps=budget)) if r < repeats}
                    return statistics.median(rows.values()) if trial is None else rows[trial]
                parents.update(sorted(candidates, key=lambda c: (-score(c), c.cpus))[:beam_width])
        candidates = {
            replace(base, cpus=tuple(sorted((*base.cpus, cpu))), threads=len(base.cpus) + 1)
            for base in parents for cpu in topology.cpus if cpu not in base.cpus
        }


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='benchmark/autotune.sqlite3')
    parser.add_argument('--steps', nargs='+', type=int, default=[32, 64, 128, 256])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--beam-width', type=int, default=12)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--binary', default='./build/omp-run')
    parser.add_argument('--model', default='./build/stories15M.bin')
    args = parser.parse_args(argv)
    topology = CPUTopology()
    runner = BenchmarkRunner(args.binary, args.model)
    context = {
        'version': 1, 'host': platform.node(), 'machine': platform.machine(),
        'cpuinfo': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'cpus': sorted(topology.cpus), 'binary': file_identity(args.binary),
        'model': file_identity(args.model),
        'plan': {'steps': sorted(set(args.steps)), 'repeats': args.repeats,
                 'beam_width': args.beam_width, 'max_depth': args.max_depth},
    }
    # cpu MHz fluctuates and is not part of the machine identity.
    context['cpuinfo'] = '\n'.join(line for line in context['cpuinfo'].splitlines()
                                    if not line.startswith('cpu MHz'))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        with collection_lock(args.output):
            dataset = BenchmarkDataset(args.output, context)
            try:
                collect(topology, runner, dataset, args.steps, args.repeats, args.beam_width, args.max_depth)
            finally:
                dataset.close()
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume.', flush=True)
    except RuntimeError as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
