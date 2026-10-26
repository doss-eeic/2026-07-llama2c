"""ワーカー管理・分配・融合を交差させる実験。設定はmain()で編集する。"""

import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

if __package__:
    from .benchmarkconfig import ROOT, BenchmarkConfig
    from .benchmark_distribute import run, build_command
    from .benchmark_cores import P4E4, P2E8, ALL14
else:
    from benchmarkconfig import ROOT, BenchmarkConfig
    from benchmark_distribute import run, build_command
    from benchmark_cores import P4E4, P2E8, ALL14


def candidates(output):
    """64 factorial cells plus the historical baselines and switching variants."""
    specs = {}

    def add(name, config, family, layout):
        specs[name] = (config, {'name': name, 'family': family, 'layout': layout,
                               'backend': config.backend,
                               'dynamic': int(config.omp_schedule == 'dynamic,16'),
                               'qkv': int(config.fuse if config.fuse_qkv is None else config.fuse_qkv),
                               'ffn': int(config.fuse if config.fuse_ffn is None else config.fuse_ffn),
                               'row4': int(config.output_block == 4)})

    def config(cpus, threads, backend, **kwargs):
        return BenchmarkConfig(cpus, threads, output, backend=backend,
                               binary=ROOT / 'build' / ('factorial-' + backend), p_weight=1, **kwargs)

    for layout, cpus in {'P4E4': P4E4, 'P2E8': P2E8}.items():
        for backend, dynamic, qkv, ffn, row4 in itertools.product(
                ('openmp', 'pthread'), (0, 1), (0, 1), (0, 1), (0, 1)):
            name = f'{layout}_{backend}_d{dynamic}_q{qkv}_f{ffn}_r{row4}'
            c = config(cpus, len(cpus), backend, fuse=False, fuse_qkv=bool(qkv), fuse_ffn=bool(ffn),
                       omp_schedule='dynamic,16' if dynamic else 'static', output_block=4 if row4 else 1)
            add(name, c, 'factorial', layout)
    for layout, cpus in {'P4': (0, 2, 1, 3), 'P4E4': P4E4, 'P2E8': P2E8, 'All14': ALL14}.items():
        for backend in ('upstream', 'stay'):
            add(f'legacy_{backend}_{layout}', config(cpus, len(cpus), backend, fuse=False), 'legacy', layout)
    for weight in (1, 2):
        c = replace(config(P4E4, 4, 'stay', fuse=False), p_weight=weight)
        add(f'legacy_stay_switch4_8_w{weight}', c, 'legacy', 'switch4_8')
    for layout, cpus in {'P4E4': P4E4, 'P2E8': P2E8, 'All14': ALL14}.items():
        add(f'legacy_parallel_{layout}', config(cpus, len(cpus), 'parallel', output_block=4 if layout == 'P2E8' else 1),
            'legacy', layout)
    for backend, dynamic, switch in itertools.product(('openmp', 'pthread'), (0, 1), (4, 8)):
        cpus = P4E4 if switch == 4 else ALL14
        name = f'switch_{backend}_{switch}_{len(cpus)}_d{dynamic}'
        c = config(cpus, switch, backend, output_threads=len(cpus),
                   omp_schedule='dynamic,16' if dynamic else 'static')
        add(name, c, 'switch', f'switch{switch}_{len(cpus)}')
    c = replace(config(P4E4, 4, 'pthread', omp_schedule='static'), p_weight=2, output_threads=8)
    add('switch_pthread_4_8_static_weight2', c, 'switch', 'switch4_8')
    return specs


def main():
    repeat = 5
    output = ROOT / 'results' / datetime.now().strftime('factorial_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    specs = candidates(output)
    print(f'Results: {output} ({len(specs)} configurations x {repeat} blocks)', flush=True)
    sources = {}
    for name in ('run.c', 'src/parallel.h', 'src/stay.h', 'src/profile.h',
                 'src/cpuprofiler/benchmarkconfig.py', 'src/cpuprofiler/benchmark_distribute.py',
                 'src/cpuprofiler/benchmark_factorial.py'):
        path = ROOT / name
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (output / 'plan.json').write_text(json.dumps({
        'repeat': repeat, 'seed': 42, 'randomization': 'shuffle within each complete replicate block',
        'factors': ['backend', 'dynamic', 'qkv', 'ffn', 'row4'],
        'note': 'Factorial dynamic uses 16 layer rows and 32 output physical rows per task. Attention is static.',
        'candidates': [meta for _, meta in specs.values()],
    }, indent=2) + '\n')
    for backend in dict.fromkeys(c.backend for c, _ in specs.values()):
        c = next(c for c, _ in specs.values() if c.backend == backend)
        directory = output / 'builds' / backend
        directory.mkdir(parents=True)
        command = build_command(c, directory, c.binary)
        with (directory / 'build.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        (directory / 'binary.sha256').write_text(hashlib.sha256(c.binary.read_bytes()).hexdigest() + '\n')
    # One untimed warm-up per backend; do not mix compiler/test load into measured blocks.
    for backend in dict.fromkeys(c.backend for c, _ in specs.values()):
        c = next(c for c, _ in specs.values() if c.backend == backend)
        run(replace(c, output=output / 'warmup' / backend, steps=32, build=False))
    fields = ['name', 'family', 'layout', 'backend', 'dynamic', 'qkv', 'ffn', 'row4',
              'replicate', 'order', 'tokens_per_second', 'ms_per_token', 'wall_s', 'stdout_sha256']
    expected = None
    with (output / 'trials.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for replicate in range(1, repeat + 1):
            names = list(specs)
            random.Random(42 + replicate).shuffle(names)
            start = time.perf_counter()
            for index, name in enumerate(names, 1):
                c, meta = specs[name]
                dest = output / name / f'run_{replicate:02d}'
                before = time.perf_counter()
                summary = run(replace(c, output=dest, build=False))
                digest = hashlib.sha256((dest / 'stdout.txt').read_bytes()).hexdigest()
                if expected is None: expected = digest
                if digest != expected:
                    raise RuntimeError(f'Generation differs: {dest}')
                writer.writerow({**meta, 'replicate': replicate, 'order': index,
                                 'tokens_per_second': summary['tokens_per_second'],
                                 'ms_per_token': summary['ms_per_token'],
                                 'wall_s': time.perf_counter() - before, 'stdout_sha256': digest})
                file.flush()
                if index % 20 == 0:
                    print(f'block {replicate}/{repeat}: {index}/{len(names)} completed', flush=True)
            print(f'block {replicate}/{repeat} complete ({time.perf_counter() - start:.1f}s)', flush=True)
    if __package__:
        from .analyze_factorial import analyze
    else:
        from analyze_factorial import analyze
    analyze(output)


if __name__ == '__main__':
    main()
