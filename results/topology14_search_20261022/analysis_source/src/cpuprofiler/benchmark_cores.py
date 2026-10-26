"""14論理CPUの割り当てを比較する。main()の候補を編集して実行する。"""

import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics

if __package__:
    from .benchmarkconfig import ROOT, BenchmarkConfig
    from .benchmark_distribute import run, save_profile_aggregates
else:
    from benchmarkconfig import ROOT, BenchmarkConfig
    from benchmark_distribute import run, save_profile_aggregates

P4E4 = (0, 2, 1, 3, 4, 5, 8, 9)
P4E8 = (*P4E4, 6, 7, 10, 11)
ALL14 = (*P4E8, 12, 13)
P2E8 = (0, 2, *range(4, 12))


def run_suite(configs, output_root, repeat=3):
    """seed固定で順序を混ぜ、各構成の結果と反復平均を保存する。"""
    binaries = {}
    for config in configs.values():
        binary = config.binary.resolve()
        kind = (config.backend, config.profile)
        if binary in binaries and binaries[binary] != kind:
            raise ValueError('Different backends/profile modes must use separate binary paths')
        binaries[binary] = kind
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (ROOT / 'run.c', ROOT / 'src/parallel.h', ROOT / 'src/stay.h', ROOT / 'src/profile.h')}
    for name in sources:
        target = output_root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output_root / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    jobs = [(name, index) for index in range(repeat) for name in configs]
    random.Random(42).shuffle(jobs)
    rows, built, outputs = [], set(), set()
    for name, index in jobs:
        config = configs[name]
        output = output_root / name / f'run_{index + 1:02d}'
        key = (config.backend, config.profile, config.binary.resolve())
        summary = run(replace(config, output=output, build=config.build and key not in built))
        built.add(key)
        outputs.add(hashlib.sha256((output / 'stdout.txt').read_bytes()).hexdigest())
        if len(outputs) != 1:
            raise RuntimeError('Generated text differs between configurations; inspect saved stdout.txt')
        rows.append({'config': name, 'run': index + 1, 'backend': config.backend,
                     'cpus': ','.join(map(str, config.cpus)), 'threads': config.threads,
                     'output_threads': config.output_threads or config.threads,
                     'output_chunk': config.output_chunk if config.backend == 'parallel' else '',
                     'schedule': config.omp_schedule if config.backend == 'parallel' else 'default',
                     'fuse': config.fuse if config.backend == 'parallel' else False,
                     'attention_threads': config.attention_threads,
                     'output_block': config.output_block if config.backend == 'parallel' else 1, **summary})
        with (output_root / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        print(f'{name} run {index + 1}: {summary["tokens_per_second"]:.2f} tokens/s', flush=True)
    averages = []
    for name in configs:
        values = [row['tokens_per_second'] for row in rows if row['config'] == name]
        averages.append({'config': name, 'runs': len(values), 'mean_tokens_per_second': statistics.mean(values),
                         'median_tokens_per_second': statistics.median(values),
                         'min_tokens_per_second': min(values), 'max_tokens_per_second': max(values)})
    with (output_root / 'rates.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(averages[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(averages)
    if all(config.profile for config in configs.values()):
        save_profile_aggregates(output_root, rows)
    for row in sorted(averages, key=lambda row: row['mean_tokens_per_second'], reverse=True):
        print(f'{row["config"]}: {row["mean_tokens_per_second"]:.2f} tokens/s', flush=True)
    return rows


def main():
    output_root = ROOT / 'results' / datetime.now().strftime('cores_%Y%m%d_%H%M%S_%f')
    base = BenchmarkConfig(P4E4, 8, output_root, binary=ROOT / 'build/parallel-run', backend='parallel')
    configs = {
        'upstream_P4E4': replace(base, backend='upstream', binary=ROOT / 'build/core-upstream'),
        'upstream_all14': replace(base, cpus=ALL14, threads=14, backend='upstream', binary=ROOT / 'build/core-upstream'),
        'dynamic_P4E4': base,
        'dynamic_P2E8': replace(base, cpus=P2E8, threads=10, output_block=4),
        'dynamic_P4E8': replace(base, cpus=P4E8, threads=12),
        'dynamic_all14': replace(base, cpus=ALL14, threads=14),
        'layer8_output12': replace(base, cpus=P4E8, output_threads=12),
        'layer8_output14': replace(base, cpus=ALL14, output_threads=14),
        'layer10_output14': replace(base, cpus=(*P2E8, 1, 3, 12, 13), threads=10, output_threads=14),
        'unfused_P4E4': replace(base, fuse=False),
        'static_P4E4': replace(base, omp_schedule='static'),
        'dynamic_all14_chunk8': replace(base, cpus=ALL14, threads=14, output_chunk=8),
        'row4_P4E4': replace(base, output_block=4),
        'dynamic_all14_chunk2': replace(base, cpus=ALL14, threads=14, output_chunk=2),
        'static_all14': replace(base, cpus=ALL14, threads=14, omp_schedule='static'),
    }
    print(f'Results: {output_root}', flush=True)
    run_suite(configs, output_root, repeat=3)


if __name__ == '__main__':
    main()
