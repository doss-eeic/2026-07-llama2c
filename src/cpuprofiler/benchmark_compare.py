"""現在・融合前・未改造本家を同じ条件で比較する。設定はmain()のCasesを編集する。"""

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess

if __package__:
    from .benchmarkconfig import ROOT, BenchmarkConfig
else:
    from benchmarkconfig import ROOT, BenchmarkConfig

UPSTREAM = '350e04fe35433e6d2941dce5a1f53308f87058eb'
PREFUSION = '666aa86490bbb61f1d0bab175e0c7ba6e08c45c4'
P4 = (0, 2, 1, 3)
P4E4 = (*P4, 4, 5, 8, 9)
P2E8 = (0, 2, *range(4, 12))
ALL14 = (*P4E4, 6, 7, 10, 11, 12, 13)


@dataclass(frozen=True)
class Case:
    config: BenchmarkConfig
    runtime_defaults: bool = False  # OMP_SCHEDULEとOUTPUT_*を設定しない


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def state():
    return {'time': datetime.now().astimezone().isoformat(),
            'platform_profile': read('/sys/firmware/acpi/platform_profile'),
            'ac_online': read('/sys/class/power_supply/AC/online'),
            'no_turbo': read('/sys/devices/system/cpu/intel_pstate/no_turbo'),
            'package_temp_mC': {str(p): read(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')
                                if read(p.parent / 'type') == 'x86_pkg_temp'},
            'throttle': {str(p): read(p) for p in Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*count')},
            'policies': {str(p): {name: read(p / name) for name in ('scaling_governor', 'energy_performance_preference',
                                                                 'scaling_min_freq', 'scaling_max_freq')}
                         for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}}


def environment(case):
    c = case.config
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('OMP_', 'GOMP_', 'KMP_', 'OUTPUT_', 'STAY_', 'PARALLEL_'))}
    env.update(OMP_NUM_THREADS=str(c.threads), OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
               OMP_PLACES=','.join(f'{{{cpu}}}' for cpu in c.cpus))
    if not case.runtime_defaults:
        env.update(OMP_SCHEDULE=c.omp_schedule, OUTPUT_THREADS=str(c.output_threads),
                   OUTPUT_SCHEDULE=c.output_schedule, OUTPUT_CHUNK=str(c.output_chunk))
    return env


def measure(case, source, output, expected_ac=None):
    c = case.config
    output.mkdir(parents=True, exist_ok=False)
    env = environment(case)
    (output / 'config.json').write_text(json.dumps({
        'config': asdict(c), 'source': source, 'runtime_defaults': case.runtime_defaults,
        'effective_environment': {key: value for key, value in env.items()
                                  if key.startswith(('OMP_', 'OUTPUT_'))},
    }, default=str, indent=2) + '\n')
    before = state()
    command = ['taskset', '-c', ','.join(map(str, c.cpus)), str(c.binary), str(c.model),
               '-z', str(c.tokenizer), '-n', str(c.steps), '-t', str(c.temperature), '-s', str(c.seed), '-i', c.prompt]
    with (output / 'stdout.txt').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True, timeout=c.timeout)
    after = state()
    match = re.search(r'achieved tok/s:\s*([\d.]+)', (output / 'stderr.log').read_text())
    if match is None: raise RuntimeError(f'No throughput: {output}')
    rate = float(match.group(1))
    if not math.isfinite(rate) or rate <= 0: raise RuntimeError(f'Invalid throughput: {output}')
    result = {'tokens_per_second': rate, 'ms_per_token': 1000 / rate,
              'stdout_sha256': digest(output / 'stdout.txt')}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'machine.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    if expected_ac is not None and (before['ac_online'] != expected_ac or after['ac_online'] != expected_ac):
        raise RuntimeError(f'AC state changed during measurement; exclude this suite: {output}')
    return result


def summarize(output, rows):
    summary = []
    for case in dict.fromkeys(r['case'] for r in rows):
        selected = [r for r in rows if r['case'] == case]
        for source in ('current', 'prefusion', 'upstream'):
            data = [r for r in selected if r['source'] == source]
            rates = [r['tokens_per_second'] for r in data]
            summary.append({'case': case, 'source': source, 'runs': len(data),
                            'mean_tokens_per_second': statistics.mean(rates),
                            'median_tokens_per_second': statistics.median(rates),
                            'mean_ms_per_token': statistics.mean(r['ms_per_token'] for r in data)})
    write_csv(output / 'summary.csv', summary)
    comparisons = []
    for case in dict.fromkeys(r['case'] for r in rows):
        by = {(r['source'], r['replicate']): r['tokens_per_second'] for r in rows if r['case'] == case}
        reps = sorted({rep for _, rep in by})
        for reference in ('upstream', 'prefusion'):
            def gain(sample):
                return (statistics.mean(by['current', i] for i in sample) /
                        statistics.mean(by[reference, i] for i in sample) - 1) * 100
            rng = random.Random(42)
            values = sorted(gain(rng.choices(reps, k=len(reps))) for _ in range(5000))
            comparisons.append({'case': case, 'reference': reference, 'throughput_gain_percent': gain(reps),
                                'ci95_low': values[125], 'ci95_high': values[4875]})
    write_csv(output / 'comparisons.csv', comparisons)
    for row in comparisons: print(row, flush=True)


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    repeat = 10
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    output = ROOT / 'results' / datetime.now().strftime('current_vs_upstream_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    (output / 'before.json').write_text(json.dumps(state(), indent=2) + '\n')
    expected_ac = read('/sys/class/power_supply/AC/online')
    versions = {'current': current, 'prefusion': PREFUSION, 'upstream': UPSTREAM}
    binaries, hashes = {}, {}
    for name, revision in versions.items():
        folder = output / 'source' / name
        folder.mkdir(parents=True)
        contents = (ROOT / 'run.c').read_bytes() if name == 'current' else subprocess.check_output(
            ['git', 'show', f'{revision}:run.c'], cwd=ROOT)
        source = folder / 'run.c'; source.write_bytes(contents)
        if name == 'current':
            (folder / 'src').mkdir()
            (folder / 'src/fused.h').write_bytes((ROOT / 'src/fused.h').read_bytes())
        binary = ROOT / 'build' / f'compare-{name}'
        command = ['gcc', '-Ofast', '-march=native', '-fopenmp', str(source), '-lm', '-o', str(binary)]
        with (folder / 'build.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[name] = binary
        hashes[name] = {'revision': revision, 'source_sha256': digest(source), 'binary_sha256': digest(binary),
                        'build_command': command}
        if name == 'current': hashes[name]['fused_header_sha256'] = digest(folder / 'src/fused.h')
    base = BenchmarkConfig(P4E4, 8, output, binary=binaries['current'], build=False,
                           output_threads=8, output_schedule='static', output_chunk=0)
    cases = {}
    for layout, cpus in {'P4': P4, 'P4E4': P4E4, 'P2E8': P2E8, 'All14': ALL14}.items():
        cases[layout + '_static'] = Case(replace(base, cpus=cpus, threads=len(cpus), output_threads=len(cpus)))
    for layout, cpus in {'P4E4': P4E4, 'P2E8': P2E8, 'All14': ALL14}.items():
        cases[layout + '_dynamic'] = Case(replace(base, cpus=cpus, threads=len(cpus), output_threads=len(cpus),
                                                 omp_schedule='dynamic,16', output_schedule='dynamic', output_chunk=128))
    for layout, cpus in {'P4E4': P4E4, 'All14': ALL14}.items():
        cases[layout + '_defaults'] = Case(replace(base, cpus=cpus, threads=len(cpus)), runtime_defaults=True)
    cases['P4_to_output8'] = Case(replace(base, threads=4, output_schedule='dynamic', output_chunk=128))
    # Probe the runtime's initial ICVs using the same environment (before run.c switches output).
    probe_source = output / 'probe.c'
    probe_source.write_text('#include <omp.h>\n#include <stdio.h>\nint main(void){omp_sched_t s;int c;omp_get_schedule(&s,&c);printf("{\\"max_threads\\":%d,\\"schedule_kind\\":%u,\\"chunk\\":%d}\\n",omp_get_max_threads(),(unsigned)s,c);}\n')
    probe = ROOT / 'build/compare-probe'
    subprocess.run(['gcc', '-fopenmp', str(probe_source), '-o', str(probe)], check=True)
    probes = {name: json.loads(subprocess.check_output([probe], env=environment(case), text=True))
              for name, case in cases.items()}
    manifest = {'versions': hashes, 'cases': {name: {'cpus': case.config.cpus, 'threads': case.config.threads,
                 'normal_schedule': None if case.runtime_defaults else case.config.omp_schedule,
                 'output': 'source defaults' if case.runtime_defaults else [case.config.output_threads,case.config.output_schedule,case.config.output_chunk]}
                 for name,case in cases.items()}, 'runtime_probes': probes,
                'model_sha256': digest(base.model), 'tokenizer_sha256': digest(base.tokenizer),
                'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0], 'repeat': repeat,
                'expected_ac_online': expected_ac,
                'randomization': 'shuffle cases within block, then shuffle adjacent three-source groups'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'benchmark_compare.py').write_bytes(Path(__file__).read_bytes())
    print(f'Results: {output} ({len(cases)} cases x 3 sources x {repeat} blocks)', flush=True)
    for source, binary in binaries.items():
        warmup = Case(replace(base, binary=binary, steps=32))
        measure(warmup, source, output / 'warmup' / source)
    rows, expected = [], None
    for replicate in range(1, repeat + 1):
        rng = random.Random(42 + replicate)
        names = list(cases)
        rng.shuffle(names)
        jobs = []
        for name in names:
            sources = list(versions)
            rng.shuffle(sources)
            jobs.extend((name, source) for source in sources)
        for order, (name, source) in enumerate(jobs, 1):
            case = cases[name]
            config = replace(case.config, binary=binaries[source])
            result = measure(Case(config, case.runtime_defaults), source,
                             output / name / source / f'run_{replicate:02d}', expected_ac)
            if expected is None: expected = result['stdout_sha256']
            if result['stdout_sha256'] != expected: raise RuntimeError('Generated text differs; inspect saved stdout')
            rows.append({'case': name, 'source': source, 'replicate': replicate, 'order': order, **result})
            write_csv(output / 'trials.csv', rows)
        print(f'block {replicate}/{repeat} complete', flush=True)
    (output / 'measurement_end.json').write_text(json.dumps({'provenance': 'Observed after all inference trials, before analysis.', 'state': state()}, indent=2) + '\n')
    summarize(output, rows)
    (output / 'after.json').write_text(json.dumps(state(), indent=2) + '\n')


if __name__ == '__main__':
    main()
