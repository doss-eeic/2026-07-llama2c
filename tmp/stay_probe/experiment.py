#!/usr/bin/env python3
"""Temporary controlled stay2 investigation (stdlib only, no privileged changes)."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cpuprofiler.benchmark_detail import parse_profile, summarize


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, PermissionError):
        return None


def sensors(cpus):
    paths = {f'cpu{cpu}_khz': Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq') for cpu in cpus}
    for hw in sorted(Path('/sys/class/hwmon').glob('hwmon*')):
        name = read(hw / 'name') or hw.name
        for p in hw.glob('temp*_input'):
            label = read(p.with_name(p.name.replace('_input', '_label'))) or p.stem
            paths[f'temp_mC:{name}:{label}'] = p
    for p in Path('/sys/class/powercap').glob('*/energy_uj'):
        paths[f'energy_uJ:{p.parent.name}'] = p
    for cpu in cpus:
        for p in Path(f'/sys/devices/system/cpu/cpu{cpu}/thermal_throttle').glob('*throttle_count'):
            paths[f'cpu{cpu}:{p.name}'] = p
    return {key: p for key, p in paths.items() if read(p) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p4', default='0,1,2,3')
    parser.add_argument('--p8', default='0,1,2,3,4,5,8,9')
    parser.add_argument('--tokens', type=Path, help='Existing profile.jsonl or whitespace-separated token IDs')
    parser.add_argument('--skip-tokens', type=int, default=1)
    parser.add_argument('--interval', type=float, default=0.05)
    parser.add_argument('--compare-weights', action='store_true', help='Interleave a Switching_weight2 candidate in the same blocks')
    parser.add_argument('--p-weight', type=int, default=1, help='Diagnostic output partition weight for P CPUs (E CPUs=1)')
    parser.add_argument('--no-sensors', action='store_true', help='Control run without sensor polling')
    parser.add_argument('--modes', default='detailed,clean,output-only')
    parser.add_argument('--monitor-cpu', type=int, help='Use an otherwise idle CPU, preferably outside worker L2 clusters')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.interval <= 0:
        parser.error('repeats and interval must be positive')
    allowed = sorted(os.sched_getaffinity(0))
    p4 = tuple(map(int, args.p4.split(',')))
    p8 = tuple(map(int, args.p8.split(',')))
    used = sorted(set(p4 + p8))
    if not set(used) <= set(allowed):
        parser.error(f'CPUs must be within allowed affinity {allowed}')
    configs = {'P4_HT': (p4, p4), 'P4X4': (p8, p8), 'Switching': (p4, p8)}
    if args.compare_weights:
        configs['Switching_weight2'] = (p4, p8)
    output = (args.output or ROOT / 'results' / datetime.now().strftime('probe_stay2_%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'Results: {output}', flush=True)
    binary = ROOT / 'tmp/stay_probe/probe'
    subprocess.run([sys.executable, str(ROOT / 'tmp/stay_probe/build_probe.py')], check=True)
    paths = {} if args.no_sensors else sensors(used)
    spare = args.monitor_cpu if args.monitor_cpu is not None else next((cpu for cpu in reversed(allowed) if cpu not in used), None)
    if spare is not None and (spare not in allowed or spare in used):
        parser.error('monitor CPU must be allowed and outside worker CPUs')
    modes = tuple(args.modes.split(','))
    if not modes or any(m not in ('detailed', 'clean', 'output-only') for m in modes):
        parser.error('invalid modes')
    topology = {}
    for cpu in used:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        topology[str(cpu)] = {key: read(base / key) for key in (
            'topology/core_id', 'topology/thread_siblings_list', 'topology/core_type',
            'cache/index2/shared_cpu_list', 'cpufreq/scaling_governor',
            'cpufreq/energy_performance_preference', 'cpufreq/scaling_max_freq')}
    metadata = {'args': vars(args) | {'tokens': str(args.tokens), 'output': str(output)},
                'allowed_cpus': allowed, 'monitor_cpu': spare, 'topology': topology,
                'sensors': {key: str(p) for key, p in paths.items()},
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256((ROOT / 'run_stay2.c').read_bytes()).hexdigest(),
                'lscpu': subprocess.run(['lscpu'], capture_output=True, text=True).stdout,
                'notes': ['scaling_cur_freq is a sysfs snapshot, not per-stage actual cycles',
                          'trace runs are separate from timed runs',
                          'output-only uses embeddings as fixed inputs, not full-forward hidden states']}
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2))
    policies = {}
    for name, (base, out) in configs.items():
        policy = output / f'{name}.policy'
        policy.write_text('-2 default ' + ','.join(map(str, base)) + '\n-1 output ' + ','.join(map(str, out)) + '\n')
        policies[name] = policy

    def execute(name, mode, destination, replay=None):
        destination.mkdir()
        env = dict(os.environ)
        for key in ('LLAMA_STAY_TIMING_ONLY', 'LLAMA_STAY_DIAGNOSTICS', 'LLAMA_PROBE_TRACE', 'LLAMA_PROBE_OUTPUT_ONLY'):
            env.pop(key, None)
        env['LLAMA_PROBE_P_WEIGHT'] = str(2 if name.endswith('_weight2') else args.p_weight)
        env['LLAMA_PROBE_P_CPUS'] = ',' + ',,'.join(map(str, p4)) + ','
        env['LLAMA_PROFILE_PATH'] = str(destination / 'profile.jsonl')
        if mode in ('clean', 'trace'):
            env['LLAMA_STAY_TIMING_ONLY'] = '1'
        if mode == 'output-only':
            env['LLAMA_PROBE_OUTPUT_ONLY'] = '1'
        if mode == 'trace':
            env['LLAMA_PROBE_TRACE'] = '1'
        command = [str(binary), str(ROOT / 'build/stories15M.bin'), '-c', str(policies[name]),
                   '-z', str(ROOT / 'tokenizer.bin'), '-n', '256', '-t', '0', '-s', '42', '-i', 'Once upon a time']
        if replay:
            command += ['-r', str(replay)]
        samples = []
        stop = threading.Event()

        def monitor():
            if spare is not None:
                os.sched_setaffinity(0, {spare})
            while not stop.is_set():
                row = {'monotonic_ns': time.monotonic_ns()}
                for key, path in paths.items():
                    value = read(path)
                    if value is not None:
                        row[key] = int(value)
                samples.append(row)
                stop.wait(args.interval)

        thread = threading.Thread(target=monitor)
        if paths:
            thread.start()
        start = time.monotonic()
        try:
            with (destination / 'stdout.txt').open('wb') as stdout, (destination / 'stderr.log').open('wb') as stderr:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True, timeout=args.timeout)
        finally:
            elapsed = time.monotonic() - start
            stop.set()
            if paths:
                thread.join()
            write_csv(destination / 'sensors.csv', samples)
        profile = parse_profile((destination / 'profile.jsonl').read_text())
        summary, stages = summarize(profile['timings'], args.skip_tokens)
        values = {s['stage']: s['ms_per_token'] for s in stages}
        tokens = [(t['pos'], t['input_token'], t['next_token']) for t in profile['tokens']]
        token_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        row = {'config': name, 'mode': mode, 'wall_seconds': elapsed, **summary, **values,
               'tokens_sha256': token_hash, 'sensor_samples': len(samples)}
        if 'layer_total' in values:
            row['other_ms'] = summary['forward_ms_per_token'] - values['layer_total'] - values['output']
        for key in paths:
            readings = [r[key] for r in samples if key in r]
            if readings:
                row[key + ':mean'] = statistics.mean(readings)
                row[key + ':max'] = max(readings)
        (destination / 'result.json').write_text(json.dumps(row, indent=2))
        return row, profile

    # Generate once, then replay identical input positions and token IDs in every trial.
    replay_path = output / 'tokens.txt'
    if args.tokens:
        content = args.tokens.read_text()
        if content.lstrip().startswith('{'):
            profile = parse_profile(content)
            inputs = [r['input_token'] for r in profile['tokens']]
        else:
            inputs = list(map(int, content.split()))
    else:
        _, profile = execute('P4_HT', 'clean', output / 'capture')
        inputs = [r['input_token'] for r in profile['tokens']]
    if len(inputs) <= args.skip_tokens:
        raise ValueError('insufficient replay tokens')
    replay_path.write_text(' '.join(map(str, inputs)) + '\n')
    rng = random.Random(args.seed)
    rows, hashes = [], {}
    # Block 0 warms every condition, is saved, and is excluded from comparison.
    for block in range(args.repeats + 1):
        cases = [(name, mode) for name in configs for mode in modes]
        rng.shuffle(cases)
        for order, (name, mode) in enumerate(cases):
            dest = output / f'b{block:02}_{order:02}_{mode}_{name}'
            row, profile = execute(name, mode, dest, replay_path)
            group = 'output-only' if mode == 'output-only' else 'full'
            hashes.setdefault(group, row['tokens_sha256'])
            if hashes[group] != row['tokens_sha256']:
                raise ValueError(f'replay/argmax mismatch in {dest}')
            row.update(block=block, order=order, trial=dest.name, warmup=block == 0)
            rows.append(row)
            write_csv(output / 'trials.csv', rows)
            print(f'b{block:02} {mode:11} {name:9} {row["forward_ms_per_token"]:.4f} ms/token', flush=True)
    # Per-worker clocks add overhead: never mix these runs into timing statistics.
    for name in configs:
        execute(name, 'trace', output / f'trace_{name}', replay_path)
    aggregates = []
    for mode in modes:
        for name in configs:
            selected = [r for r in rows if not r['warmup'] and r['mode'] == mode and r['config'] == name]
            item = {'mode': mode, 'config': name, 'runs': len(selected)}
            for key in ('forward_ms_per_token', 'output', 'layer_total', 'other_ms'):
                v = [r[key] for r in selected if key in r]
                if v:
                    m = statistics.median(v)
                    item.update({key + ':median': m, key + ':mad': statistics.median(abs(x - m) for x in v),
                                 key + ':min': min(v), key + ':max': max(v)})
            aggregates.append(item)
    write_csv(output / 'comparison.csv', aggregates)
    for mode in modes:
        pairs = []
        for block in range(1, args.repeats + 1):
            subset = {r['config']: r for r in rows if r['block'] == block and r['mode'] == mode}
            pairs.append(subset['Switching']['forward_ms_per_token'] / subset['P4_HT']['forward_ms_per_token'] - 1)
        print(f'{mode}: Switching/P4_HT median change={statistics.median(pairs):+.1%}, wins={sum(x<0 for x in pairs)}/{len(pairs)}', flush=True)
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    main()
