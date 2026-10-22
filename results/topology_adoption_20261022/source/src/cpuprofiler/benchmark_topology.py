"""Record complete adoption-space benchmarks and independent confirmation.

Run with python3 -m cpuprofiler.benchmark_topology (PYTHONPATH=src), or from an
IDE with src on its import path. All settings are Python/CLI arguments.
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from .benchmarkconfig import ROOT, BenchmarkConfig
from .benchmark_distribute import build_command, execution_environment, run
from .topology import read
from .topology_search import Candidate, Machine, catalog


def dump(path, obj):
    path.write_text(json.dumps(obj, default=str, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state():
    """Read-only context; frequencies are snapshots, not per-run average clocks."""
    return {'time_ns': time.time_ns(), 'monotonic_s': time.monotonic(),
            'loadavg': list(os.getloadavg()), 'proc_stat': read('/proc/stat'),
            'frequencies_khz': {str(i): read(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq')
                                for i in sorted(os.sched_getaffinity(0))},
            'thermal_millidegrees': {str(p): read(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')},
            'power_supply': {str(p): read(p) for pattern in ('*/status', '*/online')
                             for p in Path('/sys/class/power_supply').glob(pattern)}}


def config_for(c, output, steps=256, profile=False):
    return BenchmarkConfig(c.cpus, len(c.layer), output,
                           binary=output / 'builds' / (c.backend + ('-detail' if profile else '')),
                           backend=c.backend, output_threads=len(c.output), p_weight=1,
                           omp_schedule='dynamic,16' if c.dynamic else 'static',
                           output_dynamic=bool(c.output_dynamic), output_block=4 if c.row4 else 1,
                           fuse_qkv=bool(c.qkv), fuse_ffn=bool(c.ffn),
                           steps=steps, profile=profile, build=False)


def control(machine):
    e = tuple(i for group in machine.efficient_l2 for i in group[:max(1, len(group)//2)])
    cpus = machine.p_primary + machine.p_smt + e
    return Candidate(cpus, cpus, 'openmp', 1, 1, 0)


def ablations(machine):
    """Same two realistic layouts as earlier factorials; only missing cells added."""
    ctrl = control(machine).layer
    p2e = machine.p_primary + machine.e
    return [Candidate(cpus, cpus, b, d, d, r, q, f)
            for cpus, b, d, r, q, f in itertools.product(
                (ctrl, p2e), ('openmp', 'pthread'), (0, 1), (0, 1), (0, 1), (0, 1))]


class Recorder:
    def __init__(self, output, machine, specs, args, steps=256, profile=False, references=False):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.machine = machine
        self.steps = steps
        self.profile = profile
        self.expected = None
        self.sequence = 0
        self.stream = (self.output / 'trials.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=[
            'sequence', 'name', 'kind', 'replicate', 'order', 'start_s', 'end_s',
            'tokens_per_second', 'ms_per_token', 'wall_s', 'stdout_sha256', 'path'], lineterminator='\n')
        self.writer.writeheader()
        self.stream.flush()
        dump(self.output / 'machine.json', asdict(machine))
        dump(self.output / 'plan.json', {'args': args, 'steps': steps, 'profile': profile,
             'adoption_count': len(catalog(machine)), 'candidates': [asdict(c) for c in specs],
             'control': asdict(control(machine)), 'anchor_every': 16,
             'normalization': 'linear interpolation of within-block controls in monotonic time'})
        paths = ['run.c', 'src/parallel.h', 'src/stay.h', 'src/profile.h',
                 'src/cpuprofiler/benchmarkconfig.py', 'src/cpuprofiler/benchmark_distribute.py',
                 'src/cpuprofiler/topology.py', 'src/cpuprofiler/topology_search.py',
                 'src/cpuprofiler/benchmark_topology.py']
        sources = {}
        for name in paths:
            p = ROOT / name
            dest = self.output / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            sources[name] = sha(p)
        dump(self.output / 'sources.json', sources)
        model = ROOT / 'build/stories15M.bin'
        dump(self.output / 'environment.json', {
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
            'platform': platform.platform(), 'python': platform.python_version(),
            'compiler': subprocess.check_output(['gcc', '--version'], text=True),
            'lscpu': subprocess.check_output(['lscpu'], text=True),
            'model_sha256': sha(model), 'tokenizer_sha256': sha(ROOT / 'tokenizer.bin'),
            'initial_state': state()})
        for backend in ('openmp', 'pthread'):
            c = replace(control(machine), backend=backend)
            config = config_for(c, self.output, steps, profile)
            directory = self.output / 'builds' / (backend + '-sources')
            directory.mkdir(parents=True)
            command = build_command(config, directory, config.binary)
            with (directory / 'build.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            dump(directory / 'command.json', command)
            dump(directory / 'binary.json', {'sha256': sha(config.binary)})
        self.refs = {}
        if references:
            for name, ref, fused in (
                ('upstream', '350e04fe35433e6d2941dce5a1f53308f87058eb', False),
                ('qkvffn_branch', 'dev/wataru-qkv-ffn', True)):
                rev = subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
                directory = self.output / 'builds' / name
                directory.mkdir()
                for source in (['run.c', 'src/fused.h'] if fused else ['run.c']):
                    dest = directory / source
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(subprocess.check_output(['git', 'show', f'{rev}:{source}'], cwd=ROOT))
                binary = directory / 'run'
                command = ['gcc', '-Ofast', '-march=native', '-fopenmp', str(directory/'run.c'), '-lm', '-o', str(binary)]
                with (directory/'build.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                dump(directory/'provenance.json', {'revision': rev, 'command': command, 'sha256': sha(binary)})
                self.refs[name] = binary

    def measure(self, c, kind, replicate, order, reference=None):
        self.sequence += 1
        name = reference or c.name
        dest = self.output / 'runs' / f'{self.sequence:05d}'
        cfg = replace(config_for(c, self.output, self.steps, self.profile), output=dest)
        if reference:
            cfg = replace(cfg, backend='parallel' if reference == 'qkvffn_branch' else 'upstream',
                          binary=self.refs[reference])
        before = state()
        start = time.monotonic()
        try:
            summary = run(cfg)
        except Exception as error:
            dump(self.output / 'failure.json', {'sequence': self.sequence, 'name': name, 'error': repr(error), 'state': state()})
            raise
        end = time.monotonic()
        dump(dest/'state.json', {'before': before, 'after': state()})
        # Capture all actual effective OpenMP/pool variables. Exact branch defaults
        # use dynamic output with chunk32, which match the reference control.
        env = execution_environment(cfg)
        dump(dest/'runtime.json', {k:v for k,v in env.items() if k.startswith(('OMP_', 'STAY_', 'GOMP_', 'OUTPUT_', 'PARALLEL_'))})
        digest = sha(dest/'stdout.txt')
        if self.expected is None:
            self.expected = digest
        if digest != self.expected:
            raise RuntimeError(f'Generation mismatch, inspect {dest}')
        row = dict(sequence=self.sequence, name=name, kind=kind, replicate=replicate, order=order,
                   start_s=start, end_s=end, tokens_per_second=summary['tokens_per_second'],
                   ms_per_token=summary['ms_per_token'], wall_s=end-start,
                   stdout_sha256=digest, path=str(dest.relative_to(self.output)))
        self.writer.writerow(row)
        self.stream.flush()
        return row

    def close(self):
        self.stream.close()
        dump(self.output/'complete.json', {'measurements': self.sequence, 'stdout_sha256': self.expected, 'final_state': state()})


def sweep(output, repeats=3):
    if repeats < 2:
        raise ValueError('At least two independent blocks are required')
    machine = Machine.discover()
    adopted = catalog(machine)
    specs = list(dict.fromkeys([*adopted, *ablations(machine)]))
    recorder = Recorder(output, machine, specs, {'mode': 'sweep', 'repeats': repeats, 'seed': 20261022}, references=True)
    adopted_set = set(adopted)
    anchor = control(machine)
    print(f'{recorder.output}: {len(adopted)} adoption + {len(specs)-len(adopted)} ablation cells, {repeats} blocks', flush=True)
    for backend in ('openmp', 'pthread'):
        recorder.measure(replace(anchor, backend=backend), 'warmup', 0, 0)
    for rep in range(1, repeats+1):
        jobs = [(c, None) for c in specs] + [(anchor, name) for name in recorder.refs]
        random.Random(20261022+rep).shuffle(jobs)
        recorder.measure(anchor, 'anchor', rep, 0)
        for order, (c, reference) in enumerate(jobs, 1):
            kind = 'reference' if reference else ('adoption' if c in adopted_set else 'ablation')
            recorder.measure(c, kind, rep, order, reference)
            if order % 16 == 0 or order == len(jobs):
                recorder.measure(anchor, 'anchor', rep, order)
            if order % 64 == 0 or order == len(jobs):
                print(f'block {rep}/{repeats}: {order}/{len(jobs)}', flush=True)
    recorder.close()


def confirm(source, output, repeats=8, profile=False):
    source = Path(source)
    obj = json.loads((source/'machine.json').read_text())
    machine = Machine(tuple(map(tuple,obj['performance'])), tuple(map(tuple,obj['efficient_l2'])), tuple(obj['low_power']))
    # Match the recorded topology to the real current machine before executing.
    current = Machine.discover()
    if (machine.performance, machine.efficient_l2, machine.low_power) != (current.performance,current.efficient_l2,current.low_power):
        raise ValueError('Confirmation machine differs from source topology')
    selected = json.loads((source/'selection.json').read_text())
    specs = list(dict.fromkeys(Candidate.from_dict(c) for c in selected['candidates']))
    rec = Recorder(output, current, specs, {'mode': 'confirm', 'source': str(source.resolve()), 'repeats': repeats},
                   steps=256, profile=profile, references=not profile)
    anchor = control(current)
    for backend in ('openmp', 'pthread'):
        rec.measure(replace(anchor,backend=backend), 'warmup', 0, 0)
    for rep in range(1,repeats+1):
        jobs = [(c,None) for c in specs] + [(anchor,name) for name in rec.refs]
        random.Random(20262022+rep).shuffle(jobs)
        rec.measure(anchor,'anchor',rep,0)
        for order,(c,ref) in enumerate(jobs,1):
            rec.measure(c,'reference' if ref else 'confirmation',rep,order,ref)
        rec.measure(anchor,'anchor',rep,len(jobs))
        print(f'confirmation block {rep}/{repeats}: {len(jobs)} cells',flush=True)
    rec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--confirm', type=Path, help='Source directory with selection.json')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    output = args.output or ROOT/'results'/datetime.now().strftime('topology_%Y%m%d_%H%M%S')
    if args.confirm:
        confirm(args.confirm, output, args.repeats, args.profile)
    else:
        if args.profile:
            parser.error('--profile requires --confirm')
        sweep(output,args.repeats)


if __name__ == '__main__':
    main()
