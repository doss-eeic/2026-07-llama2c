"""Reproducible, sequential search for separate layer/output worker counts.

Run: PYTHONPATH=src python3 -m cpuprofiler.tune_output
The search uses stories15M and this machine's 225U CPU numbering.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

from .benchmarkconfig import ROOT, BenchmarkConfig
from .benchmark_distribute import run


def key(config):
    return json.dumps(asdict(replace(config, output=None)), default=str, sort_keys=True)


def candidates():
    # P cores first (one hardware thread each), then SMT and E clusters.
    orders = [
        (0, 2, 1, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13),
        (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 1, 3, 12, 13),
        (0, 2, 1, 3, 6, 7, 9, 10, 11, 4, 5, 8, 12, 13),
    ]
    base = BenchmarkConfig(cpus=orders[0], threads=8, build=False,
                           optimize_weights=True, two_stage_argmax=True,
                           omp_wait_policy="ACTIVE", omp_schedule="guided,16")
    cases = {}

    def add(c):
        cases[key(c)] = c

    for order in orders:
        for n in (2, 4, 6, 8, 9, 10, 12, 14):
            for persistent in (False, True):
                add(replace(base, cpus=order[:n], threads=n,
                            persistent_openmp=persistent))
        for total in (8, 10, 12, 14):
            for layer in (4, 6, 8, 9, 10, 12):
                if layer >= total:
                    continue
                for spin in (None, 1000):
                    add(replace(base, cpus=order[:total], threads=layer,
                                persistent_openmp=True, output_threads=total,
                                gomp_spincount=spin))
    return list(cases.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refine", type=Path, help="Previous search directory")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--final", action="store_true", help="Independent confirmation of finalists")
    parser.add_argument("--neighbors", action="store_true", help="Search adjacent counts, including fewer output workers")
    parser.add_argument("--groups", action="store_true", help="Search classifier quantization group sizes")
    parser.add_argument("--compare-original", type=Path, help="Include saved original binary and its best split in final confirmation")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if (args.final or args.neighbors or args.groups) and not args.refine:
        parser.error("--final/--neighbors/--groups require --refine")
    root = args.output.resolve() if args.output else ROOT / "results" / datetime.now().strftime("output_tune_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    binary = root / "run-fused"
    subprocess.run(["gcc", "-Ofast", "-march=native", "-fopenmp", str(ROOT / "run.c"),
                    "-lm", "-o", str(binary)], check=True)
    sources = [ROOT / "run.c", *sorted((ROOT / "src").glob("*.h"))]
    metadata = {
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "compiler": subprocess.check_output(["gcc", "--version"], text=True),
        "affinity": sorted(os.sched_getaffinity(0)),
        "arguments": vars(args),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))
    if args.refine:
        previous = json.loads((args.refine / "ranking.json").read_text())
        selected = []
        for split in (False, True):
            selected.extend([row for row in previous if row["split"] == split][:3 if args.final else 2])
        cases = {}
        for row in selected:
            data = row["config"]
            data["cpus"] = tuple(data["cpus"])
            for field in ("binary", "model", "tokenizer"):
                data[field] = Path(data[field])
            c = replace(BenchmarkConfig(**data), binary=binary, output=None)
            schedules = (c.omp_schedule,) if args.final or args.neighbors or args.groups else ("static", "static,16", "dynamic,16", "guided,1", "guided,4", "guided,16", "guided,64")
            spins = (c.gomp_spincount,) if args.final or args.neighbors or args.groups else (None, 1000, 10000)
            for schedule in schedules:
                for spin in spins:
                    case = replace(c, omp_schedule=schedule, gomp_spincount=spin)
                    variants = [case]
                    if args.neighbors:
                        # Preserve CPU order and explore both directions of the split.
                        variants = [replace(case, threads=n, output_threads=m,
                                            persistent_openmp=True)
                                    for n in range(max(1, len(c.cpus) - 4), len(c.cpus) + 1)
                                    for m in range(max(1, len(c.cpus) - 4), len(c.cpus) + 1)]
                        variants.append(case)
                    if args.groups:
                        variants = [replace(case, group_size=g) for g in (24, 32, 48, 72, 96, 144, 288)]
                    for variant in variants:
                        cases[key(variant)] = variant
        cases = list(cases.values())
    else:
        cases = [replace(c, binary=binary) for c in candidates()]
    if args.final:
        best_split = next(c for c in cases if c.persistent_openmp and c.threads != c.output_threads)
        for n in (best_split.threads, best_split.output_threads):
            # Controls distinguish the benefit of persistence from the count split.
            cases.append(replace(best_split, threads=n, output_threads=n))
        # Also check conventional OpenMP with the split's larger worker count.
        cases.append(replace(best_split, threads=max(best_split.threads, best_split.output_threads),
                             output_threads=None, persistent_openmp=False))
    if args.compare_original:
        original = args.compare_original.resolve()
        previous = json.loads((original / "ranking.json").read_text())
        best = next(row for row in previous if row["split"])
        data = best["config"]
        data["cpus"] = tuple(data["cpus"])
        for field in ("binary", "model", "tokenizer"):
            data[field] = Path(data[field])
        cases.append(replace(BenchmarkConfig(**data), binary=original / "run-fused", output=None))
        initial = replace(candidates()[0], binary=original / "run-fused",
                          cpus=(0, 2, 1, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13),
                          threads=8, persistent_openmp=True, output_threads=14)
        cases.extend([initial, replace(initial, binary=binary)])
    cases = list({key(c): c for c in cases}.values())
    ac = list(Path("/sys/class/power_supply").glob("*/online"))

    def power():
        return {str(p): p.read_text().strip() for p in ac}

    print(f"Results: {root}; {len(cases)} cases x {args.repeats}", flush=True)
    rows = []
    rng = random.Random(42)
    run(replace(cases[0], output=root / "warmup"))
    reference_hash = hashlib.sha256((root / "warmup/stdout.txt").read_bytes()).hexdigest()
    for repeat in range(args.repeats):
        order = list(enumerate(cases))
        rng.shuffle(order)
        for i, c in order:
            dest = root / f"case_{i:03d}" / f"run_{repeat:02d}"
            before = power()
            summary = run(replace(c, output=dest))
            stdout_hash = hashlib.sha256((dest / "stdout.txt").read_bytes()).hexdigest()
            row = dict(case=i, repeat=repeat, config=asdict(c), power_before=before,
                       power_after=power(), stdout_sha256=stdout_hash, **summary)
            rows.append(row)
            with (root / "runs.jsonl").open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if stdout_hash != reference_hash:
                raise RuntimeError(f"Generated output differs from warmup: {dest}")
            print(f"{repeat + 1}/{args.repeats} case={i:03d} "
                  f"{'pool' if c.persistent_openmp else 'omp'} {c.threads}->{c.output_threads or c.threads} "
                  f"{c.omp_schedule} spin={c.gomp_spincount}: {summary['tokens_per_second']:.2f}", flush=True)
        ranking = []
        for i, c in enumerate(cases):
            samples = [r for r in rows if r["case"] == i]
            ms = [r["ms_per_token"] for r in samples]
            ranking.append(dict(case=i, config=asdict(c), split=c.persistent_openmp and c.threads != (c.output_threads or c.threads),
                                tokens_per_second=1000 / statistics.mean(ms),
                                ms_per_token=statistics.mean(ms), stdev_ms=statistics.stdev(ms) if len(ms) > 1 else 0,
                                stdout_sha256=sorted({r["stdout_sha256"] for r in samples})))
        ranking.sort(key=lambda r: r["ms_per_token"])
        (root / "ranking.json").write_text(json.dumps(ranking, default=str, indent=2))
        for split in (False, True):
            print(f"Top split={split}", flush=True)
            for row in [r for r in ranking if r["split"] == split][:5]:
                print(f"  case={row['case']:03d} {row['tokens_per_second']:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
