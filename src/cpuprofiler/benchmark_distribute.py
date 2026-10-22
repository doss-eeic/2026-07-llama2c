"""BenchmarkConfigで設定し、現在のrun.cをビルド・測定する。"""

import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import random
import statistics
import subprocess

if __package__:
    from .benchmarkconfig import ROOT, BenchmarkConfig
else:
    from benchmarkconfig import ROOT, BenchmarkConfig


def project_path(path: Path) -> Path:
    """相対パスはリポジトリのルートから解決する。"""
    return (ROOT / path).resolve()


def run(config: BenchmarkConfig) -> dict[str, float]:
    """1回測定し、設定・標準出力・標準エラー・集計をoutputへ保存する。"""
    if config.output is None:
        raise ValueError("output must be specified for saved benchmark results")
    available = os.sched_getaffinity(0)
    if not config.cpus or len(set(config.cpus)) != len(config.cpus):
        raise ValueError("cpus must contain distinct CPU IDs")
    if not set(config.cpus).issubset(available):
        raise ValueError(f"cpus must be available in current affinity: {sorted(available)}")
    if config.threads < 1 or config.threads > len(config.cpus):
        raise ValueError("threads must be between 1 and the number of CPUs")
    if config.steps < 2 or config.timeout <= 0:
        raise ValueError("steps must be at least 2 and timeout must be positive")

    binary = project_path(config.binary)
    model = project_path(config.model)
    tokenizer = project_path(config.tokenizer)
    for path in (model, tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = project_path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(asdict(config), default=str, ensure_ascii=False, indent=2) + "\n"
    )
    if config.build:
        binary.parent.mkdir(parents=True, exist_ok=True)
        with (output / "build.log").open("w") as log:
            subprocess.run([
                "gcc", "-Ofast", "-march=native", "-fopenmp",
                str(ROOT / "run.c"), "-lm", "-o", str(binary),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=config.timeout)
    if not binary.is_file():
        raise FileNotFoundError(binary)

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMP_", "GOMP_", "OUTPUT_"))}
    env.update({
        "OMP_NUM_THREADS": str(config.threads),
        "OMP_SCHEDULE": config.omp_schedule,
        "OMP_DYNAMIC": "TRUE" if config.omp_dynamic else "FALSE",
        "OMP_PROC_BIND": config.omp_proc_bind,
        "OMP_PLACES": config.omp_places or ",".join(f"{{{cpu}}}" for cpu in config.cpus),
    })
    command = [
        "taskset", "-c", ",".join(map(str, config.cpus)), str(binary), str(model),
        "-z", str(tokenizer), "-n", str(config.steps), "-t", str(config.temperature),
        "-s", str(config.seed), "-i", config.prompt,
    ]
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                       check=True, timeout=config.timeout)
    match = re.search(r"achieved tok/s:\s*([\d.]+)", (output / "stderr.log").read_text())
    if match is None:
        raise RuntimeError(f"Benchmark result not found: {output / 'stderr.log'}")
    tps = float(match.group(1))
    if not math.isfinite(tps) or tps <= 0:
        raise RuntimeError(f"Invalid throughput: {tps}")
    summary = {"tokens_per_second": tps, "ms_per_token": 1000.0 / tps}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    # ここで測定条件を編集する。環境変数の設定や手動ビルドは不要。
    output_root = ROOT / "results" / datetime.now().strftime("distribute_%Y%m%d_%H%M%S_%f")
    configs = {
        "P4": BenchmarkConfig(cpus=(0, 2, 1, 3), threads=4,
                              output=output_root / "P4"),
        "P4_E4": BenchmarkConfig(cpus=(0, 2, 1, 3, 4, 5, 8, 9), threads=8,
                                 output=output_root / "P4_E4"),
        "P2_E8": BenchmarkConfig(cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11), threads=10,
                                 output=output_root / "P2_E8"),
        "All14": BenchmarkConfig(cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 1, 3, 12, 13),
                                 threads=14, output=output_root / "All14"),
    }
    repeat = 10
    print(f"Results: {output_root}", flush=True)
    rows = []
    jobs = [
        (name, config, i)
        for i in range(repeat)
        for name, config in configs.items()
    ]

    random.Random(42).shuffle(jobs)
    rates = {name: [] for name in configs}
    built = set()
    for name, config, i in jobs:
        binary = project_path(config.binary)
        should_build = config.build and binary not in built
        summary = run(replace(config, output=config.output / f"run_{i + 1:02d}",
                              build=should_build))
        if should_build:
            built.add(binary)
        rates[name].append(summary["tokens_per_second"])
        rows.append({"config": name, "run": i + 1,
                     "cpus": ",".join(map(str, config.cpus)), "threads": config.threads,
                     "schedule": config.omp_schedule, **summary})
        print(f"{name} run {i + 1}: {summary['tokens_per_second']:.2f} tokens/s", flush=True)
        with (output_root / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, values in rates.items():
        print(f"{name}: mean={statistics.mean(values):.2f}, "
              f"median={statistics.median(values):.2f} tokens/s", flush=True)

if __name__ == "__main__":
    main()
