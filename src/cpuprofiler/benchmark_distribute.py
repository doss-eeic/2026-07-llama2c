"""stay版と本家未改造run.cのOpenMP版を同じ条件で測定する。"""

import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import re
import random
import statistics
import subprocess

if __package__:
    from .benchmarkconfig import ROOT, UPSTREAM_REVISION, BenchmarkConfig
else:
    from benchmarkconfig import ROOT, UPSTREAM_REVISION, BenchmarkConfig


def project_path(path: Path) -> Path:
    """相対パスはリポジトリのルートから解決する。"""
    return (ROOT / path).resolve()


def build_command(config: BenchmarkConfig, output: Path, binary: Path) -> list[str]:
    source = ROOT / "run.c"
    flags = ["-DLLAMA_STAY", "-pthread"]
    if config.backend == "upstream":
        # Compile the exact upstream file, not our run.c with stay disabled.
        contents = subprocess.check_output(
            ["git", "show", f"{UPSTREAM_REVISION}:run.c"], cwd=ROOT, timeout=config.timeout
        )
        source = output / "run.c"
        source.write_bytes(contents)
        (output / "source.json").write_text(json.dumps({
            "revision": UPSTREAM_REVISION, "path": "run.c",
            "sha256": hashlib.sha256(contents).hexdigest(),
        }, indent=2) + "\n")
        flags = ["-fopenmp"]
    return ["gcc", "-Ofast", "-march=native", *flags,
            str(source), "-lm", "-o", str(binary)]


def execution_environment(config: BenchmarkConfig) -> dict[str, str]:
    # Use the runtime defaults for waiting; don't inherit another experiment's overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OMP_", "GOMP_", "STAY_"))}
    if config.backend == "upstream":
        env.update({
            "OMP_NUM_THREADS": str(config.threads), "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in config.cpus),
        })
    else:
        env.update({
            "STAY_CPUS": ",".join(map(str, config.cpus)),
            "STAY_LAYER_THREADS": str(config.threads),
            "STAY_P_WEIGHT": str(config.p_weight),
        })
    return env


def run(config: BenchmarkConfig) -> dict[str, float]:
    """1回測定し、設定・標準出力・標準エラー・集計をoutputへ保存する。"""
    available = os.sched_getaffinity(0)
    if config.backend not in ("stay", "upstream"):
        raise ValueError("backend must be stay or upstream")
    if not config.cpus or len(set(config.cpus)) != len(config.cpus):
        raise ValueError("cpus must contain distinct CPU IDs")
    if not set(config.cpus).issubset(available):
        raise ValueError(f"cpus must be available in current affinity: {sorted(available)}")
    if config.threads < 1 or config.threads > len(config.cpus):
        raise ValueError("threads must be between 1 and the number of CPUs")
    if not 1 <= config.p_weight <= 16:
        raise ValueError("p_weight must be between 1 and 16")
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
        command = build_command(config, output, binary)
        with (output / "build.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=config.timeout)
    if not binary.is_file():
        raise FileNotFoundError(binary)

    env = execution_environment(config)
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
    cpus = (0, 1, 2, 3, 4, 5, 8, 9)
    output_root = ROOT / "results" / datetime.now().strftime("stay_%Y%m%d_%H%M%S_%f")
    configs = {
        "Upstream_P4_HT": BenchmarkConfig(cpus=cpus[:4], threads=4,
                                          output=output_root / "Upstream_P4_HT",
                                          binary=ROOT / "build/upstream-run",
                                          backend="upstream"),
        "Upstream_P4E4": BenchmarkConfig(cpus=cpus, threads=8,
                                         output=output_root / "Upstream_P4E4",
                                         binary=ROOT / "build/upstream-run",
                                         backend="upstream"),
        "P4_fixed": BenchmarkConfig(cpus=cpus[:4], threads=4,
                                    output=output_root / "P4_fixed", p_weight=1),
        "P4E4_fixed": BenchmarkConfig(cpus=cpus, threads=8,
                                      output=output_root / "P4E4_fixed", p_weight=1),
        "Switch_equal": BenchmarkConfig(cpus=cpus, threads=4,
                                        output=output_root / "Switch_equal", p_weight=1),
        "Switch_weight2": BenchmarkConfig(cpus=cpus, threads=4,
                                          output=output_root / "Switch_weight2", p_weight=2),
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
    built = {}
    for name, config, i in jobs:
        binary = project_path(config.binary)
        if binary in built and built[binary] != config.backend:
            raise ValueError("stay and upstream must use different binary paths")
        should_build = config.build and binary not in built
        summary = run(replace(config, output=config.output / f"run_{i + 1:02d}",
                              build=should_build))
        if should_build:
            built[binary] = config.backend
        rates[name].append(summary["tokens_per_second"])
        rows.append({"config": name, "run": i + 1,
                     "backend": config.backend,
                     "cpus": ",".join(map(str, config.cpus)), "threads": config.threads,
                     "p_weight": config.p_weight, **summary})
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
