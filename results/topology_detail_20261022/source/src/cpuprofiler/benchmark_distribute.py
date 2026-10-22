"""stay版と本家未改造run.cのOpenMP版を同じ条件で測定する。"""

import csv
import gzip
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
import shutil

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
    if config.backend == "parallel":
        flags = ["-DLLAMA_PARALLEL", "-fopenmp"]
    if config.backend == "openmp":
        flags = ["-DLLAMA_PARALLEL", "-DLLAMA_EXPERIMENT", "-fopenmp"]
    if config.backend == "pthread":
        flags = ["-DLLAMA_STAY", "-DLLAMA_EXPERIMENT", "-pthread"]
    if config.backend == "upstream":
        # Compile the exact upstream file, not our run.c with stay disabled.
        contents = (ROOT / "run_detail.c").read_bytes() if config.profile else subprocess.check_output(
            ["git", "show", f"{UPSTREAM_REVISION}:run.c"], cwd=ROOT, timeout=config.timeout
        )
        source = output / "run.c"
        source.write_bytes(contents)
        (output / "source.json").write_text(json.dumps({
            "revision": "66870ea" if config.profile else UPSTREAM_REVISION,
            "path": "run_detail.c" if config.profile else "run.c",
            "sha256": hashlib.sha256(contents).hexdigest(),
        }, indent=2) + "\n")
        flags = ["-fopenmp"]
    if config.profile:
        flags.append("-DLLAMA_PROFILE")
    return ["gcc", "-Ofast", "-march=native", *flags,
            str(source), "-lm", "-o", str(binary)]


def execution_environment(config: BenchmarkConfig) -> dict[str, str]:
    # Use the runtime defaults for waiting; don't inherit another experiment's overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OMP_", "GOMP_", "STAY_", "PARALLEL_", "OUTPUT_"))}
    if config.backend in ("upstream", "parallel", "openmp"):
        env.update({
            "OMP_NUM_THREADS": str(config.threads), "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in config.cpus),
        })
        if config.backend != "upstream":
            env["OMP_SCHEDULE"] = config.omp_schedule
    else:
        env.update({
            "STAY_CPUS": ",".join(map(str, config.cpus)),
            "STAY_LAYER_THREADS": str(config.threads),
            "STAY_P_WEIGHT": str(config.p_weight),
        })
    if config.backend in ("parallel", "openmp", "pthread"):
        dynamic = config.omp_schedule.split(",")[0] == "dynamic"
        output_dynamic = config.output_dynamic
        if output_dynamic is None:
            output_dynamic = dynamic if config.backend != "parallel" else True
        parts = config.omp_schedule.split(",")
        env.update({"OUTPUT_THREADS": str(config.output_threads or config.threads),
                    "OUTPUT_CHUNK": str(config.output_chunk),
                    "OUTPUT_DYNAMIC": str(int(output_dynamic)),
                    "PARALLEL_FUSE": str(int(config.fuse)),
                    "PARALLEL_FUSE_QKV": str(int(config.fuse if config.fuse_qkv is None else config.fuse_qkv)),
                    "PARALLEL_FUSE_FFN": str(int(config.fuse if config.fuse_ffn is None else config.fuse_ffn)),
                    "PARALLEL_OUTPUT_BLOCK": str(config.output_block),
                    "PARALLEL_ATTENTION_THREADS": str(config.attention_threads),
                    "PARALLEL_DYNAMIC": str(int(dynamic)),
                    "PARALLEL_CHUNK": parts[1] if len(parts) > 1 else "16"})
    return env


def run(config: BenchmarkConfig) -> dict[str, float]:
    """1回測定し、設定・標準出力・標準エラー・集計をoutputへ保存する。"""
    available = os.sched_getaffinity(0)
    if config.backend not in ("stay", "upstream", "parallel", "openmp", "pthread"):
        raise ValueError("unknown backend")
    if config.backend in ("openmp", "pthread") and config.omp_schedule not in ("static", "dynamic,16"):
        raise ValueError("matched experiments support static or dynamic,16")
    if config.backend == "pthread" and config.output_threads not in (None, len(config.cpus)):
        raise ValueError("pthread output uses the entire pool in cpus")
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
    if config.skip_tokens < 0 or config.skip_tokens >= config.steps:
        raise ValueError("skip_tokens must be between 0 and steps - 1")
    if config.output_threads is not None and not 1 <= config.output_threads <= len(config.cpus):
        raise ValueError("output_threads must be between 1 and the number of CPUs")
    if config.output_chunk < 1 or not 0 <= config.attention_threads <= len(config.cpus):
        raise ValueError("output_chunk must be positive and attention_threads must fit cpus")
    if config.output_block not in (1, 4):
        raise ValueError("output_block must be 1 or 4")

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
    env.pop("LLAMA_PROFILE_PATH", None)
    if config.profile:
        env["LLAMA_PROFILE_PATH"] = str(output / "profile.jsonl")
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
    if config.profile:
        if __package__:
            from .benchmark_detail import parse_profile, summarize
        else:
            from benchmark_detail import parse_profile, summarize
        profile = parse_profile((output / "profile.jsonl").read_text())
        with (output / "profile.jsonl").open("rb") as source, gzip.open(output / "profile.jsonl.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        (output / "profile.jsonl").unlink()
        totals, stages = summarize(profile["timings"], config.skip_tokens)
        summary.update(totals)
        summary.update({stage["stage"] + "_ms_per_token": stage["ms_per_token"] for stage in stages})
        layer_totals = {}
        for record in profile["timings"]:
            if record["forward_id"] >= config.skip_tokens and record["layer"] >= 0:
                key = (record["layer"], record["stage"])
                layer_totals[key] = layer_totals.get(key, 0) + record["duration_ns"]
        layers = [{"layer": layer, "stage": stage,
                   "ms_per_token": ns / 1e6 / totals["measured_tokens"]}
                  for (layer, stage), ns in sorted(layer_totals.items())]
        for filename, records in (("stages.csv", stages), ("layers.csv", layers)):
            with (output / filename).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def save_profile_aggregates(output_root, rows):
    """反復の平均と、層・演算別の平均を保存する。"""
    metrics = list(dict.fromkeys(key for row in rows for key in row
                                if key.endswith("ms_per_token") or key == "tokens_per_second"))
    aggregates = []
    for name in dict.fromkeys(row["config"] for row in rows):
        values = [row for row in rows if row["config"] == name]
        means = {key: statistics.mean(row[key] for row in values if key in row)
                 for key in metrics if any(key in row for row in values)}
        if "qkv_ms_per_token" not in means:
            means["qkv_ms_per_token"] = sum(means[key + "_ms_per_token"] for key in ("q_proj", "k_proj", "v_proj"))
        if "ffn_up_gate_swiglu_ms_per_token" in means:
            means["ffn_matmul_swiglu_ms_per_token"] = means["ffn_up_gate_swiglu_ms_per_token"] + means["ffn_down_ms_per_token"]
        else:
            means["ffn_matmul_ms_per_token"] = sum(means[key + "_ms_per_token"] for key in ("ffn_up", "ffn_gate", "ffn_down"))
            means["ffn_matmul_swiglu_ms_per_token"] = sum(means[key + "_ms_per_token"] for key in ("ffn_up", "ffn_gate", "ffn_down", "swiglu"))
        means["outside_layers_output_ms_per_token"] = (
            means["forward_ms_per_token"] - means["layer_total_ms_per_token"] - means["output_ms_per_token"])
        aggregates.append({"config": name, "runs": len(values), **means})
    with (output_root / "aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in aggregates for k in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregates)
    layers = {}
    for path in output_root.glob("*/*/layers.csv"):
        with path.open() as file:
            for row in csv.DictReader(file):
                key = (path.parent.parent.name, int(row["layer"]), row["stage"])
                layers.setdefault(key, []).append(float(row["ms_per_token"]))
    with (output_root / "layers_aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["config", "layer", "stage", "runs", "ms_per_token"],
                                lineterminator="\n")
        writer.writeheader()
        for (name, layer, stage), values in sorted(layers.items()):
            writer.writerow({"config": name, "layer": layer, "stage": stage,
                             "runs": len(values), "ms_per_token": statistics.mean(values)})


def main(profile=False):
    # ここで測定条件を編集する。環境変数の設定や手動ビルドは不要。
    cpus = (0, 1, 2, 3, 4, 5, 8, 9)
    prefix = "stay_detail" if profile else "stay"
    output_root = ROOT / "results" / datetime.now().strftime(prefix + "_%Y%m%d_%H%M%S_%f")
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
    if profile:
        configs = {name: replace(config, profile=True,
                                binary=config.binary.with_name(config.binary.name + "-detail"))
                   for name, config in configs.items()}
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
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for name, values in rates.items():
        print(f"{name}: mean={statistics.mean(values):.2f}, "
              f"median={statistics.median(values):.2f} tokens/s", flush=True)
    if profile:
        save_profile_aggregates(output_root, rows)

if __name__ == "__main__":
    main()
