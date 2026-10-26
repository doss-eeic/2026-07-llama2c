"""Replay the 11 cases in output_tune_final, preserving case IDs and order seed."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

from cpuprofiler.benchmarkconfig import ROOT, BenchmarkConfig
from cpuprofiler.benchmark_distribute import run

dest = Path(__file__).resolve().parent
previous = ROOT / "results/output_tune_final"
if (dest / "runs.jsonl").exists():
    raise RuntimeError("Results already exist; use a new directory")
binary = dest / "run-fused"
source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [ROOT / "run.c", *sorted((ROOT / "src").glob("*.h"))]}
assert source_hashes == json.loads((previous / "metadata.json").read_text())["sources_sha256"]
subprocess.run(["gcc", "-Ofast", "-march=native", "-fopenmp", str(ROOT / "run.c"),
                "-lm", "-o", str(binary)], check=True)
cases = {}
for line in (previous / "runs.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["case"] in cases:
        continue
    data = row["config"]
    data["cpus"] = tuple(data["cpus"])
    for name in ("binary", "model", "tokenizer"):
        data[name] = Path(data[name])
    if data["binary"].parent == previous:
        data["binary"] = binary
    cases[row["case"]] = replace(BenchmarkConfig(**data), output=None, build=False)
cases = dict(sorted(cases.items()))
power_paths = list(Path("/sys/class/power_supply").glob("*/online"))
def power():
    return {str(p): p.read_text().strip() for p in power_paths}
metadata = dict(sources_sha256=source_hashes, previous=str(previous), repeats=20, seed=42,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                cpu=subprocess.check_output(["lscpu"], text=True),
                compiler=subprocess.check_output(["gcc", "--version"], text=True),
                model_sha256=hashlib.sha256(next(iter(cases.values())).model.read_bytes()).hexdigest())
(dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Results: {dest}; {len(cases)} cases x 20", flush=True)
run(replace(cases[0], output=dest / "warmup"))
expected_hash = hashlib.sha256((previous / "warmup/stdout.txt").read_bytes()).hexdigest()
assert hashlib.sha256((dest / "warmup/stdout.txt").read_bytes()).hexdigest() == expected_hash
rows = []
rng = random.Random(42)
for repeat in range(20):
    order = list(cases)
    rng.shuffle(order)
    for i in order:
        c = cases[i]
        output = dest / f"case_{i:03d}" / f"run_{repeat:02d}"
        before = power()
        summary = run(replace(c, output=output))
        digest = hashlib.sha256((output / "stdout.txt").read_bytes()).hexdigest()
        row = dict(case=i, repeat=repeat, config=asdict(c), power_before=before,
                   power_after=power(), stdout_sha256=digest, **summary)
        rows.append(row)
        with (dest / "runs.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        assert digest == expected_hash, output
        print(f"{repeat+1:02d}/20 case={i:03d}: {summary['tokens_per_second']:.2f} tok/s", flush=True)
    ranking = []
    for i, c in cases.items():
        samples = [r["ms_per_token"] for r in rows if r["case"] == i]
        ranking.append(dict(case=i, config=asdict(c),
                            split=c.persistent_openmp and c.threads != (c.output_threads or c.threads),
                            tokens_per_second=1000 / statistics.mean(samples),
                            ms_per_token=statistics.mean(samples),
                            stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0))
    ranking.sort(key=lambda r: r["ms_per_token"])
    (dest / "ranking.json").write_text(json.dumps(ranking, default=str, indent=2) + "\n")
    print("Ranking: " + ", ".join(f"{r['case']:03d}={r['tokens_per_second']:.2f}" for r in ranking), flush=True)
