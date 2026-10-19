"""autotunerから受けた設定で1回実行し、時間集計を返す。"""

import json
import os
import statistics
import subprocess

from cpuprofiler.benchmarkconfig import BenchmarkConfig, ROOT


def parse_profile(text):
    """Read our JSONL output; keep only completion and count checks."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or rows[0]["event"] != "meta" or rows[-1]["event"] != "end":
        raise ValueError("incomplete profile log")
    
    timings = [r for r in rows if r["event"] == "timing"]
    end = rows[-1]
    
    forwards = sum(r["stage"] == "forward_total" for r in timings)
    if len(timings) != end["timing_records"] or forwards != end["forward_calls"] or not forwards:
        raise ValueError("profile record count mismatch")
    return {"meta": rows[0], "timings": timings,
            "tokens": [r for r in rows if r["event"] == "token"], "end": end}


def summarize(records, skip_tokens, selected_stages=None):
    forwards = [r for r in records if r["stage"] == "forward_total"]
    measured = [r for r in forwards if r["forward_id"] >= skip_tokens]
    if not measured:
        raise ValueError("no measured tokens remain after skip_tokens")
    totals = {}
    for record in records:
        if record["forward_id"] >= skip_tokens and (selected_stages is None or record["stage"] in selected_stages):
            totals.setdefault(record["stage"], []).append(record["duration_ns"] / 1e6)
    stages = [
        {"stage": stage, "calls": len(values),
         "mean_call_ms": statistics.mean(values),
         "ms_per_token": sum(values) / len(measured)}
        for stage, values in sorted(totals.items())
    ]
    durations = [r["duration_ns"] / 1e6 for r in measured]
    return {"forward_calls": len(forwards), "measured_tokens": len(measured),
            "forward_ms_per_token": statistics.mean(durations)}, stages


def run(config: BenchmarkConfig):
    """1回実行して生ログを保存し、(全体の集計, stage別の集計)を返す。"""
    output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = ["taskset", "-c", ",".join(map(str, config.cpus)), str(config.binary.resolve()),
               str(config.model.resolve()), "-z", str(config.tokenizer.resolve()),
               "-n", str(config.steps), "-t", str(config.temperature),
               "-s", str(config.seed), "-i", config.prompt]
    env = {**os.environ,
           "OMP_NUM_THREADS": str(config.threads),
           "OMP_DYNAMIC": "TRUE" if config.omp_dynamic else "FALSE",
           "OMP_PROC_BIND": config.omp_proc_bind,
           "OMP_PLACES": config.omp_places or ",".join(f"{{{cpu}}}" for cpu in config.cpus),
           "LLAMA_PROFILE_PATH": str(output / "profile.jsonl")}
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                       check=True, timeout=config.timeout)
    profile = parse_profile((output / "profile.jsonl").read_text())
    return summarize(profile["timings"], config.skip_tokens, config.stages)


def main():
    import csv
    from datetime import datetime

    configs = {
        "P1":     (0,),
        "P2":     (0, 2),
        "P4_HT":  (0, 1, 2, 3),
        "E4":     (4, 5, 6, 7),
        "X4_2_2": (4, 5, 8, 9),
        # P + E 混合
        "P2E2":   (0, 2, 4, 5),
        "P2E4":   (0, 2, 4, 5, 6, 7),

        # P + SMT + E 混合
        "P4E2":   (0, 1, 2, 3, 4, 5),
        "P4E4":   (0, 1, 2, 3, 4, 5, 6, 7),
        "P4X4":   (0, 1, 2, 3, 4, 5, 8, 9),

        # 全コア利用
        "P4E8":   tuple(range(12)),
    }

    output_root = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True)

    results = []

    for name, cpus in configs.items():
        config = BenchmarkConfig(
            cpus=cpus,
            threads=len(cpus),
            output=output_root / name,
        )

        summary, stages = run(config)

        row = {
            "config": name,
            "cpus": ",".join(map(str, cpus)),
            **summary,
            **{s["stage"]: s["ms_per_token"] for s in stages},
        }
        results.append(row)

        print(f"{name}: {summary['forward_ms_per_token']:.3f} ms/token")

    # CSV保存
    keys = list(dict.fromkeys(k for row in results for k in row))
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()