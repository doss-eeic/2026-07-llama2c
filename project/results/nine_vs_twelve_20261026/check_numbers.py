#!/usr/bin/env python3
"""Recompute the numbers in README.md that are not printed by the per-session summary.txt files.

usage (from the repository root, after `git fetch origin`):
    python3 project/results/nine_vs_twelve_20261026/check_numbers.py

Part 1 reads this directory (gap/ and noise/).
Part 2 reads the team's records straight from pinned git commits:
    dev/wataru2 1e1fa037  results/master_threads_20261026/*.jsonl, results/output_tune_rerun_20261026/*
    dev/omp-tmp 736c7ebe  results/omp_tmp_*_20261026/{runs.jsonl,console.log}
"""
import csv
import json
import os
import re
import statistics as st
import subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WATARU2 = "1e1fa037adf0e40184da2ceb3c08ffe3b416a634"
OMPTMP = "736c7ebefc289fe02a5fb3b0a8bd04355ade8462"

print("== part 1: our sessions ==")
rows = list(csv.DictReader(open(os.path.join(HERE, "gap", "raw.tsv")), delimiter="\t"))
print("gap_check: mean CPU frequency (MHz) just before the run, per phase")
for ph in ("cool", "hot", "recover"):
    cells = []
    for i in ("S_N9_gd4", "S_PE12_static"):
        v = [int(r["freq_before"]) for r in rows if r["phase"] == ph and r["id"] == i]
        cells.append(f"{i} {st.mean(v):.0f}")
    print(f"  {ph:<8}" + "   ".join(cells))
trace = list(csv.DictReader(open(os.path.join(HERE, "gap", "trace.tsv")), delimiter="\t"))
print("gap_check: 2-second trace, temperature (C) and mean P/E frequency (MHz) per phase")
for ph in ("idle", "cool", "heat", "hot", "recover"):
    t = [int(r["temp"]) for r in trace if r["phase"] == ph]
    f = [int(r["freq_pe"]) for r in trace if r["phase"] == ph]
    if t:
        print(f"  {ph:<8} samples {len(t):>3}  temp mean {st.mean(t):.1f} max {max(t)}  freq mean {st.mean(f):.0f}")
env = open(os.path.join(HERE, "noise", "environment.txt")).read()
print("noise_check: last-level cache sharing recorded on the machine")
for cpu, lst in re.findall(r"^\s+cpu(\d+): (\S+)$", env, re.M):
    if cpu in ("0", "11", "12"):
        print(f"  cpu{cpu}: shares with {lst}")


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


print("\n== part 2: the team's records (pinned commits) ==")
try:
    git_show(WATARU2, "results/master_threads_20261026/bench9v0_11.jsonl")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("  git objects not available: run `git fetch origin` inside the repository first")
    raise SystemExit(0)

for name, label in (("bench9v0_11", "9 (0,2,4-10) vs 12 (0-11)"), ("bench9v12", "9 (0,2,4-10) vs 12 (0,2,4-13, LP-E)")):
    data = [json.loads(l) for l in git_show(WATARU2, f"results/master_threads_20261026/{name}.jsonl").splitlines()]
    print(f"master-threads {label}: {name}.jsonl")
    for c in ("9core", "12core"):
        v = [r["tps"] for r in data if r["case"] == c]
        thirds = []
        for lo, hi in ((0, 10), (10, 20), (20, 30)):
            thirds.append(st.median([r["tps"] for r in data if r["case"] == c and lo <= r["round"] < hi]))
        print(f"  {c:<7} n {len(v)}  mean {st.mean(v):.1f}  median {st.median(v):.0f}  min {min(v):.0f}  max {max(v):.0f}"
              f"  median of rounds 0-9 / 10-19 / 20-29: " + " / ".join(f"{x:.0f}" for x in thirds))
    by = defaultdict(dict)
    for r in data:
        by[r["round"]][r["case"]] = r["tps"]
    rs = [d["9core"] / d["12core"] for d in by.values()]
    print(f"  ratio 9core / 12core per round: median {st.median(rs):.3f}, 12 faster in {sum(x < 1 for x in rs)}/{len(rs)} rounds")

comp = json.loads(git_show(WATARU2, "results/output_tune_rerun_20261026/comparison.json"))
blk = comp["block_tokens_per_second"]
print("output-tuning rerun: tok/s of all 11 configurations per round")
print("  " + "  ".join(f"{k}:{v:.0f}" for k, v in sorted(blk.items(), key=lambda kv: int(kv[0]))))
first = json.loads(git_show(WATARU2, "results/output_tune_rerun_20261026/runs.jsonl").splitlines()[0])
print("  power supplies before the first run:", {k.split("/")[-2]: v for k, v in first["power_before"].items()})

print("dev/omp-tmp benchmarks on the same day (UTC; JST = UTC+9)")
for d in ("omp_tmp_coarse_20261026", "omp_tmp_tune_20261026", "omp_tmp_final_20261026"):
    runs = [json.loads(l) for l in git_show(OMPTMP, f"results/{d}/runs.jsonl").splitlines()]
    ac = runs[0]["before"]["ac"]
    print(f"  {d:<26} runs {len(runs):>3}  {runs[0]['finished_utc'][11:19]} - {runs[-1]['finished_utc'][11:19]}"
          f"  supplies {({k.split('/')[-2]: v for k, v in ac.items()})}")
paths = set()
for d in ("omp_tmp_coarse_20261026", "omp_tmp_tune_20261026", "omp_tmp_final_20261026"):
    for f in ("runs.jsonl", "console.log"):
        paths |= set(re.findall(r"/home/denjo/[A-Za-z0-9_.-]+", git_show(OMPTMP, f"results/{d}/{f}")))
paths |= set(re.findall(r"/home/denjo/[A-Za-z0-9_.-]+", git_show(WATARU2, "results/output_tune_rerun_20261026/runs.jsonl")))
print("  working directories that appear in these records:", sorted(paths))
log = subprocess.run(["git", "log", "-1", "--format=%ci %s%n%b", WATARU2], capture_output=True, text=True).stdout
print("commit 1e1fa037:", " | ".join(l for l in log.splitlines() if l.strip()))
