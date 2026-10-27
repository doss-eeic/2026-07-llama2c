#!/usr/bin/env python3
"""How kasikoi2 searched: best layout per CPU count under the default settings, the layouts it tuned, and temperature.

usage: python3 layout_search.py SESSION_DIR      (reads runs.jsonl written by the recording wrapper)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = [json.loads(l) for l in open(Path(sys.argv[1]) / "runs.jsonl")]


def group(r):
    a = r["args"]
    return a[a.index("-g") + 1] if "-g" in a else "-"


def is_default(r):
    """BenchmarkConfig defaults, used by kasikoi2 for calibration and the CPU search."""
    e = r["env"]
    return (e.get("OMP_SCHEDULE") == "dynamic,16" and "OMP_WAIT_POLICY" not in e and "GOMP_SPINCOUNT" not in e
            and e.get("OMP_DYNAMIC") == "FALSE" and e.get("OMP_PROC_BIND") == "close"
            and e.get("OMP_PLACES") == ",".join(f"{{{c}}}" for c in r["cpus"]) and group(r) == "144")


for ph in ("search1", "search2"):
    rs = [r for r in runs if r["phase"] == ph and r["exit"] == 0 and r["tok_s"] and r["tok_s"] > 0]
    fifth = max(1, len(rs) // 5)
    temps = [max(r["temp_after"] for r in rs[i:i + fifth]) for i in range(0, fifth * 5, fifth)]
    print(f"== {ph}: {len(rs)} runs, highest temperature in each fifth of the search: {temps} C")
    best = {}
    for r in rs:
        if is_default(r):
            n = len(r["cpus"])
            if n not in best or r["tok_s"] > best[n]["tok_s"]:
                best[n] = r
    print("  fastest layout per CPU count with the default settings (dynamic,16, no wait policy, g144):")
    for n in sorted(best):
        r = best[n]
        print(f"    {n:>2} CPUs {r['tok_s']:7.0f} tok/s  temp {r['temp_before']:>3} C  cpus={','.join(map(str, r['cpus']))}")
    tuned = defaultdict(int)
    for r in rs:
        if not is_default(r):
            tuned[",".join(map(str, r["cpus"]))] += 1
    print("  layouts measured with other settings (runtime tuning), number of runs:")
    for k, v in sorted(tuned.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>3}  cpus={k}")
    twelve = [r for r in rs if r["cpus"] == list(range(12))]
    print(f"  runs on CPUs 0-11: {len(twelve)}, with ACTIVE or a spin count: "
          f"{sum('OMP_WAIT_POLICY' in r['env'] or 'GOMP_SPINCOUNT' in r['env'] for r in twelve)}, "
          f"with static: {sum(r['env'].get('OMP_SCHEDULE') == 'static' for r in twelve)}")
