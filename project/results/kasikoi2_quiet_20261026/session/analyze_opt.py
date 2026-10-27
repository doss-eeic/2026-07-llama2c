#!/usr/bin/env python3
"""usage: python3 analyze_opt.py RESULT_DIR"""
import json, random, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
runs = [json.loads(l) for l in open(d / "runs.jsonl")]
rng = random.Random(42)
bad = [r for r in runs if r["exit"] != 0 or r["tok_s"] is None or r["tok_s"] <= 0]
print(f"benchmark runs recorded: {len(runs)}   failed or invalid tok/s (measured again): {len(bad)}")
runs = [r for r in runs if r not in bad]
shas = {r["sha"] for r in runs if r["phase"] != "setup"}
print(f"distinct generated texts: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK)"))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


print("\n== quietness per phase (from the wrapper) ==")
print("  nivcsw = times another task took a CPU from llama2c during one run; other = CPU seconds used by everything else")
by_phase = defaultdict(list)
for r in runs:
    by_phase[r["phase"]].append(r)
for ph, rs in by_phase.items():
    niv = [r["nivcsw"] for r in rs]
    oth = [r["other_cpu_s"] for r in rs]
    print(f"  {ph:<16} runs {len(rs):>4}  nivcsw median {st.median(niv):>5.0f} p90 {q(niv, 0.9):>5}  "
          f"other CPU s median {st.median(oth):.3f} p90 {q(oth, 0.9):.3f}  temp max {max(r['temp_after'] for r in rs)} C")

print("\n== kasikoi2 searches ==")
for log in sorted(d.glob("search*.log")):
    t = log.read_text()
    best = re.search(r"^Best configuration: (.*)$", t, re.M)
    perf = re.search(r"^Best performance: ([\d.]+)", t, re.M)
    srch = re.search(r"^Search: ([\d.]+) sec, (\d+) measurements", t, re.M)
    print(f"  {log.stem}: " + (f"{float(perf.group(1)):.0f} tok/s, {srch.group(1)} s, {srch.group(2)} runs" if perf and srch else "did not finish"))
    if best:
        b = best.group(1)
        keep = [m.group(0) for m in re.finditer(r"(cpus=\([^)]*\)|threads=\d+|omp_\w+=(?:'[^']*'|[^,)]+)|gomp_spincount=[^,)]+|group_size=\d+)", b)]
        print("    " + ", ".join(keep))

cf = d / "confirm.jsonl"
if cf.exists():
    rows = [json.loads(l) for l in open(cf)]
    by = defaultdict(dict)
    for r in rows:
        by[r["name"]][r["block"]] = r["tok_s"]
    med = {n: st.median(v.values()) for n, v in by.items()}
    order = sorted(by, key=lambda n: -med[n])
    best = order[0]
    print(f"\n== confirmation ({len({r['block'] for r in rows})} blocks, random order; 'team' = 1000 / mean ms per token) ==")
    print(f"  {'name':<22}{'n':>4}{'median':>8}{'mean':>8}{'team':>8}{'min':>7}{'CV%':>6}   vs best (paired median [95% CI] faster-blocks)")
    for n in order:
        v = list(by[n].values())
        hm = len(v) / sum(1 / x for x in v)
        line = f"  {n:<22}{len(v):>4}{med[n]:>8.0f}{st.mean(v):>8.0f}{hm:>8.0f}{min(v):>7.0f}{st.pstdev(v) / st.mean(v) * 100:>6.1f}"
        if n != best:
            rs = [by[n][k] / by[best][k] for k in by[n] if k in by[best]]
            m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(3000))
            line += f"   {st.median(rs):.3f} [{m[75]:.3f},{m[2924]:.3f}] {sum(x > 1 for x in rs)}/{len(rs)}"
        print(line)
    cands = d / "confirm_candidates.txt"
    if cands.exists():
        print("\n  candidates:")
        for l in open(cands):
            n, c = l.rstrip("\n").split("\t", 1)
            keep = [m.group(0) for m in re.finditer(r"(cpus=\([^)]*\)|threads=\d+|omp_\w+=(?:'[^']*'|[^,)]+)|gomp_spincount=[^,)]+|group_size=\d+)", c)]
            print(f"    {n:<22}" + ", ".join(keep))
