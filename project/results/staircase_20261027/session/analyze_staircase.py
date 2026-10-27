#!/usr/bin/env python3
"""usage: python3 analyze_staircase.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)
Prints summary tables and writes staircase.csv (one row per main step) into the first directory."""
import csv, json, os, random, statistics as st, sys
from collections import defaultdict

rows, label = [], {}
for s, d in enumerate(sys.argv[1:], 1):
    for l in open(os.path.join(d, "runs.jsonl")):
        r = json.loads(l); r["session"] = s; rows.append(r)
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("cand "):
            f = l[5:].rstrip("\n").split("|"); label.setdefault(f[0], f[7])
ok = [r for r in rows if r["exit"] == 0 and r["tok_s"] and r["tok_s"] > 0]
by = defaultdict(dict)
for r in ok:
    by[r["id"]][(r["session"], r["block"])] = r["tok_s"]
rng = random.Random(42)
ids = [i for i in label if i in by]


def med(i):
    return st.median(by[i].values())


def paired(a, b, n=4000):
    rs = [by[a][k] / by[b][k] for k in by[a] if k in by[b]]
    m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), m[int(0.025 * n)], m[int(0.975 * n)], sum(x > 1 for x in rs), len(rs)


def ci(v, n=4000):
    m = sorted(st.median(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def fp(p):
    r, lo, hi, w, n = p
    return f"{r:6.3f} [{lo:.3f}, {hi:.3f}] {w:>2}/{n:<2}" + (" *" if lo > 1 or hi < 1 else "  ")


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   failed or invalid: {len(rows) - len(ok)}")
shas = {r["sha"] for r in ok}
print(f"distinct generated texts: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK)"))
niv = [r["nivcsw"] for r in ok]; oth = [r["other_cpu_s"] for r in ok]
tb = [r["temp_before"] for r in ok]
print(f"quietness: involuntary context switches per run median {st.median(niv):.0f}, p90 {sorted(niv)[int(0.9 * len(niv))]}; "
      f"other processes' CPU s per run median {st.median(oth):.3f}; temperature before runs {min(tb)}-{max(tb)} C")

base = "S1_upstream"
print(f"\n== all candidates (tok/s; 'team' = 1000 / mean ms per token) ==")
print(f"  {'id':<19}{'n':>4}{'median':>8}{'95% CI':>14}{'mean':>7}{'team':>7}{'min':>7}{'CV%':>6}{'x S1':>7}   label")
for i in ids:
    v = list(by[i].values()); lo, hi = ci(v)
    print(f"  {i:<19}{len(v):>4}{med(i):>8.0f}{f'[{lo:.0f},{hi:.0f}]':>14}{st.mean(v):>7.0f}{len(v) / sum(1 / x for x in v):>7.0f}"
          f"{min(v):>7.0f}{st.pstdev(v) / st.mean(v) * 100:>6.1f}{med(i) / med(base):>7.2f}   {label[i]}")

steps = [i for i in ids if i.startswith("S")]
print("\n== staircase: each step against the previous one (paired within block, median [95% CI] faster-blocks, * = CI excludes 1) ==")
out = []
prev = None
for i in steps:
    v = list(by[i].values())
    step = paired(i, prev) if prev else None
    total = paired(i, base) if i != base else None
    print(f"  {i:<14}{med(i):>7.0f} tok/s   step {fp(step) if step else '-':<32}   x S1 {total[0] if total else 1:.2f}   {label[i]}")
    out.append({"step": i, "label": label[i], "median_tok_s": round(med(i), 1), "mean_tok_s": round(st.mean(v), 1),
                "step_ratio": round(step[0], 4) if step else "", "step_ci_low": round(step[1], 4) if step else "",
                "step_ci_high": round(step[2], 4) if step else "", "ratio_to_upstream": round(total[0], 4) if total else 1.0,
                "n": len(v)})
    prev = i
with open(os.path.join(sys.argv[1], "staircase.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

Q = [("fusion alone (no hugepage) vs original", "F_fused_only", "S1_upstream"),
     ("hugepage alone vs original", "S2_hugepage", "S1_upstream"),
     ("hugepage on top of fusion", "S3_fused", "F_fused_only"),
     ("sakurada -w 5 vs master -w 1", "F_ours_w5", "S3_fused"),
     ("sakurada -w 5 -a 1 vs master -w 1 -a 1", "F_ours_w5a1", "S4_argmax"),
     ("ACTIVE on 9 CPUs (guided,4)", "S6_nine", "F_nine_default"),
     ("guided,4 vs static on 9 CPUs", "S6_nine", "F_nine_static"),
     ("12 threads: static vs guided,4", "S8_twelve", "F_twelve_guided4"),
     ("team's 9 vs 12 test: 9 guided,4 / 12 guided,4", "S6_nine", "F_twelve_guided4"),
     ("kasikoi2 pick vs its start (all 14, defaults)", "S7_kasikoi2", "F_all14_default"),
     ("original code: 0-11 vs 0-3", "F_upstream_twelve", "S1_upstream"),
     ("code effect on 0-11: S8 vs original on 0-11", "S8_twelve", "F_upstream_twelve")]
print("\n== factor checks (paired ratio first / second) ==")
for lab, a, b in Q:
    if a in by and b in by:
        print(f"  {lab:<48} {fp(paired(a, b))}")
print(f"\nstaircase.csv written to {sys.argv[1]}")
