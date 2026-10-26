#!/usr/bin/env python3
"""usage: python3 analyze_noise.py RESULT_DIR [RESULT_DIR ...]"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows = []
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
by = defaultdict(dict)  # (id, cond) -> (session, block) -> tok/s
for r in ok:
    by[(r["id"], r["cond"])][(r["session"], int(r["block"]))] = float(r["tok_s"])
IDS = ["N9_gd4", "PE12_gd4", "PE12_static", "LP12_gd4"]
CONDS = ["file", "sha", "py", "bg10", "bg50"]
rng = random.Random(42)


def paired(a, b, n=3000):
    A, B = by.get(a, {}), by.get(b, {})
    rs = [A[k] / B[k] for k in A if k in B]
    if not rs:
        return None
    m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), m[int(0.025 * n)], m[int(0.975 * n)], sum(x > 1 for x in rs), len(rs)


def fp(p):
    if p is None:
        return f"{'-':>26}"
    r, lo, hi, w, n = p
    return f"{r:6.3f}{'*' if lo > 1 or hi < 1 else ' '}[{lo:.3f},{hi:.3f}] {w:>2}/{n:<2}"


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows:
    shas[r["sha"]].add((r["id"], r["cond"]))
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
tb = [int(r["temp_before"]) for r in rows if r["temp_before"].isdigit()]
th = sum(int(r["throttle_delta"]) for r in rows if r["throttle_delta"].lstrip("-").isdigit())
print(f"temperature before runs min/median/max {min(tb)}/{st.median(tb):.0f}/{max(tb)} C   throttle events {th}")

for label, f in (("median tok/s", st.median), ("min tok/s", min), ("CV %", lambda v: st.pstdev(v) / st.mean(v) * 100)):
    print(f"\n== {label} ==")
    print(f"  {'':<14}" + "".join(f"{c:>9}" for c in CONDS))
    for i in IDS:
        print(f"  {i:<14}" + "".join(f"{f(list(by[(i, c)].values())):>9.{1 if label.startswith('CV') else 0}f}" if by.get((i, c)) else f"{'-':>9}" for c in CONDS))

print("\n== each condition relative to 'file' (paired within block, median [95% CI] faster-blocks) ==")
for i in IDS:
    print(f"  {i:<14}" + "".join(f"{c}: {fp(paired((i, c), (i, 'file')))}  " for c in CONDS[1:]))

print("\n== 9 CPUs vs the others under each condition: ratio N9_gd4 / other (team saw 1.10 for 0-11, 1.12 for LP-E) ==")
print(f"  {'':<18}" + "".join(f"{c:^28}" for c in CONDS))
for o in ("PE12_gd4", "PE12_static", "LP12_gd4"):
    print(f"  N9 / {o:<13}" + "".join(fp(paired(("N9_gd4", c), (o, c))) + "  " for c in CONDS))
