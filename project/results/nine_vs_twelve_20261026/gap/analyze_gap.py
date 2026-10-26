#!/usr/bin/env python3
"""usage: python3 analyze_gap.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows, cand, blocks = [], {}, {}
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
    for r in csv.DictReader(open(os.path.join(d, "blocks.tsv")), delimiter="\t"):
        blocks[(s, r["phase"], int(r["block"]))] = r
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("cand "):
            f = l[5:].rstrip("\n").split("|")
            cand.setdefault(f[0], f)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
for r in ok:
    r["tok"] = float(r["tok_s"])
PH = [p for p in ("cool", "hot", "recover") if any(r["phase"] == p for r in ok)]
ids = [i for i in cand if any(r["id"] == i for r in ok)]
by = defaultdict(lambda: defaultdict(dict))  # phase -> id -> (session, block) -> tok/s
for r in ok:
    by[r["phase"]][r["id"]][(r["session"], int(r["block"]))] = r["tok"]
rng = random.Random(42)
nan = float("nan")


def med(ph, i):
    v = list(by[ph][i].values())
    return st.median(v) if v else nan


def ci_median(v, n=3000):
    m = sorted(st.median(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def paired(ph, a, b, n=3000):
    A, B = by[ph].get(a, {}), by[ph].get(b, {})
    rs = [A[k] / B[k] for k in A if k in B]
    if not rs:
        return None
    lo, hi = ci_median(rs, n)
    return st.median(rs), lo, hi, sum(x > 1 for x in rs), len(rs)


def fmt_pair(p):
    if p is None:
        return f"{'-':>27}"
    r, lo, hi, w, n = p
    mark = "*" if lo > 1 or hi < 1 else " "
    return f"{r:6.3f}{mark}[{lo:.3f},{hi:.3f}] {w:>2}/{n:<2}"


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows:
    shas[r["sha"]].add(r["id"])
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else
      "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
print("\n== conditions per phase ==")
for ph in PH:
    R = [r for r in rows if r["phase"] == ph]
    tb = [int(r["temp_before"]) for r in R if r["temp_before"].isdigit()]
    th = sum(int(r["throttle_delta"]) for r in R if r["throttle_delta"].lstrip("-").isdigit())
    W = [float(r["energy_uj"]) / 1e6 / float(r["wall_s"]) for r in R if r["energy_uj"] not in ("NA", "") and float(r["wall_s"]) > 0]
    B = [b for k, b in blocks.items() if k[1] == ph]
    waits = [int(b["wait_s"]) for b in B]
    print(f"  {ph:<8} blocks {len(B):>3}  temp before run min/median/max {min(tb)}/{st.median(tb):.0f}/{max(tb)} C"
          f"  throttle events {th}" + (f"  package W median {st.median(W):.1f}" if W else "")
          + (f"  rest per block median {st.median(waits):.0f}s" if any(waits) else ""))

base = "B_up_P4"
for ph in PH:
    order = sorted(ids, key=lambda i: -med(ph, i))
    print(f"\n== ranking, phase {ph} (tok/s; 'team' = 1000 / mean ms per token, the team's statistic) ==")
    print(f"  {'id':<20}{'n':>4}{'median':>8}{'95% CI':>14}{'mean':>7}{'team':>7}{'min':>7}{'CV%':>6}{'xbase':>7}{'drops':>6}   note")
    cm = {i: med("cool", i) if "cool" in by else med(ph, i) for i in ids}
    for i in order:
        v = list(by[ph][i].values())
        if not v:
            continue
        lo, hi = ci_median(v, 2000)
        hm = len(v) / sum(1 / x for x in v)
        drops = sum(x < 0.85 * cm[i] for x in v)
        print(f"  {i:<20}{len(v):>4}{med(ph, i):>8.0f}{f'[{lo:.0f},{hi:.0f}]':>14}{st.mean(v):>7.0f}{hm:>7.0f}{min(v):>7.0f}"
              f"{st.pstdev(v) / st.mean(v) * 100:>6.1f}{med(ph, i) / med(ph, base):>7.2f}{drops:>6}   {cand[i][7]}")
print("  (drops = runs below 0.85 x the candidate's cool-phase median)")

Q = [
    ("Q1 team's own test: 9 / 12, both guided,4", "S_N9_gd4", "S_PE12_gd4"),
    ("Q2 each at my best: 12 static / 9 guided,4", "S_PE12_static", "S_N9_gd4"),
    ("Q3 12: guided,4 / static", "S_PE12_gd4", "S_PE12_static"),
    ("Q3 9: guided,4 / static", "S_N9_gd4", "S_N9_static"),
    ("Q3 12: guided,16 / static", "S_PE12_gd16", "S_PE12_static"),
    ("Q4 12 static: g144 / g288", "G_PE12_static_g144", "S_PE12_static"),
    ("Q5 LP-E 12 guided,4 / 9 guided,4", "X_LP12_gd4", "S_N9_gd4"),
    ("Q6 ACTIVE effect, 9 guided,4", "S_N9_gd4", "W_N9_gd4_def"),
    ("Q6 ACTIVE effect, 12 static", "S_PE12_static", "W_PE12_static_def"),
    ("Q6 no ACTIVE: 12 static / 9 guided,4", "W_PE12_static_def", "W_N9_gd4_def"),
    ("Q7 persistent 9->10 / normal 9 g4", "P_9to10_gd4", "S_N9_gd4"),
    ("Q7 persistent 9->10 / normal 12 static", "P_9to10_gd4", "S_PE12_static"),
    ("Q7 persistent 12 / normal 12 (static)", "P_PE12_static", "S_PE12_static"),
    ("Q8 check_top#2 guided,16 / 9 guided,4", "X_T9b_gd16", "S_N9_gd4"),
    ("Q8 0,2,4-11 guided,4 / 9 guided,4", "X_PE10_gd4", "S_N9_gd4"),
]
print("\n== questions: paired ratio first/second within the same block, median [95% CI] first-faster blocks ==")
print("   (* = CI excludes 1)")
print(f"  {'':<42}" + "".join(f"{ph:^28}" for ph in PH))
for label, a, b in Q:
    print(f"  {label:<42}" + "".join(fmt_pair(paired(ph, a, b)) + " " for ph in PH))
for ph in PH:
    b12 = max((i for i in ids if i.startswith("S_PE12_")), key=lambda i: med(ph, i), default=None)
    b9 = max((i for i in ids if i.startswith("S_N9_")), key=lambda i: med(ph, i), default=None)
    if b12 and b9:
        print(f"  best schedule each ({ph}): {b12} / {b9}  {fmt_pair(paired(ph, b12, b9))}")

print("\n== schedule curve: median tok/s relative to the same CPU set with static ==")
scheds = ["static", "dyn16", "gd1", "gd4", "gd16"]
for ph in PH:
    print(f"  phase {ph}: {'':<6}" + "".join(f"{s:>8}" for s in scheds))
    for cs in ("N9", "PE12"):
        ref = med(ph, f"S_{cs}_static")
        print(f"    {cs:<14}" + "".join(f"{med(ph, f'S_{cs}_{s}') / ref:>8.3f}" for s in scheds) + f"   (static = {ref:.0f})")

print("\n== temperature: paired ratio by the temperature just before the pair (all phases pooled) ==")
temp = {(r["session"], r["phase"], int(r["block"]), r["id"]): int(r["temp_before"]) for r in rows if r["temp_before"].isdigit()}
bins = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 200)]
for label, a, b in (("12 static / 9 guided,4", "S_PE12_static", "S_N9_gd4"), ("12 guided,4 / 9 guided,4", "S_PE12_gd4", "S_N9_gd4"),
                    ("persistent 9->10 / 12 static", "P_9to10_gd4", "S_PE12_static")):
    cells = []
    for lo, hi in bins:
        rs = []
        for ph in PH:
            A, B = by[ph].get(a, {}), by[ph].get(b, {})
            for k in A:
                if k in B:
                    t = (temp.get((k[0], ph, k[1], a), 0) + temp.get((k[0], ph, k[1], b), 0)) / 2
                    if lo <= t < hi:
                        rs.append(A[k] / B[k])
        cells.append(f"{st.median(rs):.3f} (n{len(rs)})" if rs else "-")
    print(f"  {label:<30}" + "".join(f"{f'{lo}-{hi}C' if hi < 200 else f'>={lo}C':>8}: {c:<14}" for (lo, hi), c in zip(bins, cells)))
print("  tok/s median by temperature before the run:")
for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4", "X_T9b_gd16"):
    if i not in ids:
        continue
    cells = []
    for lo, hi in bins:
        v = [r["tok"] for r in ok if r["id"] == i and r["temp_before"].isdigit() and lo <= int(r["temp_before"]) < hi]
        cells.append(f"{st.median(v):.0f} (n{len(v)})" if v else "-")
    print(f"  {i:<30}" + "".join(f"{f'{lo}-{hi}C' if hi < 200 else f'>={lo}C':>8}: {c:<14}" for (lo, hi), c in zip(bins, cells)))

if "hot" in by:
    print("\n== hot phase block by block: tok/s relative to the cool median ==")
    show = [i for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4") if i in ids]
    print(f"  {'block':>5}{'temp':>6}" + "".join(f"{i:>16}" for i in show))
    for (s, ph, b), blk in sorted(blocks.items()):
        if ph != "hot":
            continue
        print(f"  {b:>5}{blk['temp_start']:>6}" + "".join(
            f"{by['hot'][i].get((s, b), nan) / (med('cool', i) if 'cool' in by else nan):>16.3f}" for i in show))

if len(sys.argv) > 2:
    print("\n== per-session median tok/s (reproducibility) ==")
    for ph in PH:
        for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4"):
            if i in ids:
                print(f"  {ph:<8}{i:<18}" + "  ".join(
                    f"s{s}: {st.median([v for k, v in by[ph][i].items() if k[0] == s]):.0f}"
                    for s in range(1, len(sys.argv)) if any(k[0] == s for k in by[ph][i])))
