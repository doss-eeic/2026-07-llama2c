#!/usr/bin/env python3
"""usage: python3 analyze_rigorous.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows = []
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
for r in ok:
    r["tok"] = float(r["tok_s"])
ids = sorted({r["id"] for r in ok})
note = {}
for d in sys.argv[1:]:
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("C") and "|" in l:
            f = l.strip().split("|"); note[f[0]] = f"{f[1]} {f[3]} {f[4]} cpus={f[2]}"
by = defaultdict(dict)  # id -> (session, block) -> tok
for r in ok:
    by[r["id"]][(r["session"], r["block"])] = r["tok"]
blocks = sorted({k for v in by.values() for k in v})
rng = random.Random(42)

def boot_median(id_, n=4000):
    keys = list(by[id_]); vals = [by[id_][k] for k in keys]
    meds = sorted(st.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]

def paired(a, b, n=4000):
    keys = [k for k in by[a] if k in by[b]]
    rs = [by[a][k] / by[b][k] for k in keys]
    if not rs: return float("nan"), (float("nan"), float("nan")), 0, 0
    meds = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), (meds[int(0.025 * n)], meds[int(0.975 * n)]), sum(x > 1 for x in rs), len(rs)

print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows: shas[r["sha"]].add(r["id"])
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
thr = sum(int(r["throttle_delta"]) for r in rows if r["throttle_delta"].lstrip("-").isdigit())
temps = [int(r["temp_after"]) for r in rows if r["temp_after"].isdigit()]
print(f"thermal throttle events during runs: {thr}   max temperature seen: {max(temps) if temps else 'n/a'} C")

med = {i: st.median(by[i].values()) for i in ids}
order = sorted(ids, key=lambda i: -med[i])
best = order[0]
base = "C01_up_P4" if "C01_up_P4" in by else order[-1]
print(f"\n== ranking (median tok/s, block-bootstrap 95% CI, x baseline {base}) ==")
print(f"  {'id':<18}{'n':>4}{'median':>9}{'95% CI':>18}{'CV%':>6}{'x base':>8}   setting")
for i in order:
    v = list(by[i].values()); lo, hi = boot_median(i)
    cv = st.pstdev(v) / st.mean(v) * 100
    print(f"  {i:<18}{len(v):>4}{med[i]:>9.0f}{f'[{lo:.0f}, {hi:.0f}]':>18}{cv:>6.1f}{med[i] / med[base]:>8.2f}   {note.get(i, '')}")

print(f"\n== paired comparison with the fastest ({best}): ratio of other / best within the same block ==")
for i in order[1:]:
    r, (lo, hi), w, n = paired(i, best)
    verdict = "slower (CI below 1)" if hi < 1 else ("faster?" if lo > 1 else "not distinguishable")
    print(f"  {i:<18} {r:.3f}  CI [{lo:.3f}, {hi:.3f}]  faster than best in {w}/{n} blocks  -> {verdict}")

pairs = [("C03_team_T9b", "C07_team_PE12", "team 9 CPUs vs all 12"), ("C11_ours_PE12", "C07_team_PE12", "ours vs team (PE12)"),
         ("C12_ours_T9b", "C03_team_T9b", "ours vs team (9 CPUs)"), ("C03_team_T9b", "C06_team_T9b_def", "ACTIVE effect (team 9 CPUs)"),
         ("C02_up_P4_act", "C01_up_P4", "ACTIVE effect (original code)"), ("C08_team_PE12_g", "C07_team_PE12", "guided vs static (PE12)")]
print("\n== specific questions (paired, ratio first/second) ==")
for a, b, label in pairs:
    if a in by and b in by:
        r, (lo, hi), w, n = paired(a, b)
        print(f"  {label:<32} {r:.3f}  CI [{lo:.3f}, {hi:.3f}]  first faster in {w}/{n}")

print("\n== drift: median tok/s per quarter of the blocks, relative to the candidate's own median ==")
for s in sorted({k[0] for k in blocks}):
    bl = sorted(int(k[1]) for k in blocks if k[0] == s); q = max(1, len(bl) // 4)
    quarters = [bl[i:i + q] for i in range(0, len(bl), q)][:4]
    cells = []
    for qb in quarters:
        rel = [by[i][(s, str(b))] / med[i] for i in ids for b in qb if (s, str(b)) in by[i]]
        cells.append(f"{st.median(rel):.3f}")
    print(f"  session {s}: " + "  ".join(cells))
if len(sys.argv) > 2:
    print("\n== per-session median tok/s (reproducibility) ==")
    for i in order:
        print(f"  {i:<18}" + "  ".join(f"s{s}: {st.median([v for k, v in by[i].items() if k[0] == s]):.0f}" for s in range(1, len(sys.argv))))
