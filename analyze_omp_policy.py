#!/usr/bin/env python3
"""Summarize omp_policy_check.sh results.

raw.txt lines:
  RAW section repeat round config threads cpus w a policy schedule tok layers final sha
Writes a report to stdout and medians.csv next to raw.txt.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

raw = sys.argv[1]
keys = ["section", "repeat", "round", "config", "threads", "cpus", "w", "a", "policy", "sched",
        "tok", "layers", "final", "sha"]
runs = []
for line in open(raw):
    f = line.split()
    if len(f) != 15 or f[0] != "RAW" or "NA" in f[11:14]:
        continue
    d = dict(zip(keys, f[1:]))
    for k in ("tok", "layers", "final"):
        d[k] = float(d[k])
    d["forward"] = d["layers"] + d["final"]
    runs.append(d)

ORDER = ["P1", "P2", "P4_HT", "E4", "X4_2_2", "P2E4", "PE10", "PE12"]
POL = ["default", "PASSIVE", "ACTIVE"]
COMBO = [("0", "0"), ("5", "0"), ("5", "1")]
CNAME = {("0", "0"): "w0a0", ("5", "0"): "w5a0", ("5", "1"): "w5a1"}


def sel(**kw):
    return [r for r in runs if all(r[k] == v for k, v in kw.items())]


def med(rows, key):
    v = [r[key] for r in rows]
    return st.median(v) if v else float("nan")


def paired(new, old, key):
    a = {(r["repeat"], r["round"]): r[key] for r in old}
    b = {(r["repeat"], r["round"]): r[key] for r in new}
    rs = [(b[k] / a[k]) if key == "tok" else (a[k] / b[k]) for k in a if k in b]
    return (st.median(rs), sum(x > 1 for x in rs), len(rs)) if rs else (float("nan"), 0, 0)


out = []
p = out.append
for l in open(os.path.join(os.path.dirname(raw), "environment.txt")):
    if l.startswith("profile="):
        p("[" + l.strip() + "]")
        break

W = sel(section="W")
if W:
    cfgs = [c for c in ORDER if any(r["config"] == c for r in W)]
    pols = [x for x in POL if any(r["policy"] == x for r in W)]
    p("\n== W) OMP_WAIT_POLICY (static schedule; median over all rounds and repeats) ==")
    for key, label, fmt in (("tok", "tok/s", "{:.0f}"), ("layers", "Transformer blocks ms", "{:.3f}"),
                            ("final", "vocab score ms", "{:.3f}")):
        p(f"\n  {label}")
        p(f"  {'config':<8}" + "".join(f"{CNAME[c] + ' ' + x[:3]:>12}" for c in COMBO for x in pols))
        for cfg in cfgs:
            p(f"  {cfg:<8}" + "".join(f"{fmt.format(med(sel(section='W', config=cfg, w=c[0], a=c[1], policy=x), key)):>12}"
                                      for c in COMBO for x in pols))
    p("\n  ACTIVE vs default (paired median speedup of forward, wins)")
    p(f"  {'config':<8}" + "".join(f"{CNAME[c]:>16}" for c in COMBO))
    for cfg in cfgs:
        cells = []
        for c in COMBO:
            t = paired(sel(section="W", config=cfg, w=c[0], a=c[1], policy="ACTIVE"),
                       sel(section="W", config=cfg, w=c[0], a=c[1], policy="default"), "forward")
            cells.append(f"{t[0]:.3f}x {t[1]:>2}/{t[2]:<2}")
        p(f"  {cfg:<8}" + "".join(f"{x:>16}" for x in cells))
    p("\n  gain of -w 5 (w5a0 vs w0a0, Transformer blocks) under each policy: does it shrink with ACTIVE?")
    p(f"  {'config':<8}" + "".join(f"{x:>16}" for x in pols))
    for cfg in cfgs:
        cells = []
        for x in pols:
            t = paired(sel(section="W", config=cfg, w="5", a="0", policy=x),
                       sel(section="W", config=cfg, w="0", a="0", policy=x), "layers")
            cells.append(f"{t[0]:.3f}x {t[1]:>2}/{t[2]:<2}")
        p(f"  {cfg:<8}" + "".join(f"{c:>16}" for c in cells))
    p("\n  best configuration per combination and policy (tok/s)")
    for c in COMBO:
        for x in pols:
            best = max(cfgs, key=lambda cfg: med(sel(section="W", config=cfg, w=c[0], a=c[1], policy=x), "tok"))
            p(f"  {CNAME[c]} {x:<8} best {best:<7} {med(sel(section='W', config=best, w=c[0], a=c[1], policy=x), 'tok'):.0f}")
    reps = sorted({r["repeat"] for r in W})
    if len(reps) > 1:
        p("\n  reproducibility: median tok/s of w5a1 per repeat (default / ACTIVE)")
        for cfg in cfgs:
            p(f"  {cfg:<8}" + "   ".join(
                f"rep{rp}: {med(sel(section='W', config=cfg, w='5', a='1', policy='default', repeat=rp), 'tok'):.0f}"
                f" / {med(sel(section='W', config=cfg, w='5', a='1', policy='ACTIVE', repeat=rp), 'tok'):.0f}" for rp in reps))

C = sel(section="C")
if C:
    cfgs = [c for c in ORDER if any(r["config"] == c for r in C)]
    scheds = []
    for r in C:
        if r["sched"] not in scheds:
            scheds.append(r["sched"])
    scheds.sort(key=lambda s: ["static", "dynamic,16", "dynamic,4", "guided"].index(s) if s in
                ["static", "dynamic,16", "dynamic,4", "guided"] else 9)
    p("\n== C) OMP_SCHEDULE on mixed P/E configurations (w5a1, tok/s median) ==")
    p(f"  {'config':<8}{'policy':<9}" + "".join(f"{s:>12}" for s in scheds))
    for cfg in cfgs:
        for x in ("default", "ACTIVE"):
            p(f"  {cfg:<8}{x:<9}" + "".join(f"{med(sel(section='C', config=cfg, policy=x, sched=s), 'tok'):>12.0f}" for s in scheds))
    best = max(((cfg, x, s) for cfg in cfgs for x in ("default", "ACTIVE") for s in scheds),
               key=lambda t: med(sel(section="C", config=t[0], policy=t[1], sched=t[2]), "tok"))
    p(f"\n  best: {best[0]} {best[1]} {best[2]} = {med(sel(section='C', config=best[0], policy=best[1], sched=best[2]), 'tok'):.0f} tok/s")

groups = defaultdict(list)
for r in runs:
    groups[(r["section"], r["config"], r["threads"], r["cpus"], r["w"], r["a"], r["policy"], r["sched"])].append(r)
with open(os.path.join(os.path.dirname(raw), "medians.csv"), "w", newline="") as fc:
    wr = csv.writer(fc)
    wr.writerow(["section", "config", "threads", "cpus", "w", "a", "policy", "schedule", "runs",
                 "tok_s", "blocks_ms", "vocab_ms", "forward_ms"])
    for k, rows in sorted(groups.items()):
        wr.writerow(list(k) + [len(rows), f"{med(rows, 'tok'):.1f}", f"{med(rows, 'layers'):.4f}",
                               f"{med(rows, 'final'):.4f}", f"{med(rows, 'forward'):.4f}"])
shas = {r["sha"] for r in runs}
p(f"\noutputs: {len(shas)} distinct " + ("(OK, all identical)" if len(shas) == 1 else "(check!)"))
print("\n".join(out))
