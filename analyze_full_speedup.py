#!/usr/bin/env python3
"""Summarize full_speedup_check.sh results.

Input lines (raw.txt):
  RAW section repeat round config threads cpus w a g n profile tok layers final cand_mean cand_max sha
  VER prompt_id config w a1_same a3_same a3_argmax_mm a3_value_mm a3_missed a3g32_argmax_mm a3g32_missed a2_same a2_first_diff cand_mean cand_max
Writes a text report to stdout and medians.csv next to raw.txt.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

raw_path = sys.argv[1]
out_dir = os.path.dirname(raw_path)
runs = []   # dicts
vers = []
for line in open(raw_path):
    f = line.split()
    if not f:
        continue
    if f[0] == "RAW" and len(f) == 18:
        keys = ["section", "repeat", "round", "config", "threads", "cpus", "w", "a", "g", "n", "profile",
                "tok", "layers", "final", "cand_mean", "cand_max", "sha"]
        d = dict(zip(keys, f[1:]))
        if "NA" in (d["tok"], d["layers"], d["final"]):
            continue
        for k in ("tok", "layers", "final"):
            d[k] = float(d[k])
        d["forward"] = d["layers"] + d["final"]
        d["cand_mean"] = float(d["cand_mean"]) if d["cand_mean"] != "NA" else None
        d["cand_max"] = int(d["cand_max"]) if d["cand_max"] != "NA" else None
        runs.append(d)
    elif f[0] == "VER" and len(f) == 15:
        keys = ["pid", "config", "w", "a1_same", "a3_same", "a3_argmax_mm", "a3_value_mm", "a3_missed",
                "a3g32_argmax_mm", "a3g32_missed", "a2_same", "a2_first_diff", "cand_mean", "cand_max"]
        vers.append(dict(zip(keys, f[1:])))

CONFIG_ORDER = ["P1", "P2", "P4_HT", "E4", "X4_2_2", "E1", "E2"]
COMBO_NAME = {("0", "0"): "w0 a0 (original)", ("3", "0"): "w3 a0 (+hugepage)", ("5", "0"): "w5 a0 (+fused)",
              ("0", "1"): "w0 a1 (int8 only)", ("3", "1"): "w3 a1", ("5", "1"): "w5 a1 (all)",
              ("5", "2"): "w5 a2 (int8 approx)"}
COMBO_ORDER = [("0", "0"), ("3", "0"), ("5", "0"), ("0", "1"), ("3", "1"), ("5", "1"), ("5", "2")]


def sel(**kw):
    return [r for r in runs if all(r[k] == v for k, v in kw.items())]


def med(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return st.median(vals) if vals else float("nan")


def paired(rows_new, rows_old, key):
    """median over (repeat, round) pairs of old/new for times, new/old for tok/s; plus wins"""
    a = {(r["repeat"], r["round"]): r[key] for r in rows_old}
    b = {(r["repeat"], r["round"]): r[key] for r in rows_new}
    rs = [(b[k] / a[k]) if key == "tok" else (a[k] / b[k]) for k in a if k in b and a[k] > 0 and b[k] > 0]
    if not rs:
        return float("nan"), 0, 0
    return st.median(rs), sum(x > 1 for x in rs), len(rs)


out = []
p = out.append

# ------------------------------------------------------------------ environment line
env = os.path.join(out_dir, "environment.txt")
if os.path.exists(env):
    for l in open(env):
        if l.startswith("profile="):
            p("[" + l.strip() + "]")
            break

# ------------------------------------------------------------------ V: correctness
p("\n== V) correctness: int8 output layer vs fp32 (-t 0) ==")
if vers:
    n = len(vers)
    a1 = sum(v["a1_same"] == "1" for v in vers)
    a3 = sum(v["a3_same"] == "1" for v in vers)
    mm = sum(int(v["a3_argmax_mm"]) for v in vers)
    vm = sum(int(v["a3_value_mm"]) for v in vers)
    ms = sum(int(v["a3_missed"]) for v in vers)
    mm32 = sum(int(v["a3g32_argmax_mm"]) for v in vers)
    ms32 = sum(int(v["a3g32_missed"]) for v in vers)
    a2 = sum(v["a2_same"] == "1" for v in vers)
    p(f"runs (prompt x config x -w): {n}")
    p(f"-a 1 output identical to fp32:        {a1}/{n}")
    p(f"-a 3 output identical to fp32:        {a3}/{n}")
    p(f"-a 3 (g=144) argmax mismatches:       {mm}   re-ranked value mismatches: {vm}   true best missed: {ms}")
    p(f"-a 3 (g=32)  argmax mismatches:       {mm32}   true best missed: {ms32}")
    p(f"-a 2 (int8 approx, lossy) output identical to fp32: {a2}/{n}")
    diffs = [int(v["a2_first_diff"]) for v in vers if v["a2_same"] != "1" and v["a2_first_diff"].isdigit()]
    if diffs:
        p(f"   when different, first differing byte of the text: median {st.median(diffs):.0f}, min {min(diffs)}, max {max(diffs)}")
    cm = [float(v["cand_mean"]) for v in vers if v["cand_mean"] not in ("NA", "")]
    cx = [int(v["cand_max"]) for v in vers if v["cand_max"].isdigit()]
    if cm:
        p(f"candidates re-ranked per token (g=144): mean of means {st.mean(cm):.2f}, largest {max(cx)}")
else:
    p("(no data)")

# ------------------------------------------------------------------ M: main matrix
M = sel(section="M", profile="performance")
configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in M)]
combos = [c for c in COMBO_ORDER if any((r["w"], r["a"]) == c for r in M)]
if M:
    p("\n== M) main matrix (performance profile, -g 144 for a1/a2; median over all rounds and repeats) ==")
    for key, label, fmt in (("tok", "tok/s", "{:.0f}"), ("layers", "layers ms", "{:.3f}"),
                            ("final", "output layer ms", "{:.3f}"), ("forward", "forward ms", "{:.3f}")):
        p(f"\n  {label}")
        p(f"  {'config':<8}" + "".join(f"{COMBO_NAME[c].split(' (')[0]:>9}" for c in combos))
        for cfg in configs:
            p(f"  {cfg:<8}" + "".join(f"{fmt.format(med(sel(section='M', profile='performance', config=cfg, w=c[0], a=c[1]), key)):>9}" for c in combos))

    p("\n  speedup of each combination vs w0 a0 (forward time, paired median, wins)")
    p(f"  {'config':<8}" + "".join(f"{COMBO_NAME[c].split(' (')[0]:>16}" for c in combos[1:]))
    for cfg in configs:
        base = sel(section="M", profile="performance", config=cfg, w="0", a="0")
        cells = []
        for c in combos[1:]:
            r, wn, nn = paired(sel(section="M", profile="performance", config=cfg, w=c[0], a=c[1]), base, "forward")
            cells.append(f"{r:.3f}x {wn:>2}/{nn:<2}")
        p(f"  {cfg:<8}" + "".join(f"{x:>16}" for x in cells))

    steps = [(("3", "0"), ("0", "0"), "hugepage   (w3a0/w0a0)"), (("5", "0"), ("3", "0"), "fused      (w5a0/w3a0)"),
             (("5", "1"), ("5", "0"), "int8 exact (w5a1/w5a0)"), (("0", "1"), ("0", "0"), "int8 alone (w0a1/w0a0)"),
             (("5", "2"), ("5", "1"), "approx     (w5a2/w5a1)")]
    p("\n  what each step adds (paired median speedup; layers / output layer / forward)")
    for new, old, label in steps:
        if new not in combos or old not in combos:
            continue
        p(f"  {label}")
        for cfg in configs:
            rn = sel(section="M", profile="performance", config=cfg, w=new[0], a=new[1])
            ro = sel(section="M", profile="performance", config=cfg, w=old[0], a=old[1])
            cells = [paired(rn, ro, k) for k in ("layers", "final", "forward")]
            p(f"    {cfg:<8}" + "".join(f"{c[0]:>8.3f}x {c[1]:>2}/{c[2]:<2}" for c in cells))

    p("\n  best configuration per combination (tok/s) and share of the output layer in forward")
    for c in combos:
        best = max(configs, key=lambda cfg: med(sel(section="M", profile="performance", config=cfg, w=c[0], a=c[1]), "tok"))
        rows = sel(section="M", profile="performance", config=best, w=c[0], a=c[1])
        share = med(rows, "final") / med(rows, "forward") * 100
        p(f"  {COMBO_NAME[c]:<22} best {best:<7} {med(rows, 'tok'):>5.0f} tok/s   output layer {share:4.1f}% of forward")
    b0 = max(med(sel(section="M", profile="performance", config=cfg, w="0", a="0"), "tok") for cfg in configs)
    if ("5", "1") in combos:
        b1 = max(med(sel(section="M", profile="performance", config=cfg, w="5", a="1"), "tok") for cfg in configs)
        p(f"\n  overall: best exact (w5 a1) {b1:.0f} tok/s vs best original (w0 a0) {b0:.0f} tok/s = x{b1 / b0:.2f}")

    reps = sorted({r["repeat"] for r in M})
    if len(reps) > 1:
        p("\n  reproducibility: median tok/s per repeat (w0 a0 / w5 a1)")
        for cfg in configs:
            cells = []
            for rp in reps:
                x0 = med(sel(section="M", profile="performance", config=cfg, w="0", a="0", repeat=rp), "tok")
                x1 = med(sel(section="M", profile="performance", config=cfg, w="5", a="1", repeat=rp), "tok")
                cells.append(f"rep{rp}: {x0:.0f} / {x1:.0f}")
            p(f"  {cfg:<8}" + "   ".join(cells))

    cand = [r for r in M if r["a"] == "1" and r["cand_mean"] is not None]
    if cand:
        p(f"\n  candidates re-ranked per token in M (a1): mean {st.mean(r['cand_mean'] for r in cand):.2f}, "
          f"largest {max(r['cand_max'] for r in cand)}")

# ------------------------------------------------------------------ G: group size sweep
G = sel(section="G")
if G:
    p("\n== G) group size sweep (w5 a1, performance) ==")
    gs = sorted({r["g"] for r in G if r["a"] == "1"}, key=int)
    gcfg = [c for c in CONFIG_ORDER if any(r["config"] == c for r in G)]
    p(f"  {'g':>5}  {'int8 table':>11}" + "".join(f"{c + ' out ms':>15}{'cand':>7}" for c in gcfg))
    for g in gs:
        mb = 32000 * 288 / 1e6 + 32000 * (288 // int(g)) * 4 / 1e6
        cells = ""
        for c in gcfg:
            rows = sel(section="G", config=c, g=g, a="1")  # fp32 reference runs are also recorded with g=144
            cells += f"{med(rows, 'final'):>15.3f}{med(rows, 'cand_mean'):>7.2f}"
        p(f"  {g:>5}  {mb:>9.2f}MB" + cells)
    fp = sel(section="G", a="0")
    if fp:
        p("  fp32 reference (a0):" + "".join(f"  {c} {med(sel(section='G', a='0', config=c), 'final'):.3f} ms" for c in gcfg))

# ------------------------------------------------------------------ S: sequence length
S = sel(section="S")
if S:
    p("\n== S) generated length (P4_HT, performance) ==")
    p(f"  {'-n':>5}{'w0 a0 tok/s':>14}{'w5 a1 tok/s':>14}{'ratio':>8}")
    for n in sorted({r["n"] for r in S}, key=int):
        x0 = med(sel(section="S", n=n, w="0", a="0"), "tok")
        x1 = med(sel(section="S", n=n, w="5", a="1"), "tok")
        p(f"  {n:>5}{x0:>14.0f}{x1:>14.0f}{x1 / x0:>7.2f}x")

# ------------------------------------------------------------------ B: power profiles
B = sel(section="B")
if B:
    p("\n== B) power profiles (tok/s, median) ==")
    profs = [x for x in ("power-saver", "balanced", "performance") if any(r["profile"] == x for r in B)]
    bcfg = [c for c in CONFIG_ORDER if any(r["config"] == c for r in B)]
    p(f"  {'config / combo':<22}" + "".join(f"{x:>14}" for x in profs))
    for c in bcfg:
        for w, a in (("0", "0"), ("5", "1")):
            p(f"  {c + ' w' + w + ' a' + a:<22}" + "".join(f"{med(sel(section='B', profile=x, config=c, w=w, a=a), 'tok'):>14.0f}" for x in profs))

# ------------------------------------------------------------------ CSV of medians
groups = defaultdict(list)
for r in runs:
    groups[(r["section"], r["config"], r["threads"], r["cpus"], r["w"], r["a"], r["g"], r["n"], r["profile"])].append(r)
with open(os.path.join(out_dir, "medians.csv"), "w", newline="") as fcsv:
    wr = csv.writer(fcsv)
    wr.writerow(["section", "config", "threads", "cpus", "w", "a", "g", "n", "profile", "runs",
                 "tok_s", "layers_ms", "output_ms", "forward_ms", "cand_mean", "cand_max"])
    for k, rows in sorted(groups.items()):
        cm = [r["cand_mean"] for r in rows if r["cand_mean"] is not None]
        cx = [r["cand_max"] for r in rows if r["cand_max"] is not None]
        wr.writerow(list(k) + [len(rows), f"{med(rows, 'tok'):.1f}", f"{med(rows, 'layers'):.4f}",
                               f"{med(rows, 'final'):.4f}", f"{med(rows, 'forward'):.4f}",
                               f"{st.mean(cm):.2f}" if cm else "", max(cx) if cx else ""])
shas = {r["sha"] for r in runs if r["a"] in ("0", "1") and r["n"] == "256"}
p(f"\noutputs of exact runs (-a 0/1, -n 256): {len(shas)} distinct " + ("(OK, all identical)" if len(shas) == 1 else "(check!)"))
print("\n".join(out))
