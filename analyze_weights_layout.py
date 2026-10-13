#!/usr/bin/env python3
"""Summarize bench_weights_layout.sh results (runs.tsv)."""
import csv
import sys
from collections import defaultdict
from statistics import median

NAMES = {"0": "mmap", "1": "copy", "2": "aligned", "3": "aligned+huge"}
# each step changes exactly one thing
STEPS = [("1", "0", "anon copy vs file mmap"),
         ("2", "1", "alignment"),
         ("3", "2", "huge pages"),
         ("3", "0", "total")]


def main(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # tok[threads][mode][block] = tok/s
    tok = defaultdict(lambda: defaultdict(dict))
    load = defaultdict(list)
    for r in rows:
        tok[r["threads"]][r["mode"]][r["block"]] = float(r["tok_s"])
        load[r["mode"]].append(int(r["load_ms"]))

    print("=== Validation ===")
    shas = {r["sha256"] for r in rows}
    print(f"runs={len(rows)}  distinct outputs={len(shas)}  "
          + ("OK (all modes produce identical text)" if len(shas) == 1 else "MISMATCH!"))
    for mode in sorted(load):
        offs = sorted({r["offset_mod64"] for r in rows if r["mode"] == mode})
        huge = sorted({int(r["anon_huge_kb"]) for r in rows if r["mode"] == mode})
        print(f"  {NAMES[mode]:<13} offset_mod64={','.join(offs):<4} "
              f"anon_huge_kb={huge[0]}..{huge[-1]}  load_ms(median)={median(load[mode]):.0f}")

    for threads in sorted(tok, key=int):
        modes = tok[threads]
        print(f"\n=== threads={threads} ===")
        print(f"{'mode':<13}{'n':>3}{'median tok/s':>14}{'ms/token':>10}{'min':>9}{'max':>9}{'range%':>8}")
        for mode in sorted(modes):
            v = list(modes[mode].values())
            m = median(v)
            print(f"{NAMES[mode]:<13}{len(v):>3}{m:>14.1f}{1000 / m:>10.3f}"
                  f"{min(v):>9.1f}{max(v):>9.1f}{(max(v) - min(v)) / m * 100:>8.1f}")
        print(f"{'comparison':<34}{'median ratio':>13}{'paired median':>15}{'wins':>8}")
        for new, old, label in STEPS:
            if new not in modes or old not in modes:
                continue
            blocks = sorted(set(modes[new]) & set(modes[old]))
            ratios = [modes[new][b] / modes[old][b] for b in blocks]
            wins = sum(r > 1 for r in ratios)
            ratio = median(modes[new].values()) / median(modes[old].values())
            print(f"{NAMES[new] + ' / ' + NAMES[old] + ' (' + label + ')':<34}"
                  f"{ratio:>12.3f}x{median(ratios):>14.3f}x{wins:>5}/{len(blocks)}")
    print("\nratio > 1 means the first mode is faster; 'wins' counts blocks where it was faster")


if __name__ == "__main__":
    main(sys.argv[1])
