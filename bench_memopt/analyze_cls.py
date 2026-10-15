#!/usr/bin/env python3
"""run_cls.sh の結果（results.csv）を集計する。

各条件の繰り返し（round）の中央値を取り、実験ごとに表にする。

使い方: python3 bench_memopt/analyze_cls.py <out_dir>
出力:   <out_dir>/summary.csv と画面の表
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

KEYS = ("exp", "config", "label", "wait")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def med(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def fmt(v, spec):
    if v is None:
        return "-".rjust(int(spec.split(".")[0]))
    return format(v, spec)


def load(out_dir):
    groups = defaultdict(list)
    order = []
    with (out_dir / "results.csv").open() as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in KEYS)
            if key not in groups:
                order.append(key)
            groups[key].append(row)

    summary = []
    for key in order:
        rows = groups[key]
        first = rows[0]
        s = dict(zip(KEYS, key))
        s.update({
            "rounds": len(rows),
            "kernel": first["kernel"], "page": first["page"], "rows": int(first["rows"]),
            "bytes": int(first["bytes"]), "split": int(first["split"]),
            "threads": first["threads"], "cpus": first["cpus"],
            "anon_huge_kb": first["anon_huge_kb"],
            "argmax_ok": "/".join(sorted({r["argmax_ok"] for r in rows})),
            "max_abs_err": first["max_abs_err"],
        })
        for col in ("median_us", "p10_us", "p90_us", "min_us", "gbps_median", "gbps_best", "ghz"):
            s[col] = med(num(r[col]) for r in rows)
        us = [num(r["median_us"]) for r in rows if num(r["median_us"])]
        s["spread_pct"] = (max(us) - min(us)) / median(us) * 100 if len(us) > 1 else None
        summary.append(s)
    return summary


def show_formats(rows):
    rows = [r for r in rows if r["exp"] == "formats"]
    if not rows:
        return
    configs = list(dict.fromkeys(r["config"] for r in rows))
    labels = list(dict.fromkeys(r["label"] for r in rows))
    table = {(r["config"], r["label"]): r for r in rows}
    print("\n=== formats: 出力層 1回の時間 [us] / 実効帯域 [GB/s] ===")
    print(f"{'label':<18}{'MB':>6}" + "".join(f"{c:>18}" for c in configs) + f"{'huge kB':>9}{'argmax':>8}")
    for label in labels:
        any_row = next(table[(c, label)] for c in configs if (c, label) in table)
        cells = ""
        for c in configs:
            r = table.get((c, label))
            cells += f"{fmt(r and r['median_us'], '8.1f')} /{fmt(r and r['gbps_median'], '6.1f')}  " if r else f"{'-':>18}"
        print(f"{label:<18}{any_row['bytes'] / 1e6:6.1f}{cells}{any_row['anon_huge_kb']:>9}{any_row['argmax_ok']:>8}")
    print("（GHz）" + "".join(
        f"  {c}: {fmt(med(r['ghz'] for r in rows if r['config'] == c), '4.2f')}" for c in configs))


def show_size(rows):
    rows = [r for r in rows if r["exp"] == "size"]
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        kind, mb = r["label"].rsplit("_", 1)
        by[int(mb[:-2])][kind] = r
    kinds = list(dict.fromkeys(r["label"].rsplit("_", 1)[0] for r in rows))
    print("\n=== size: データ量と実効帯域 [GB/s]（P4_HT、L3 = 12MB） ===")
    print(f"{'MB':>4}" + "".join(f"{k:>14}" for k in kinds))
    for mb in sorted(by):
        print(f"{mb:>4}" + "".join(f"{fmt(by[mb].get(k, {}).get('gbps_median'), '14.1f')}" for k in kinds))


def fit_line(points):
    """y = a + b x の最小二乗。points = [(x, y)]"""
    n = len(points)
    if n < 2:
        return None, None
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    d = n * sxx - sx * sx
    if d == 0:
        return None, None
    b = (n * sxy - sx * sy) / d
    return (sy - b * sx) / n, b


def show_split(rows):
    rows = [r for r in rows if r["exp"] == "split"]
    if not rows:
        return
    print("\n=== split: 同じ行列を K 回の並列区間に分けたときの時間 [us] ===")
    splits = sorted({r["split"] for r in rows})
    print(f"{'config':<7}{'wait':<9}{'kernel':<11}" + "".join(f"{'K=' + str(k):>9}" for k in splits)
          + f"{'us/call':>9}")
    groups = defaultdict(dict)
    for r in rows:
        kernel = r["label"].rsplit("_s", 1)[0]
        groups[(r["config"], r["wait"], kernel)][r["split"]] = r["median_us"]
    for (config, wait, kernel), by_k in groups.items():
        _, slope = fit_line([(k, v) for k, v in by_k.items() if v is not None])
        print(f"{config:<7}{wait:<9}{kernel:<11}"
              + "".join(fmt(by_k.get(k), "9.1f") for k in splits) + fmt(slope, "9.2f"))
    print("us/call: 時間を K に対して直線で当てはめた傾き（並列区間1回あたりの追加コスト）")


def show_pe(rows):
    rows = [r for r in rows if r["exp"] == "pe"]
    if not rows:
        return
    print("\n=== pe: P コア + E コア（P4X = CPU 0-3, 4, 5, 8, 9）での分担 ===")
    print(f"{'label':<20}{'wait':<9}{'us':>9}{'GB/s':>8}{'GHz':>6}{'argmax':>8}")
    for r in rows:
        print(f"{r['label']:<20}{r['wait']:<9}{fmt(r['median_us'], '9.1f')}"
              f"{fmt(r['gbps_median'], '8.1f')}{fmt(r['ghz'], '6.2f')}{r['argmax_ok']:>8}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1])
    summary = load(out_dir)

    fields = list(summary[0].keys()) if summary else []
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})

    show_formats(summary)
    show_size(summary)
    show_split(summary)
    show_pe(summary)

    bad = [s for s in summary if s["argmax_ok"] not in ("1", "-1")]
    noisy = [s for s in summary if s["spread_pct"] is not None and s["spread_pct"] > 10]
    print()
    if bad:
        print("WARNING: argmax mismatch: " + ", ".join(f"{s['config']}/{s['label']}" for s in bad))
    if noisy:
        print("NOTE: spread > 10%: " + ", ".join(
            f"{s['exp']}/{s['config']}/{s['label']}/{s['wait']} ({s['spread_pct']:.0f}%)" for s in noisy))
    print(f"Results saved to: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
