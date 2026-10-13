#!/usr/bin/env python3
"""run_baseline.sh の結果を集計する。

方針: プロジェクト/output-layer-memory-optimization.md 第6, 7節

入力（<out_dir> 内）:
  runs.csv            各実行の case, bin, cpus, 開始/終了時刻, rc, sha_ok
  <case>_r<N>.log     run_timing のステージ計測ログ（従来形式）と achieved tok/s
  <case>_r<N>.perf    perf stat -x, の出力
  turbostat.txt       turbostat のログ（任意）
  environment.txt     Model のパスなど

出力:
  summary.tsv         1行1構成の集計
  per_token.tsv       全トークン・全ステージの生データ（ns）

使い方: python3 bench_memopt/analyze_baseline.py <out_dir> [--warmup 16] [--model PATH]
"""

import argparse
import csv
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parent.parent

STAGES = {
    "001": "embedding",
    "002": "attention_norm",
    "003": "qkv",
    "004": "rope",
    "005": "compute_attention",
    "006": "attention_output",
    "007": "ffn_norm",
    "008": "ffn_up",
    "009": "swiglu",
    "010": "ffn_down",
    "011": "final",
    "102": "layers",
    "103": "forward",
}
TOKEN_STAGES = {"001", "011", "103"}  # 1トークンに1回。それ以外は層ごと

LINE_RE = re.compile(
    r"\[token:(\d+)(?:, layer:(\d+))?\] (\d{3}):[\w-]+: "
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) seconds"
)
TOKS_RE = re.compile(r"achieved tok/s: ([0-9.]+)")

# 完了条件（Step 1）
MAX_OVERHEAD_PCT = 1.0
MAX_SPREAD_PCT = 5.0


# ==========================================
# Model
# ==========================================

def read_model_config(path):
    """llama2.c の .bin ヘッダ（int32 x 7）から1トークンで読む重みのバイト数を求める。"""
    with open(path, "rb") as f:
        dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack("7i", f.read(28))
    shared = vocab > 0
    vocab = abs(vocab)
    kv_dim = dim * n_kv_heads // n_heads
    per_layer = (
        dim * dim          # wq
        + 2 * dim * kv_dim  # wk, wv
        + dim * dim        # wo
        + 3 * dim * hidden  # w1, w2, w3
        + 2 * dim          # rms_att, rms_ffn
    )
    cls_bytes = vocab * dim * 4
    return {
        "dim": dim,
        "n_layers": n_layers,
        "vocab": vocab,
        "shared_weights": shared,
        "cls_bytes": cls_bytes,
        "forward_bytes": n_layers * per_layer * 4 + dim * 4 + cls_bytes,
    }


def model_path_from_env(out_dir):
    env = out_dir / "environment.txt"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("Model: "):
                return REPO_ROOT / line[len("Model: "):].strip()
    return REPO_ROOT / "stories15M.bin"


# ==========================================
# Parsers
# ==========================================

def parse_stage_log(path, n_layers):
    """ステージ計測ログを読む。

    戻り値: (per_token, records, tok_s)
      per_token[stage_name][pos] = その token での秒数（層ごとのステージは全層の合計）
      records = [(pos, layer, stage_name, ns), ...]（per_token.tsv 用）
    """
    per_token = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    records = []
    tok_s = None

    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            pos = int(m.group(1))
            layer = int(m.group(2)) if m.group(2) is not None else -1
            code = m.group(3)
            if code not in STAGES:
                continue
            sec = float(m.group(4))
            name = STAGES[code]
            per_token[name][pos] += sec
            counts[code][pos] += 1
            records.append((pos, layer, name, round(sec * 1e9)))
            continue
        m = TOKS_RE.search(line)
        if m:
            tok_s = float(m.group(1))

    if not per_token:
        return None, [], tok_s

    # すべてのトークンで全ステージがそろっているか確認
    positions = sorted(per_token["forward"])
    for code in STAGES:
        expected = 1 if code in TOKEN_STAGES else n_layers
        for pos in positions:
            if counts[code][pos] != expected:
                raise ValueError(
                    f"{path}: stage {code} token {pos} "
                    f"count={counts[code][pos]}, expected={expected}"
                )
    if positions != list(range(len(positions))):
        raise ValueError(f"{path}: token positions are not contiguous")

    return per_token, records, tok_s


def parse_tok_s(path):
    m = TOKS_RE.search(path.read_text()) if path.exists() else None
    return float(m.group(1)) if m else None


def parse_perf(path):
    """perf stat -x, の出力から cycles, instructions, task-clock[ms] を読む。

    ハイブリッド CPU では cpu_core/cycles/ と cpu_atom/cycles/ に分かれるので合計する。
    """
    result = {"cycles": 0.0, "cycles_core": 0.0, "cycles_atom": 0.0,
              "instructions": 0.0, "task_clock_ms": None}
    if not path.exists():
        return None
    found = False
    for row in csv.reader(path.read_text().splitlines()):
        if not row or row[0].startswith("#") or len(row) < 3:
            continue
        value, unit, event = row[0], row[1], row[2]
        try:
            v = float(value)
        except ValueError:  # <not counted>, <not supported>
            continue
        # ハイブリッド CPU では、実際に動かなかった側の PMU（例: P コアだけで動かした時の
        # cpu_atom）がわずかな有効時間から外挿された大きな値になるので、捨てる
        try:
            if len(row) > 4 and row[4] and float(row[4]) < 1.0:
                continue
        except ValueError:
            pass
        ev = event.split(":")[0]  # cycles:u -> cycles
        if "task-clock" in ev:
            result["task_clock_ms"] = v if unit in ("msec", "") else v / 1e6
            found = True
        elif "cycles" in ev:
            result["cycles"] += v
            if "cpu_core" in ev:
                result["cycles_core"] += v
            elif "cpu_atom" in ev:
                result["cycles_atom"] += v
            found = True
        elif "instructions" in ev:
            result["instructions"] += v
    return result if found else None


def parse_turbostat(path):
    """turbostat --out のログを読む。ヘッダ行は interval ごとに繰り返し出る。"""
    rows = []
    if not path.exists():
        return rows
    header = None
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if "Time_Of_Day_Seconds" in fields:
            header = fields
            continue
        # PkgWatt / PkgTmp はパッケージの先頭 CPU の行にしか出ないので、他の CPU の行は列が少ない
        if header is None or len(fields) < 2 or len(fields) > len(header):
            continue
        row = dict(zip(header, fields))
        try:
            row["Time_Of_Day_Seconds"] = float(row["Time_Of_Day_Seconds"])
        except ValueError:
            continue
        rows.append(row)
    return rows


def read_runs_csv(path):
    """runs.csv を読む。

    初期の run_baseline.sh は cpus（"0,2" など）を引用符なしで書いていたので、
    列が多すぎる行は先頭4列と末尾4列を固定とみなし、間を cpus として復元する。
    """
    lines = path.read_text().splitlines()
    header = next(csv.reader(lines[:1]))
    runs = []
    for row in csv.reader(lines[1:]):
        if not row:
            continue
        if len(row) > len(header):
            row = row[:4] + [",".join(row[4:-4])] + row[-4:]
        runs.append(dict(zip(header, row)))
    return runs


def parse_cpu_list(spec):
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(str(c) for c in range(int(a), int(b) + 1))
        elif part:
            cpus.add(part)
    return cpus


def turbostat_window(rows, start, end, cpus):
    """[start, end] に入る行から、使った CPU の Bzy_MHz と、パッケージの電力・温度を求める。"""
    win = [r for r in rows if start <= r["Time_Of_Day_Seconds"] <= end]

    def num(r, key):
        try:
            return float(r[key])
        except (KeyError, ValueError):
            return None

    # Bzy_MHz は Busy% で重み付けして平均する（ほぼ止まっていた区間の影響を小さくする）
    w_sum = mhz_sum = 0.0
    for r in win:
        if r.get("CPU") in cpus:
            busy, mhz = num(r, "Busy%"), num(r, "Bzy_MHz")
            if busy is not None and mhz is not None:
                w_sum += busy
                mhz_sum += busy * mhz
    summary = [r for r in win if r.get("CPU") == "-"]
    watts = [v for v in (num(r, "PkgWatt") for r in summary) if v is not None]
    temps = [v for v in (num(r, "PkgTmp") for r in summary) if v is not None]
    return {
        "bzy_mhz": mhz_sum / w_sum if w_sum > 0 else None,
        "pkg_watt": mean(watts) if watts else None,
        "pkg_tmp_max": max(temps) if temps else None,
        "turbo_samples": len(summary),
    }


# ==========================================
# Aggregation
# ==========================================

def med(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def spread_pct(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return (max(values) - min(values)) / median(values) * 100


def fmt(v, spec):
    """None は '-' を同じ幅で右寄せして表示する（spec は '9.4f' の形）。"""
    if v is None:
        return "-".rjust(int(spec.split(".")[0]))
    return format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--warmup", type=int, default=16, help="集計から除く先頭トークン数")
    ap.add_argument("--model", type=Path, default=None, help="モデルファイル（既定: environment.txt の Model）")
    args = ap.parse_args()

    out_dir = args.out_dir
    runs_csv = out_dir / "runs.csv"
    if not runs_csv.exists():
        sys.exit(f"runs.csv not found in {out_dir}")

    model = read_model_config(args.model or model_path_from_env(out_dir))
    n_layers = model["n_layers"]
    turbo_rows = parse_turbostat(out_dir / "turbostat.txt")

    runs = read_runs_csv(runs_csv)

    case_order = []
    case_info = {}
    case_runs = defaultdict(list)
    failed = []
    token_counts = defaultdict(set)

    per_token_out = (out_dir / "per_token.tsv").open("w", newline="")
    per_token_writer = csv.writer(per_token_out, delimiter="\t")
    per_token_writer.writerow(["case", "round", "pos", "layer", "stage", "ns"])

    for run in runs:
        name = run["case"]
        if name not in case_info:
            case_order.append(name)
            case_info[name] = {"threads": run["threads"], "cpus": run["cpus"], "bin": run["bin"]}

        tag = f"{name}_r{run['round']}"
        if run["rc"] != "0" or run["sha_ok"] != "1":
            failed.append(f"{tag} (rc={run['rc']}, sha_ok={run['sha_ok']})")
            continue

        result = {"round": run["round"]}

        log = out_dir / f"{tag}.log"
        if run["bin"] == "timing":
            per_token, records, tok_s = parse_stage_log(log, n_layers)
            if per_token is None:
                raise ValueError(f"{log}: no stage timing found (was run_timing built with -DSTAGE_TIMING?)")
            n_tokens = len(per_token["forward"])
            if n_tokens <= args.warmup:
                raise ValueError(f"{log}: only {n_tokens} tokens (<= warmup {args.warmup})")
            token_counts[name].add(n_tokens)

            for pos, layer, stage, ns in records:
                per_token_writer.writerow([name, run["round"], pos, layer, stage, ns])

            for stage, by_pos in per_token.items():
                steady = [by_pos[p] for p in range(args.warmup, n_tokens)]
                result[stage] = median(steady) * 1e3                                # ms
                result[stage + "_mean_all"] = sum(by_pos.values()) / n_tokens * 1e3  # 従来の集計方法
            result["tokens"] = n_tokens
        else:
            tok_s = parse_tok_s(log)

        result["tok_s"] = tok_s

        perf = parse_perf(out_dir / f"{tag}.perf")
        if perf and perf["task_clock_ms"]:
            result["ghz"] = perf["cycles"] / (perf["task_clock_ms"] * 1e6)
            result["ipc"] = perf["instructions"] / perf["cycles"] if perf["cycles"] else None

        if turbo_rows:
            result.update(turbostat_window(
                turbo_rows, float(run["start"]), float(run["end"]),
                parse_cpu_list(run["cpus"]),
            ))

        case_runs[name].append(result)

    per_token_out.close()

    # ---- 構成ごとに集計（各実行の代表値の中央値） ----
    stage_names = list(STAGES.values())
    summary = []
    for name in case_order:
        rs = case_runs[name]
        row = {"case": name, **case_info[name], "runs": len(rs)}
        if not rs:
            summary.append(row)
            continue
        for key in stage_names + [s + "_mean_all" for s in stage_names]:
            row[key] = med(r.get(key) for r in rs)
        for key in ("tok_s", "ghz", "ipc", "bzy_mhz", "pkg_watt"):
            row[key] = med(r.get(key) for r in rs)
        temps = [r.get("pkg_tmp_max") for r in rs if r.get("pkg_tmp_max") is not None]
        row["pkg_tmp_max"] = max(temps) if temps else None
        row["tokens"] = med(r.get("tokens") for r in rs)

        if row.get("final"):
            row["final_gbps"] = model["cls_bytes"] / (row["final"] * 1e-3) / 1e9
        if row.get("forward"):
            row["forward_gbps"] = model["forward_bytes"] / (row["forward"] * 1e-3) / 1e9

        # ばらつき: timing 版は forward の中央値、plain 版は tok/s で見る
        key = "forward" if case_info[name]["bin"] == "timing" else "tok_s"
        row["spread_pct"] = spread_pct(r.get(key) for r in rs)
        summary.append(row)

    by_case = {r["case"]: r for r in summary}

    # ---- 計測コードのオーバーヘッド（X と X_plain の tok/s） ----
    overheads = []
    for row in summary:
        name = row["case"]
        if name.endswith("_plain"):
            base = by_case.get(name[: -len("_plain")])
            if base and base.get("tok_s") and row.get("tok_s"):
                pct = (row["tok_s"] - base["tok_s"]) / row["tok_s"] * 100
                base["overhead_pct"] = pct
                overheads.append((base["case"], pct))

    # ---- summary.tsv ----
    fields = (
        ["case", "bin", "threads", "cpus", "runs", "tokens"]
        + stage_names
        + ["final_gbps", "forward_gbps", "tok_s", "overhead_pct", "ghz", "ipc",
           "bzy_mhz", "pkg_watt", "pkg_tmp_max", "spread_pct"]
        + [s + "_mean_all" for s in stage_names]
    )
    with (out_dir / "summary.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow({k: ("" if row.get(k) is None else
                            (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k]))
                        for k in fields})

    # ---- 表示 ----
    print(f"\nModel: dim={model['dim']} layers={n_layers} vocab={model['vocab']} "
          f"cls={model['cls_bytes'] / 1e6:.1f} MB forward={model['forward_bytes'] / 1e6:.1f} MB "
          f"(warmup={args.warmup} tokens excluded, median per token)")

    base_fwd = next((r["forward"] for r in summary if r.get("forward")), None)
    print("\n=== Benchmark Summary ===")
    print(f"{'Case':<13}{'Layers':>9}{'Final':>9}{'Forward':>9}{'FinGB/s':>9}{'FwdGB/s':>9}"
          f"{'tok/s':>8}{'GHz':>6}{'BzyMHz':>8}{'PkgW':>6}{'Tmp':>5}{'Spread':>8}{'Speedup':>9}")
    print(f"{'':<13}{'(ms)':>9}{'(ms)':>9}{'(ms)':>9}")
    for r in summary:
        speedup = (base_fwd / r["forward"]) if base_fwd and r.get("forward") else None
        print(
            f"{r['case']:<13}"
            f"{fmt(r.get('layers'), '9.4f')}"
            f"{fmt(r.get('final'), '9.4f')}"
            f"{fmt(r.get('forward'), '9.4f')}"
            f"{fmt(r.get('final_gbps'), '9.1f')}"
            f"{fmt(r.get('forward_gbps'), '9.1f')}"
            f"{fmt(r.get('tok_s'), '8.0f')}"
            f"{fmt(r.get('ghz'), '6.2f')}"
            f"{fmt(r.get('bzy_mhz'), '8.0f')}"
            f"{fmt(r.get('pkg_watt'), '6.1f')}"
            f"{fmt(r.get('pkg_tmp_max'), '5.0f')}"
            f"{fmt(r.get('spread_pct'), '7.1f')}%"
            f"{fmt(speedup, '8.3f')}{'x' if speedup else ' '}"
        )

    # ---- 完了条件の判定 ----
    print("\n=== Checks ===")
    ok_all = True

    def check(ok, label, detail=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"[{'OK' if ok else 'NG'}] {label}{(': ' + detail) if detail else ''}")

    check(not failed, "sha256 / exit code",
          "all runs matched golden" if not failed else "failed: " + ", ".join(failed))

    if overheads:
        for name, pct in overheads:
            check(abs(pct) < MAX_OVERHEAD_PCT, f"timing overhead {name}",
                  f"{pct:+.2f}% (limit {MAX_OVERHEAD_PCT}%)")
    else:
        check(False, "timing overhead", "no X / X_plain pair found")

    for r in summary:
        if r.get("spread_pct") is not None:
            check(r["spread_pct"] < MAX_SPREAD_PCT, f"spread {r['case']}",
                  f"{r['spread_pct']:.1f}% (limit {MAX_SPREAD_PCT}%)")

    all_counts = set().union(*token_counts.values()) if token_counts else set()
    check(len(all_counts) <= 1, "token count",
          f"{sorted(all_counts)}" + ("" if len(all_counts) <= 1 else " (check EOS termination)"))

    check(any(r.get("ghz") for r in summary), "frequency recorded (perf)")
    if not any(r.get("bzy_mhz") or r.get("pkg_watt") for r in summary):
        print("[--] no usable turbostat data (Bzy_MHz / PkgWatt / PkgTmp skipped; turbostat needs root)")

    print(f"\nResults saved to: {out_dir / 'summary.tsv'}, {out_dir / 'per_token.tsv'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
