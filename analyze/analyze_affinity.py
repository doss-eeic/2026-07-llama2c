#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict
from statistics import median


# ============================================================
# Configuration
# ============================================================

# analyze/ の親ディレクトリをプロジェクトルートとする
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCH_ROOT = PROJECT_ROOT / "benchmark_affinity_stories15M"

EXPECTED_RUNS = 5

# ケース名: (スレッド数, CPU affinity)
CASES = {
    "auto_1": (1, "-"),
    "auto_2": (2, "-"),
    "auto_4": (4, "-"),
    "auto_8": (8, "-"),
    "P1": (1, "0"),
    "P2": (2, "0,2"),
    "P4_HT": (4, "0-3"),
    "E2": (2, "4,5"),
    "E4": (4, "4-7"),
    "E8": (8, "4-11"),
    "P2E2": (4, "0,2,4,5"),
    #L2キャッシュ用
    "E2_B": (2, "8,9"),
    "E2_cross": (2, "4,8"),
    "E4_B": (4, "8-11"),
    "E4_cross": (4, "4,5,8,9"),
}

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

# Projectionに該当するステージ
PROJECTION_STAGES = [
    "003",  # QKV
    "006",  # Attention Output
    "008",  # FFN Up
    "010",  # FFN Down
    "011",  # Final Projection
]

# ログの形式:
# 003:qkv-projection:: 0.000100000 seconds
# 102:layer-compute-total: 0.000500000 seconds
TIME_PATTERN = re.compile(
    r"\b(?P<stage>\d{3}):[\w-]+:{1,2}\s*"
    r"(?P<time>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"\s+seconds"
)

# 例: P2_r1.log / auto_4_r5.log
FILE_PATTERN = re.compile(
    r"^(?P<case>.+)_r(?P<run>\d+)\.log$"
)


# ============================================================
# Directory
# ============================================================

def find_latest_benchmark():
    """最新のベンチマークディレクトリを取得"""

    if not BENCH_ROOT.exists():
        raise FileNotFoundError(
            f"Benchmark directory not found: {BENCH_ROOT}"
        )

    candidates = [
        p for p in BENCH_ROOT.iterdir()
        if p.is_dir()
        and any(p.glob("*_r*.log"))
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No benchmark logs found in {BENCH_ROOT}"
        )

    # ディレクトリ名 YYYYMMDD_HHMMSS を想定
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_benchmark_dir(args):
    """引数で指定されたディレクトリ、または最新のものを選択"""

    if args.dir is None:
        return find_latest_benchmark()

    path = Path(args.dir)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(path)

    return path


# ============================================================
# Log parsing
# ============================================================

def parse_log(path):
    """ステージごとの計測時間を秒単位で取得"""

    data = defaultdict(list)

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = TIME_PATTERN.search(line)

            if match is None:
                continue

            stage = match.group("stage")

            if stage not in STAGES:
                continue

            elapsed = float(match.group("time"))
            data[stage].append(elapsed)

    return data


def validate_log(data, path):
    """全ステージの記録数が正しいことを確認"""

    tokens = len(data["103"])
    layers = len(data["102"])

    if tokens == 0 or layers == 0:
        raise ValueError(
            f"{path}: No forward/layer measurements found"
        )

    if layers % tokens != 0:
        raise ValueError(
            f"{path}: Invalid layer count "
            f"(tokens={tokens}, layers={layers})"
        )

    n_layers = layers // tokens

    for stage in STAGES:

        if stage in ("001", "011", "103"):
            expected = tokens
        else:
            expected = layers

        actual = len(data[stage])

        if actual != expected:
            raise ValueError(
                f"{path}: Stage {stage} count mismatch "
                f"(expected={expected}, actual={actual})"
            )

    return tokens, layers, n_layers


# ============================================================
# Per-run analysis
# ============================================================

def analyze_run(path, case, run_number):
    """1回の実行結果を解析"""

    data = parse_log(path)

    tokens, layers, n_layers = validate_log(data, path)

    threads, cpus = CASES.get(case, (None, "-"))

    result = {
        "case": case,
        "run": run_number,
        "threads": threads,
        "cpus": cpus,
        "tokens": tokens,
        "layers": layers,
        "n_layers": n_layers,
    }

    # 各ステージの処理時間
    for stage_id, stage_name in STAGES.items():

        values = data[stage_id]

        # 1回の呼び出し当たり [µs]
        mean_call_us = sum(values) / len(values) * 1e6

        # 1トークン当たり [ms]
        time_per_token_ms = sum(values) / tokens * 1e3

        result[f"us_{stage_name}"] = mean_call_us
        result[f"ms_{stage_name}"] = time_per_token_ms

    forward_ms = result["ms_forward"]

    # Projection全体
    projection_ms = sum(
        result[f"ms_{STAGES[stage]}"]
        for stage in PROJECTION_STAGES
    )

    result["projection_ms"] = projection_ms

    # Forwardに占める割合
    result["projection_pct"] = (
        projection_ms / forward_ms * 100
    )

    result["final_pct"] = (
        result["ms_final"] / forward_ms * 100
    )

    result["layers_pct"] = (
        result["ms_layers"] / forward_ms * 100
    )

    return result


# ============================================================
# Collect results
# ============================================================

def collect_results(benchmark_dir):
    """指定ディレクトリの全ログを収集"""

    grouped = defaultdict(list)

    for path in benchmark_dir.glob("*.log"):

        match = FILE_PATTERN.match(path.name)

        if match is None:
            continue

        case = match.group("case")
        run_number = int(match.group("run"))

        grouped[case].append((run_number, path))

    if not grouped:
        raise ValueError(
            f"No valid log files found in {benchmark_dir}"
        )

    results = {}

    # 定義済みケース順 + その他のケース
    ordered_cases = [
        case for case in CASES
        if case in grouped
    ]

    ordered_cases += sorted(
        case for case in grouped
        if case not in CASES
    )

    for case in ordered_cases:

        runs = []

        files = sorted(
            grouped[case],
            key=lambda x: x[0]
        )

        for run_number, path in files:

            result = analyze_run(
                path,
                case,
                run_number,
            )

            runs.append(result)

            print(
                f"{case:<10} "
                f"run={run_number:<2} "
                f"tokens={result['tokens']:<4} "
                f"layers={result['layers']:<5} "
                f"forward={result['ms_forward']:.4f} ms"
            )

        results[case] = runs

    return results


# ============================================================
# Statistics
# ============================================================

def med(runs, key):
    """各実行の値の中央値"""

    return median(
        run[key] for run in runs
    )


def summarize_cases(results):
    """CPU構成ごとの代表値を集計"""

    summaries = []

    for case, runs in results.items():

        first = runs[0]

        summary = {
            "case": case,
            "threads": first["threads"],
            "cpus": first["cpus"],
            "runs": len(runs),
            "tokens_min": min(r["tokens"] for r in runs),
            "tokens_max": max(r["tokens"] for r in runs),
            "forward_ms": med(runs, "ms_forward"),
            "forward_min_ms": min(
                r["ms_forward"] for r in runs
            ),
            "forward_max_ms": max(
                r["ms_forward"] for r in runs
            ),
            "layers_ms": med(runs, "ms_layers"),
            "final_ms": med(runs, "ms_final"),
            "projection_ms": med(runs, "projection_ms"),
            "projection_pct": med(runs, "projection_pct"),
            "final_pct": med(runs, "final_pct"),
            "layers_pct": med(runs, "layers_pct"),
        }

        summaries.append(summary)

    # auto_1を基準とする
    baseline = next(
        (
            s["forward_ms"]
            for s in summaries
            if s["case"] == "auto_1"
        ),
        summaries[0]["forward_ms"],
    )

    for summary in summaries:
        summary["speedup"] = (
            baseline / summary["forward_ms"]
        )

    return summaries


def summarize_stages(results):
    """ステージごとの性能を集計"""

    summaries = []

    for case, runs in results.items():

        for stage_id, stage_name in STAGES.items():

            us_key = f"us_{stage_name}"
            ms_key = f"ms_{stage_name}"

            per_call_us = med(runs, us_key)
            per_token_ms = med(runs, ms_key)

            # 割合は各runで計算してから中央値
            percentages = [
                run[ms_key] / run["ms_forward"] * 100
                for run in runs
            ]

            summary = {
                "case": case,
                "stage_id": stage_id,
                "stage": stage_name,
                "us_per_call": per_call_us,
                "ms_per_token": per_token_ms,
                "forward_pct": median(percentages),
            }

            summaries.append(summary)

    return summaries


# ============================================================
# CSV output
# ============================================================

def write_csv(path, rows):
    """辞書の配列をCSVとして保存"""

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Report
# ============================================================

def create_report(
    benchmark_dir,
    results,
    cases,
    stages,
):
    """分析レポートをテキストで作成"""

    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 85)
    add("LLAMA2.C CPU AFFINITY BENCHMARK")
    add("=" * 85)

    add(f"Directory: {benchmark_dir}")
    add()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    add("=== Measurement validation ===")
    add()

    all_token_counts = set()
    all_layer_counts = set()

    for case, runs in results.items():

        token_counts = [r["tokens"] for r in runs]
        layer_counts = [r["layers"] for r in runs]

        all_token_counts.update(token_counts)
        all_layer_counts.update(layer_counts)

        add(
            f"{case:<10} "
            f"runs={len(runs):<2} "
            f"tokens={min(token_counts)}-{max(token_counts)} "
            f"layers={min(layer_counts)}-{max(layer_counts)}"
        )

        if len(runs) != EXPECTED_RUNS:
            add(
                f"  WARNING: Expected {EXPECTED_RUNS} runs"
            )

    if len(all_token_counts) != 1:
        add()
        add("WARNING: Token counts differ between runs!")
        add("Sequence length may affect attention timing.")

    if len(all_layer_counts) != 1:
        add()
        add("WARNING: Layer counts differ between runs!")

    add()

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    add("=== Overall performance ===")
    add()

    add(
        f"{'Case':<12}"
        f"{'Threads':>8}"
        f"{'Layers(ms)':>13}"
        f"{'Final(ms)':>12}"
        f"{'Forward(ms)':>14}"
        f"{'Speedup':>11}"
    )

    for s in cases:

        add(
            f"{s['case']:<12}"
            f"{str(s['threads']):>8}"
            f"{s['layers_ms']:>13.4f}"
            f"{s['final_ms']:>12.4f}"
            f"{s['forward_ms']:>14.4f}"
            f"{s['speedup']:>10.3f}x"
        )

    add()

    # --------------------------------------------------------
    # Fastest configuration
    # --------------------------------------------------------

    fastest = min(
        cases,
        key=lambda x: x["forward_ms"],
    )

    fastest_name = fastest["case"]

    add("=== Fastest configuration ===")
    add()

    add(f"Case: {fastest_name}")
    add(f"Threads: {fastest['threads']}")
    add(f"CPU affinity: {fastest['cpus']}")
    add(f"Forward: {fastest['forward_ms']:.4f} ms/token")
    add(f"Speedup: {fastest['speedup']:.3f}x")
    add()

    # --------------------------------------------------------
    # Run-to-run variation
    # --------------------------------------------------------

    add("=== Run-to-run variation ===")
    add()

    add(
        f"{'Case':<12}"
        f"{'Min(ms)':>12}"
        f"{'Median(ms)':>14}"
        f"{'Max(ms)':>12}"
        f"{'Range(%)':>12}"
    )

    for s in cases:

        low = s["forward_min_ms"]
        high = s["forward_max_ms"]
        center = s["forward_ms"]

        variation = (high - low) / center * 100

        add(
            f"{s['case']:<12}"
            f"{low:>12.4f}"
            f"{center:>14.4f}"
            f"{high:>12.4f}"
            f"{variation:>12.2f}"
        )

    add()

    # --------------------------------------------------------
    # Projection ratio
    # --------------------------------------------------------

    add("=== Projection bottleneck ===")
    add()

    add(
        f"{'Case':<12}"
        f"{'Projection(ms)':>16}"
        f"{'Projection(%)':>16}"
        f"{'Final(%)':>12}"
        f"{'Layers(%)':>12}"
    )

    for s in cases:

        add(
            f"{s['case']:<12}"
            f"{s['projection_ms']:>16.4f}"
            f"{s['projection_pct']:>16.2f}"
            f"{s['final_pct']:>12.2f}"
            f"{s['layers_pct']:>12.2f}"
        )

    add()

    # --------------------------------------------------------
    # Detailed stages for fastest case
    # --------------------------------------------------------

    add(f"=== Stage breakdown: {fastest_name} ===")
    add()

    add(
        f"{'Stage':<23}"
        f"{'Per call (us)':>16}"
        f"{'Per token (ms)':>18}"
        f"{'Forward (%)':>15}"
    )

    fastest_stages = [
        s for s in stages
        if s["case"] == fastest_name
    ]

    for s in fastest_stages:

        add(
            f"{s['stage']:<23}"
            f"{s['us_per_call']:>16.3f}"
            f"{s['ms_per_token']:>18.4f}"
            f"{s['forward_pct']:>15.2f}"
        )

    add()

    # --------------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------------

    add("=== CPU affinity comparisons ===")
    add()

    case_map = {
        s["case"]: s
        for s in cases
    }

    comparisons = [
        ("P2", "E2"),
        ("P2", "E4"),
        ("P2", "E8"),
        ("P2", "P4_HT"),
        ("P2", "P2E2"),
        ("E2", "E4"),
        ("E4", "E8"),
        ("auto_2", "P2"),
        ("auto_4", "P2E2"),
        #L2キャッシュ用
        ("E2", "E2_B"),
        ("E2", "E2_cross"),

        ("E4", "E4_B"),
        ("E4", "E4_cross"),
        ("E4_B", "E4_cross"),

        ("E2", "E4"),
        ("E4", "E8"),
        ("E4_cross", "E8"),
    ]

    for a, b in comparisons:

        if a not in case_map or b not in case_map:
            continue

        ta = case_map[a]["forward_ms"]
        tb = case_map[b]["forward_ms"]

        speedup = ta / tb

        add(
            f"{a:>8} -> {b:<8} "
            f"speedup={speedup:.3f}x "
            f"({ta:.4f} -> {tb:.4f} ms)"
        )

    add()

    # --------------------------------------------------------
    # Recommended interpretation
    # --------------------------------------------------------

    add("=== Interpretation guide ===")
    add()

    add("P2 vs E2:")
    add("  Compare P-core and E-core performance at 2 threads.")

    add("E2 vs E4 vs E8:")
    add("  Evaluate scaling using homogeneous E-cores.")

    add("P2 vs P4_HT:")
    add("  Evaluate Hyper-Threading on the same physical P-cores.")

    add("P2 vs P2E2:")
    add("  Evaluate mixed P/E-core scheduling.")

    add("auto_2 vs P2:")
    add("  Evaluate explicit CPU affinity.")

    add()
    add("NOTE:")
    add("  These measurements exclude most logging overhead.")
    add("  Slowdowns alone do not establish DRAM bandwidth saturation.")
    add("  CPU frequency, contention and OpenMP overhead also matter.")
    add("  Medians are calculated independently per metric.")
    add()

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze llama2.c CPU affinity benchmark logs"
        )
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=(
            "Benchmark directory. "
            "Default: latest directory in "
            "benchmark_affinity_stories15M"
        ),
    )

    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args)

    print(f"Benchmark directory: {benchmark_dir}")
    print()

    # ログの読み込み
    results = collect_results(benchmark_dir)

    # 集計
    cases = summarize_cases(results)
    stages = summarize_stages(results)

    # 全試行の結果をフラット化
    runs = [
        run
        for case_runs in results.values()
        for run in case_runs
    ]

    # 出力先
    output_dir = benchmark_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    # CSV出力
    write_csv(output_dir / "cases.csv", cases)
    write_csv(output_dir / "stages.csv", stages)
    write_csv(output_dir / "runs.csv", runs)

    # レポート生成
    report = create_report(
        benchmark_dir,
        results,
        cases,
        stages,
    )

    # 保存
    report_path = output_dir / "report.txt"
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ターミナルに表示
    print()
    print(report)

    print("=== Saved files ===")
    print(output_dir / "cases.csv")
    print(output_dir / "stages.csv")
    print(output_dir / "runs.csv")
    print(report_path)


if __name__ == "__main__":
    main()