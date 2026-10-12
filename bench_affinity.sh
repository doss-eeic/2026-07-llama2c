#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Configuration
# ==========================================

MODEL="./stories15M.bin"
REPEATS=5

# 推論条件
STEPS=256
TEMPERATURE=0
SEED=42
PROMPT="Once upon a time"

# 既存の benchmark/ とは別に保存
BENCH_ROOT="benchmark_affinity_stories15M"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${BENCH_ROOT}/${TIMESTAMP}"

mkdir -p "$OUT_DIR"

# ==========================================
# Benchmark cases
# format: name|threads|cpus
# '-' means no CPU affinity restriction
# ==========================================

CASES=(
    "auto_1|1|-"
    "auto_2|2|-"
    "auto_4|4|-"
    "auto_8|8|-"

    "P1|1|0"
    "P2|2|0,2"
    "P4_HT|4|0-3"

    "E2|2|4,5"
    "E4|4|4-7"
    "E8|8|4-11"

    "P2E2|4|0,2,4,5"
)

# ==========================================
# Check environment
# ==========================================

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL"
    exit 1
fi

echo "Building OpenMP version..."
make runomp CC=gcc

if ! ldd ./run | grep -Eq 'libgomp|libomp'; then
    echo "OpenMP runtime not detected!"
    exit 1
fi

# 実験条件を保存
{
    echo "Model: $MODEL"
    echo "Steps: $STEPS"
    echo "Temperature: $TEMPERATURE"
    echo "Seed: $SEED"
    echo "Prompt: $PROMPT"
    echo "Repeats: $REPEATS"
    echo
    lscpu
} > "$OUT_DIR/environment.txt"

echo "Output directory: $OUT_DIR"

# ==========================================
# Run benchmarks
# ==========================================

# 各条件を交互に測定することで
# 時間経過による環境変動の偏りを軽減する

for ((r=1; r<=REPEATS; r++)); do

    for spec in "${CASES[@]}"; do

        IFS='|' read -r name threads cpus <<< "$spec"

        echo "=================================="
        echo "Run: $r / $REPEATS"
        echo "Case: $name"
        echo "Threads: $threads"
        echo "CPUs: $cpus"
        echo "=================================="

        STDOUT="$OUT_DIR/${name}_r${r}.out"
        STDERR="$OUT_DIR/${name}_r${r}.log"

        # OpenMP configuration
        export OMP_NUM_THREADS="$threads"
        export OMP_DYNAMIC=FALSE
        export OMP_PROC_BIND=TRUE
        export OMP_PLACES=threads

        if [[ "$cpus" == "-" ]]; then

            ./run "$MODEL" \
                -n "$STEPS" \
                -t "$TEMPERATURE" \
                -s "$SEED" \
                -i "$PROMPT" \
                > "$STDOUT" 2> "$STDERR"

        else

            taskset -c "$cpus" \
                ./run "$MODEL" \
                -n "$STEPS" \
                -t "$TEMPERATURE" \
                -s "$SEED" \
                -i "$PROMPT" \
                > "$STDOUT" 2> "$STDERR"

        fi

    done
done

# ==========================================
# Analyze results
# ==========================================

python3 - "$OUT_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import median

directory = Path(sys.argv[1])

cases = [
    ("auto_1", 1, "-"),
    ("auto_2", 2, "-"),
    ("auto_4", 4, "-"),
    ("auto_8", 8, "-"),
    ("P1", 1, "0"),
    ("P2", 2, "0,2"),
    ("P4_HT", 4, "0-3"),
    ("E2", 2, "4,5"),
    ("E4", 4, "4-7"),
    ("E8", 8, "4-11"),
    ("P2E2", 4, "0,2,4,5"),
]

stages = {
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

pattern = re.compile(
    r"\b(\d{3}):[\w-]+:{1,2}\s*"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"\s+seconds"
)

results = []
all_token_counts = set()

for name, threads, cpus in cases:
    run_results = []

    for path in sorted(directory.glob(f"{name}_r*.log")):
        data = defaultdict(list)

        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if match and match.group(1) in stages:
                data[match.group(1)].append(
                    float(match.group(2))
                )

        tokens = len(data["103"])
        layers = len(data["102"])

        if tokens == 0 or layers == 0:
            raise ValueError(f"Missing timing data: {path}")

        # すべてのステージが記録されているか確認
        for stage_id in stages:
            expected = (
                tokens if stage_id in ("001", "011", "103")
                else layers
            )
            if len(data[stage_id]) != expected:
                raise ValueError(
                    f"{path}: stage {stage_id} "
                    f"count={len(data[stage_id])}, "
                    f"expected={expected}"
                )

        all_token_counts.add(tokens)

        # 全ステージを1トークン当たりの時間[ms]に換算
        values = {
            stages[k]: sum(v) / tokens * 1000
            for k, v in data.items()
        }

        run_results.append(values)

        print(
            f"{name:10s} {path.name:20s} "
            f"tokens={tokens:3d} "
            f"layers={layers:4d} "
            f"forward={values['forward']:.4f} ms"
        )

    if not run_results:
        raise ValueError(f"No logs found for {name}")

    # 各実行の平均を求め、その中央値を採用
    result = {
        "case": name,
        "threads": threads,
        "cpus": cpus,
    }

    for stage in stages.values():
        result[stage] = median(
            run[stage] for run in run_results
        )

    results.append(result)

# TSV出力
output = directory / "summary.tsv"

with output.open("w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case", "threads", "cpus", *stages.values()],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(results)

# 表示
baseline = results[0]["forward"]

print("\n=== Benchmark Summary ===")
print(
    f"{'Case':<12}"
    f"{'Layer(ms)':>12}"
    f"{'Final(ms)':>12}"
    f"{'Forward(ms)':>14}"
    f"{'Speedup':>10}"
)

for r in results:
    print(
        f"{r['case']:<12}"
        f"{r['layers']:>12.4f}"
        f"{r['final']:>12.4f}"
        f"{r['forward']:>14.4f}"
        f"{baseline/r['forward']:>10.3f}x"
    )

if len(all_token_counts) != 1:
    print(
        "\nWARNING: Token counts differ across runs! "
        "Check EOS termination."
    )

print(f"\nResults saved to: {output}")
PY

echo ""
echo "Benchmark completed."
echo "Results: $OUT_DIR"