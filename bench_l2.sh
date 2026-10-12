#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="./stories15M.bin"
REPEATS=5

STEPS=256
TEMPERATURE=0
SEED=42
PROMPT="Once upon a time"

BENCH_ROOT="benchmark_affinity_stories15M"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${BENCH_ROOT}/${TIMESTAMP}_l2"

mkdir -p "$OUT_DIR"

CASES=(
    "P1|1|0"

    "E2|2|4,5"
    "E2_B|2|8,9"
    "E2_cross|2|4,8"

    "E4|4|4-7"
    "E4_B|4|8-11"
    "E4_cross|4|4,5,8,9"

    "E8|8|4-11"
)

# OpenMP版をビルド
make runomp CC=gcc

if ! ldd ./run | grep -Eq 'libgomp|libomp'; then
    echo "ERROR: OpenMP is not enabled"
    exit 1
fi

# 環境を保存
{
    date
    lscpu
    echo
    echo "Model: $MODEL"
    echo "Steps: $STEPS"
    echo "Seed: $SEED"
    echo "Repeats: $REPEATS"
} > "$OUT_DIR/environment.txt"

for ((r=1; r<=REPEATS; r++)); do
    for spec in "${CASES[@]}"; do

        IFS='|' read -r name threads cpus <<< "$spec"

        echo "Run $r/$REPEATS: $name"

        taskset -c "$cpus" \
            env \
                OMP_NUM_THREADS="$threads" \
                OMP_DYNAMIC=FALSE \
                OMP_PROC_BIND=TRUE \
                OMP_PLACES=threads \
            ./run "$MODEL" \
                -n "$STEPS" \
                -t "$TEMPERATURE" \
                -s "$SEED" \
                -i "$PROMPT" \
            > "$OUT_DIR/${name}_r${r}.out" \
            2> "$OUT_DIR/${name}_r${r}.log"

    done
done

echo "Benchmark completed: $OUT_DIR"

uv run analyze/analyze_affinity.py --dir "$OUT_DIR"