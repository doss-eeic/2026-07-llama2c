#!/usr/bin/env bash
# 出力層の単体ベンチマーク（bench_cls）を条件を変えて一括で回す。
# 方針: プロジェクト/output-layer-memory-optimization.md 第6節 Step 2
#
# 使い方: bash bench_memopt/run_cls.sh [formats] [size] [split] [pe]
#   引数なしなら全部。環境変数で REPEATS（既定 3）、MIN_TIME（1回の計測時間[秒]、既定 0.5）、
#   COOLDOWN（各実行の間の待ち時間[秒]、既定 1）を変えられる
#
# 実験:
#   formats  データ形式・ページサイズ・カーネル × スレッド配置（P1 / P4_HT / X4_2_2 / P4X）
#   size     データ量 2〜64MB のスイープ（L3 12MB の内外）。P4_HT、乱数の重み
#   split    同じ行列を K 回の並列区間に分けて読む（呼び出しごとの固定コスト）。OMP_WAIT_POLICY も変える
#   pe       P コア + E コアでの出力層の分担（均等 / 帯域比で重み付け）

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# ==========================================
# Configuration
# ==========================================

REPEATS=${REPEATS:-3}
COOLDOWN=${COOLDOWN:-1}
MIN_TIME=${MIN_TIME:-0.5}
MODEL="./stories15M.bin"

CC=gcc
CFLAGS="-Ofast -fopenmp -march=native -falign-functions=64 -falign-loops=64"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="bench_memopt/cls/${TIMESTAMP}"
BIN="$OUT_DIR/bin/bench_cls"
RESULTS="$OUT_DIR/results.csv"

EXPS=("$@")
(( ${#EXPS[@]} > 0 )) || EXPS=(formats size split pe)

# スレッド配置: name|threads|cpus
declare -A CONFIG=(
    [P1]="1|0"
    [P4_HT]="4|0-3"
    [X4_2_2]="4|4,5,8,9"
    [P4X]="8|0-3,4,5,8,9"       # P4_HT + X4_2_2（スレッド 0-3 が P、4-7 が E）
)
# P4X の帯域比による配分（ベースラインの出力層 GB/s: P4_HT 48.5、X4_2_2 42.6 を 4 スレッドずつに分けた比）
PE_WEIGHTS="48.5,48.5,48.5,48.5,42.6,42.6,42.6,42.6"

# ==========================================
# Preflight
# ==========================================

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; WARNINGS+=("$*"); }
WARNINGS=()

[[ -f "$MODEL" ]] || die "Model not found: $MODEL"
ac_online=$(cat /sys/class/power_supply/AC/online 2>/dev/null || echo "?")
[[ "$ac_online" == "1" ]] || die "AC power is not connected (AC/online=$ac_online)"
epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo "?")
[[ "$epp" == "performance" ]] || warn "EPP is '$epp' (expected 'performance')"
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
USE_PERF=1
if (( paranoid > 2 )) || ! command -v perf >/dev/null; then
    warn "perf unavailable (perf_event_paranoid=$paranoid); ghz column will be empty"
    USE_PERF=0
fi

mkdir -p "$OUT_DIR/bin"
$CC $CFLAGS bench_memopt/bench_cls.c -lm -o "$BIN" || die "build failed"

{
    echo "Timestamp: $TIMESTAMP"
    echo "Experiments: ${EXPS[*]}"
    echo "Repeats: $REPEATS"
    echo "Min time per run: ${MIN_TIME}s"
    echo "Model: $MODEL"
    echo "Git HEAD: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "Compiler: $($CC --version | head -1)"
    echo "CFLAGS: $CFLAGS"
    echo "AC online: $ac_online"
    echo "EPP: $epp"
    echo "Power profile: $(powerprofilesctl get 2>/dev/null || echo '?')"
    echo "perf_event_paranoid: $paranoid"
    echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo "Kernel: $(uname -r)"
    if (( ${#WARNINGS[@]} > 0 )); then
        echo "Warnings:"
        printf '  %s\n' "${WARNINGS[@]}"
    fi
} > "$OUT_DIR/environment.txt"
cp -r bench_memopt/env "$OUT_DIR/" 2>/dev/null || true

echo "exp,config,label,wait,round,$("$BIN" --header),ghz" > "$RESULTS"

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=threads

# ==========================================
# Run one case
# ==========================================

PERF_TMP="$OUT_DIR/.perf.tmp"

# perf stat の出力から平均周波数（cycles ÷ task-clock）を求める。
# 実際に動かなかった側の PMU（有効時間 1% 未満）は外挿値なので捨てる
perf_ghz() {
    awk -F, '
        /^#/ || NF < 5 { next }
        $3 ~ /task-clock/ { ms = $1 }
        $3 ~ /cycles/ && $5 + 0 >= 1 && $1 ~ /^[0-9.]+$/ { cyc += $1 }
        END { if (ms > 0 && cyc > 0) printf "%.3f", cyc / (ms * 1e6) }
    ' "$1"
}

# run_case <exp> <config> <label> <wait> <round> [bench_cls args...]
run_case() {
    local exp=$1 config=$2 label=$3 wait=$4 round=$5
    shift 5
    local threads cpus
    IFS='|' read -r threads cpus <<< "${CONFIG[$config]}"

    local env_args=(OMP_NUM_THREADS="$threads")
    [[ "$wait" == "default" ]] || env_args+=(OMP_WAIT_POLICY="$wait")

    local cmd=(env "${env_args[@]}" taskset -c "$cpus" "$BIN" --model "$MODEL" --min-time "$MIN_TIME" "$@")
    local line rc ghz=""
    if (( USE_PERF )); then
        line=$(perf stat -x, -e cycles,task-clock -o "$PERF_TMP" -- "${cmd[@]}")
        rc=$?
        ghz=$(perf_ghz "$PERF_TMP")
    else
        line=$("${cmd[@]}")
        rc=$?
    fi
    if [[ $rc -ne 0 || -z "$line" ]]; then
        echo "  FAILED: $exp $config $label $*" >&2
        return
    fi
    echo "$exp,$config,$label,$wait,$round,$line,$ghz" >> "$RESULTS"
    printf '  %-8s %-7s %-22s %-8s %s\n' "$exp" "$config" "$label" "$wait" \
        "$(cut -d, -f14,18 <<< "$line" | awk -F, '{printf "%9.1f us %6.1f GB/s", $1, $2}')"
    sleep "$COOLDOWN"
}

# ==========================================
# Experiments
# ==========================================

exp_formats() {
    local r=$1
    local variants=(
        "fp32_file|--kernel fp32 --page file"
        "fp32_4k|--kernel fp32 --page 4k"
        "fp32_2m|--kernel fp32 --page 2m"
        "fp32_r4_2m|--kernel fp32_r4 --page 2m"
        "fp32_r4_pf2k_2m|--kernel fp32_r4 --page 2m --prefetch 2048"
        "fp16_2m|--kernel fp16 --page 2m"
        "int8_2m|--kernel int8 --page 2m"
        "read_2m|--kernel read --page 2m"
    )
    for config in P1 P4_HT X4_2_2 P4X; do
        for v in "${variants[@]}"; do
            IFS='|' read -r label args <<< "$v"
            # shellcheck disable=SC2086
            run_case formats "$config" "$label" default "$r" $args
        done
    done
}

exp_size() {
    local r=$1
    for mb in 2 4 6 8 10 12 14 16 20 24 32 48 64; do
        local rows=$(( mb * 1024 * 1024 / (288 * 4) ))
        run_case size P4_HT "fp32_${mb}MB" default "$r" --kernel fp32 --page 2m --src synth --rows "$rows"
        run_case size P4_HT "read_${mb}MB" default "$r" --kernel read --page 2m --src synth --rows "$rows"
        # 他の層の重みを読んだ後を模して、毎回 64MB 読んでキャッシュを追い出す
        run_case size P4_HT "fp32flush_${mb}MB" default "$r" --kernel fp32 --page 2m --src synth --rows "$rows" --flush-mb 64
    done
}

exp_split() {
    local r=$1
    for config in P1 P4_HT; do
        for wait in default active; do
            for k in 1 2 4 8 16 43; do
                run_case split "$config" "none_s$k" "$wait" "$r" --kernel none --page 2m --split "$k"
                run_case split "$config" "fp32_file_s$k" "$wait" "$r" --kernel fp32 --page file --split "$k"
            done
        done
    done
}

exp_pe() {
    local r=$1
    for wait in default active; do
        run_case pe P4X "fp32_2m_equal" "$wait" "$r" --kernel fp32 --page 2m
        run_case pe P4X "fp32_2m_weighted" "$wait" "$r" --kernel fp32 --page 2m --weights "$PE_WEIGHTS"
        run_case pe P4X "int8_2m_equal" "$wait" "$r" --kernel int8 --page 2m
        run_case pe P4X "int8_2m_weighted" "$wait" "$r" --kernel int8 --page 2m --weights "$PE_WEIGHTS"
    done
}

# モデルファイルをページキャッシュに載せておく（page=file の初回だけ遅くならないように）
cat "$MODEL" > /dev/null

echo "Output directory: $OUT_DIR"
for ((r = 1; r <= REPEATS; r++)); do
    for exp in "${EXPS[@]}"; do
        echo "[round $r/$REPEATS] $exp"
        case "$exp" in
            formats) exp_formats "$r" ;;
            size)    exp_size "$r" ;;
            split)   exp_split "$r" ;;
            pe)      exp_pe "$r" ;;
            *)       die "unknown experiment: $exp" ;;
        esac
    done
done
rm -f "$PERF_TMP"

echo ""
echo "Benchmark completed: $OUT_DIR"
python3 bench_memopt/analyze_cls.py "$OUT_DIR"
