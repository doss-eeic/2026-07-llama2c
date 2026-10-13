#!/usr/bin/env bash
# 出力層メモリ最適化 Step 1: ログを出さない計測モードでベースラインを取る。
# 方針: プロジェクト/output-layer-memory-optimization.md 第6, 7節
#
# 前提:
#   - run.c に STAGE_TIMING 版の計測コード（生成後にまとめて従来形式で出力）が入っている
#   - AC 電源接続、perf_event_paranoid <= 2、実行前に `sudo -v` 済み
#
# 使い方: bash bench_memopt/run_baseline.sh   （リポジトリのどこから実行してもよい）

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# ==========================================
# Configuration
# ==========================================

MODEL="./stories15M.bin"
REPEATS=5

# 推論条件（従来の bench_affinity.sh と同じ）
STEPS=256
TEMPERATURE=0
SEED=42
PROMPT="Once upon a time"

WARMUP_TOKENS=16   # 集計時に捨てる先頭トークン数（analyze_baseline.py に渡す）
COOLDOWN=3         # 各実行の間の待ち時間[秒]

CC=gcc
# make runomp と同じフラグ + 関数とループの先頭を 64 バイト境界に揃える。
# 揃えないと、コードが少し変わるだけで matmul のホットループの配置がずれ、
# P1 で約 10% 速度が変わる（機械語は同じでも、計測あり版が遅く見えた）。
CFLAGS="-Ofast -fopenmp -march=native -falign-functions=64 -falign-loops=64"

GOLDEN="bench_memopt/golden.sha256"
ANALYZER="bench_memopt/analyze_baseline.py"

BENCH_ROOT="bench_memopt/baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${BENCH_ROOT}/${TIMESTAMP}"
BIN_DIR="${OUT_DIR}/bin"

# ==========================================
# Benchmark cases
# format: name|threads|cpus|bin
#   bin: timing = STAGE_TIMING 版, plain = 計測コードなし（オーバーヘッド確認用）
# ==========================================

CASES=(
    "P1|1|0|timing"
    "P2|2|0,2|timing"
    "P4_HT|4|0-3|timing"
    "E4|4|4-7|timing"
    "X4_2_2|4|4,5,8,9|timing"

    "P1_plain|1|0|plain"
    "P4_HT_plain|4|0-3|plain"
)

# ==========================================
# Preflight checks
# ==========================================

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; WARNINGS+=("$*"); }
WARNINGS=()

[[ -f "$MODEL" ]] || die "Model not found: $MODEL"
[[ -f tokenizer.bin ]] || die "tokenizer.bin not found"

ac_online=$(cat /sys/class/power_supply/AC/online 2>/dev/null || echo "?")
[[ "$ac_online" == "1" ]] || die "AC power is not connected (AC/online=$ac_online)"

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
(( paranoid <= 2 )) || die "perf_event_paranoid=$paranoid (need <= 2): sudo sysctl kernel.perf_event_paranoid=1"

command -v perf >/dev/null || die "perf not found"
command -v turbostat >/dev/null || die "turbostat not found"
sudo -n true 2>/dev/null || die "sudo needs a password; run 'sudo -v' first"

epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo "?")
[[ "$epp" == "performance" ]] || warn "EPP is '$epp' (expected 'performance')"

grep -q "STAGE_TIMING" run.c || die "run.c has no STAGE_TIMING code"

mkdir -p "$BIN_DIR" "$OUT_DIR/env"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    warn "working tree has uncommitted changes (saved to git.diff)"
    git diff HEAD > "$OUT_DIR/git.diff"
fi

# ==========================================
# Build
# ==========================================

echo "Building..."
# 使ったバイナリを結果と一緒に残すため OUT_DIR/bin にビルドする
$CC $CFLAGS run.c -lm -o "$BIN_DIR/run"                       || die "build failed: run"
$CC $CFLAGS -DSTAGE_TIMING run.c -lm -o "$BIN_DIR/run_timing" || die "build failed: run_timing"

for b in run run_timing; do
    ldd "$BIN_DIR/$b" | grep -Eq 'libgomp|libomp' || die "OpenMP runtime not detected in $b"
done

bin_path() {
    case "$1" in
        timing) echo "$BIN_DIR/run_timing" ;;
        plain)  echo "$BIN_DIR/run" ;;
        *)      die "unknown bin: $1" ;;
    esac
}

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=threads

run_model() {  # run_model <bin> <cpus> <stdout> <stderr>
    taskset -c "$2" "$1" "$MODEL" \
        -n "$STEPS" -t "$TEMPERATURE" -s "$SEED" -i "$PROMPT" \
        > "$3" 2> "$4"
}

# ==========================================
# Golden output (master の run.c, P4_HT)
# ==========================================

if [[ ! -f "$GOLDEN" ]]; then
    echo "Creating $GOLDEN from master:run.c ..."
    git show master:run.c > "$BIN_DIR/run_master.c"
    $CC $CFLAGS "$BIN_DIR/run_master.c" -lm -o "$BIN_DIR/run_master" || die "build failed: run_master"
    OMP_NUM_THREADS=4 run_model "$BIN_DIR/run_master" 0-3 \
        "$OUT_DIR/golden.out" "$OUT_DIR/golden.log" || die "golden run failed"
    sha256sum < "$OUT_DIR/golden.out" > "$GOLDEN"
fi
GOLDEN_HASH=$(cut -d' ' -f1 "$GOLDEN")
cp "$GOLDEN" "$OUT_DIR/"

# ==========================================
# Environment
# ==========================================

cp bench_memopt/env/* "$OUT_DIR/env/" 2>/dev/null || true

{
    echo "Timestamp: $TIMESTAMP"
    echo "Model: $MODEL"
    echo "Steps: $STEPS"
    echo "Temperature: $TEMPERATURE"
    echo "Seed: $SEED"
    echo "Prompt: $PROMPT"
    echo "Repeats: $REPEATS"
    echo "Warmup tokens: $WARMUP_TOKENS"
    echo "Cooldown: ${COOLDOWN}s"
    echo
    echo "Git HEAD: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "Compiler: $($CC --version | head -1)"
    echo "CFLAGS: $CFLAGS"
    echo "Golden sha256: $GOLDEN_HASH"
    echo
    echo "AC online: $ac_online"
    echo "EPP: $epp"
    echo "Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "Power profile: $(powerprofilesctl get 2>/dev/null || echo '?')"
    echo "No turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
    echo "perf_event_paranoid: $paranoid"
    echo "Kernel: $(uname -r)"
    echo
    env | grep '^OMP_' | sort
    echo
    echo "Cases:"
    printf '  %s\n' "${CASES[@]}"
    if (( ${#WARNINGS[@]} > 0 )); then
        echo
        echo "Warnings:"
        printf '  %s\n' "${WARNINGS[@]}"
    fi
    echo
    lscpu
} > "$OUT_DIR/environment.txt"

# ==========================================
# turbostat (background, whole session)
# ==========================================

TURBO_PID_FILE="$OUT_DIR/turbostat.pid"

cleanup() {
    if [[ -f "$TURBO_PID_FILE" ]]; then
        sudo -n kill -INT "$(cat "$TURBO_PID_FILE")" 2>/dev/null
        sleep 0.5
        sudo -n chown "$(id -u):$(id -g)" "$OUT_DIR/turbostat.txt" "$TURBO_PID_FILE" 2>/dev/null
        rm -f "$TURBO_PID_FILE"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# P4_HT では 256 トークンが 0.4 秒程度で終わるので 0.1 秒間隔で取る
sudo -n sh -c "echo \$\$ > '$TURBO_PID_FILE'; exec turbostat --quiet --interval 0.1 \
    --enable Time_Of_Day_Seconds \
    --show Time_Of_Day_Seconds,CPU,Busy%,Bzy_MHz,PkgWatt,PkgTmp \
    --out '$OUT_DIR/turbostat.txt'" &
sleep 1
[[ -s "$TURBO_PID_FILE" ]] || die "turbostat failed to start"

echo "Output directory: $OUT_DIR"

# ==========================================
# Run benchmarks
# ==========================================

echo "round,case,bin,threads,cpus,start,end,rc,sha_ok" > "$OUT_DIR/runs.csv"

n_cases=${#CASES[@]}
n_fail=0

for ((r=1; r<=REPEATS; r++)); do
    # 周ごとに開始位置をずらし、熱の影響が特定の構成に偏らないようにする
    for ((k=0; k<n_cases; k++)); do
        spec=${CASES[$(( (k + r - 1) % n_cases ))]}
        IFS='|' read -r name threads cpus bin <<< "$spec"

        tag="${name}_r${r}"
        export OMP_NUM_THREADS="$threads"

        printf '[%d/%d] %-12s threads=%s cpus=%-8s bin=%s\n' \
            "$r" "$REPEATS" "$name" "$threads" "$cpus" "$bin"

        t0=$(date +%s.%N)
        perf stat -x, -e cycles,instructions,task-clock -o "$OUT_DIR/$tag.perf" -- \
            taskset -c "$cpus" "$(bin_path "$bin")" "$MODEL" \
                -n "$STEPS" -t "$TEMPERATURE" -s "$SEED" -i "$PROMPT" \
            > "$OUT_DIR/$tag.out" 2> "$OUT_DIR/$tag.log"
        rc=$?
        t1=$(date +%s.%N)

        hash=$(sha256sum < "$OUT_DIR/$tag.out" | cut -d' ' -f1)
        if [[ "$rc" == 0 && "$hash" == "$GOLDEN_HASH" ]]; then
            sha_ok=1
        else
            sha_ok=0
            n_fail=$((n_fail + 1))
            echo "  -> FAILED (rc=$rc, sha256 match=$([[ "$hash" == "$GOLDEN_HASH" ]] && echo yes || echo no))" >&2
        fi

        # cpus は "0,2" のようにカンマを含むので引用符で囲む
        echo "$r,$name,$bin,$threads,\"$cpus\",$t0,$t1,$rc,$sha_ok" >> "$OUT_DIR/runs.csv"
        sleep "$COOLDOWN"
    done
done

cleanup
trap - EXIT

echo ""
echo "Benchmark completed: $OUT_DIR"
(( n_fail == 0 )) || echo "WARNING: $n_fail run(s) failed or mismatched golden sha256 (see runs.csv)" >&2

# ==========================================
# Analyze
# ==========================================

if [[ -f "$ANALYZER" ]]; then
    python3 "$ANALYZER" "$OUT_DIR" --warmup "$WARMUP_TOKENS"
else
    echo "Analyzer not found ($ANALYZER); run it later:"
    echo "  python3 $ANALYZER $OUT_DIR --warmup $WARMUP_TOKENS"
fi
