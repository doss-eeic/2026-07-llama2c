#!/usr/bin/env bash
# Compare how the weights are placed in memory (run -w option):
#   0=mmap (original), 1=copy (anon memory, same 28 byte misalignment),
#   2=aligned (2MB aligned start, 4KB pages), 3=aligned+hugepage (MADV_HUGEPAGE)
# 1 vs 0: file mmap vs anonymous copy, 2 vs 1: alignment, 3 vs 2: huge pages.
# All modes must produce byte-identical output; this is checked for every run.
set -Eeuo pipefail
export LC_ALL=C

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
MODEL="${MODEL:-./stories15M.bin}"
# the team branches keep the model in build/
[[ -f "$MODEL" || ! -f ./build/stories15M.bin ]] || MODEL=./build/stories15M.bin
RUN_BIN="${RUN_BIN:-./run}"
OUT_ROOT="${OUT_ROOT:-./benchmark_weights_layout}"
REPEATS="${REPEATS:-10}"
WARMUPS="${WARMUPS:-1}"
STEPS="${STEPS:-256}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Once upon a time}"
ORDER_SEED="${ORDER_SEED:-20261013}"
MODES="${MODES:-0 1 2 3}"
# never leave OMP_NUM_THREADS unset: under WSL all 14 threads is ~100x slower
THREADS="${THREADS:-1 4 8}"
REBUILD="${REBUILD:-1}"

if [[ "$REBUILD" == 1 ]]; then
  echo 'Rebuilding OpenMP executable: make runomp CC=gcc'
  make runomp CC=gcc
fi
[[ -x "$RUN_BIN" ]] || { echo "ERROR: executable not found: $RUN_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }

# Output folder is intentionally newly created: never overwrite prior measurements.
mkdir -p "$OUT_ROOT"
OUT_DIR="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/stdout"

cat > "$OUT_DIR/environment.txt" <<META
start_local=$(date -Is)
hostname=$(hostname)
run_bin=$RUN_BIN
model=$MODEL
repeats=$REPEATS
warmups=$WARMUPS
steps=$STEPS
seed=$SEED
prompt=$PROMPT
order_seed=$ORDER_SEED
modes=$MODES
threads=$THREADS
uname=$(uname -a)
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo n/a)
META
lscpu > "$OUT_DIR/lscpu.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  { git rev-parse HEAD; git status --short; git diff; } > "$OUT_DIR/git_revision_and_status.txt"
fi

# A different reproducible random order of all cases per block, so that drift in clock,
# temperature and background load does not favor one mode.
random_cases() {
  python3 - "$ORDER_SEED" "$1" "$MODES" "$THREADS" <<'PY'
import random, sys
cases = [(m, t) for m in sys.argv[3].split() for t in sys.argv[4].split()]
random.Random(int(sys.argv[1]) + int(sys.argv[2])).shuffle(cases)
for m, t in cases:
    print(m, t)
PY
}

run_case() {
  local mode="$1" threads="$2" stdout="$3" stderr="$4"
  OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE \
    "$RUN_BIN" "$MODEL" -n "$STEPS" -t 0 -s "$SEED" -i "$PROMPT" -w "$mode" \
    >"$stdout" 2>"$stderr"
}

printf 'Output directory: %s\n' "$OUT_DIR"
for ((round=1; round<=WARMUPS; round++)); do
  echo "Warm-up block $round/$WARMUPS"
  while read -r mode threads; do
    run_case "$mode" "$threads" /dev/null /dev/null
  done < <(random_cases "$(( -round ))")
done

printf 'block\tmode\tthreads\ttok_s\tload_ms\toffset_mod64\tanon_huge_kb\tsha256\n' > "$OUT_DIR/runs.tsv"
for ((round=1; round<=REPEATS; round++)); do
  echo "Measured block $round/$REPEATS"
  while read -r mode threads; do
    stem="w${mode}_t${threads}_r$(printf '%02d' "$round")"
    run_case "$mode" "$threads" "$OUT_DIR/stdout/$stem.out" "$OUT_DIR/logs/$stem.log"
    log="$OUT_DIR/logs/$stem.log"
    toks="$(sed -n 's/^achieved tok\/s: //p' "$log")"
    info="$(grep '^weights:' "$log")"
    [[ -n "$toks" && -n "$info" ]] || { echo "ERROR: unexpected log $log" >&2; exit 1; }
    load="$(sed -n 's/.*load_ms=\([0-9-]*\).*/\1/p' <<< "$info")"
    off="$(sed -n 's/.*offset_mod64=\([0-9-]*\).*/\1/p' <<< "$info")"
    huge="$(sed -n 's/.*anon_huge_kb=\([0-9-]*\).*/\1/p' <<< "$info")"
    sha="$(sha256sum "$OUT_DIR/stdout/$stem.out" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$round" "$mode" "$threads" "$toks" "$load" "$off" "$huge" "$sha" \
      >> "$OUT_DIR/runs.tsv"
    printf '  mode=%s threads=%s  %8.1f tok/s\n' "$mode" "$threads" "$toks"
  done < <(random_cases "$round")
done

echo "end_local=$(date -Is)" >> "$OUT_DIR/environment.txt"
python3 "$PROJECT_DIR/analyze_weights_layout.py" "$OUT_DIR/runs.tsv" | tee "$OUT_DIR/report.txt"
echo "Saved: $OUT_DIR"
