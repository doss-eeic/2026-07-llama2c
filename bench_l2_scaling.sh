#!/usr/bin/env bash
# L2 topology scaling experiment for instrumented llama2.c (OpenMP).
# Run this script from anywhere after copying it into ~/2026-07-llama2c/.
set -Eeuo pipefail
export LC_ALL=C

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
MODEL="${MODEL:-./stories15M.bin}"
RUN_BIN="${RUN_BIN:-./run}"
OUT_ROOT="${OUT_ROOT:-./benchmark_affinity_stories15M}"
REPEATS="${REPEATS:-7}"
WARMUPS="${WARMUPS:-1}"
STEPS="${STEPS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Once upon a time}"
ORDER_SEED="${ORDER_SEED:-20261012}"
REBUILD="${REBUILD:-0}"
PERF="${PERF:-0}"
PERF_REPEATS="${PERF_REPEATS:-2}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-references,cache-misses}"
ANALYZER="${ANALYZER:-./analyze/analyze_l2_scaling.py}"

for integer in REPEATS WARMUPS STEPS ORDER_SEED PERF_REPEATS; do
  if ! [[ "${!integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $integer must be a nonnegative integer." >&2; exit 1
  fi
done
if (( REPEATS < 1 || STEPS < 1 )); then echo "ERROR: REPEATS and STEPS must be >= 1" >&2; exit 1; fi
for tool in python3 taskset sha256sum; do
  command -v "$tool" >/dev/null || { echo "ERROR: missing $tool" >&2; exit 1; }
done
if [[ "$REBUILD" == 1 ]]; then
  echo 'Rebuilding OpenMP executable: make runomp CC=gcc'
  make runomp CC=gcc
fi
[[ -x "$RUN_BIN" ]] || { echo "ERROR: executable not found: $RUN_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -f "$ANALYZER" ]] || { echo "ERROR: analyzer missing: $ANALYZER" >&2; exit 1; }

# Output folder is intentionally newly created: never overwrite prior measurements.
mkdir -p "$OUT_ROOT"
OUT_DIR="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_l2_scaling"
mkdir "$OUT_DIR"
mkdir "$OUT_DIR/logs" "$OUT_DIR/stdout" "$OUT_DIR/perf"

# cluster_a = CPUs 4-7 (shared 2 MiB L2); cluster_b = CPUs 8-11 (shared 2 MiB L2).
# The A/B symmetry and different splits at fixed thread count help separate
# cluster balance from a specific CPU cluster's performance.
cat > "$OUT_DIR/config.tsv" <<'CASES'
case	threads	cpus	cluster_a	cluster_b
P1	1	0	0	0
A1	1	4	1	0
B1	1	8	0	1
A2	2	4,5	2	0
B2	2	8,9	0	2
X2_1_1	2	4,8	1	1
A3	3	4,5,6	3	0
B3	3	8,9,10	0	3
X3_2_1	3	4,5,8	2	1
X3_1_2	3	4,8,9	1	2
A4	4	4,5,6,7	4	0
B4	4	8,9,10,11	0	4
X4_3_1	4	4,5,6,8	3	1
X4_2_2	4	4,5,8,9	2	2
X4_1_3	4	4,8,9,10	1	3
X5_4_1	5	4,5,6,7,8	4	1
X5_3_2	5	4,5,6,8,9	3	2
X5_2_3	5	4,5,8,9,10	2	3
X5_1_4	5	4,8,9,10,11	1	4
X6_4_2	6	4,5,6,7,8,9	4	2
X6_3_3	6	4,5,6,8,9,10	3	3
X6_2_4	6	4,5,8,9,10,11	2	4
X7_4_3	7	4,5,6,7,8,9,10	4	3
X7_3_4	7	4,5,6,8,9,10,11	3	4
X8_4_4	8	4,5,6,7,8,9,10,11	4	4
CASES

# Check the actual L2 topology; abort rather than silently comparing
# arrangements that don't have the assumed cache-sharing relationship.
for spec in '4:4-7' '5:4-7' '6:4-7' '7:4-7' '8:8-11' '9:8-11' '10:8-11' '11:8-11'; do
  cpu="${spec%%:*}"; expected="${spec#*:}"; found=0
  for cache in /sys/devices/system/cpu/cpu"$cpu"/cache/index*; do
    [[ -f "$cache/level" ]] || continue
    if [[ "$(< "$cache/level")" == 2 ]]; then
      actual="$(< "$cache/shared_cpu_list")"
      if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: CPU $cpu L2 shared with '$actual', expected '$expected'" >&2
        exit 1
      fi
      found=1
      break
    fi
  done
  (( found == 1 )) || { echo "ERROR: CPU $cpu L2 topology unavailable" >&2; exit 1; }
done

# Validate that all requested affinity masks are permitted for this user.
while IFS=$'\t' read -r name threads cpus a b; do
  [[ "$name" == case ]] && continue
  if ! taskset -c "$cpus" /bin/true >/dev/null 2>&1; then
    echo "ERROR: CPU affinity '$cpus' ($name) not permitted here" >&2
    exit 1
  fi
done < "$OUT_DIR/config.tsv"

cat > "$OUT_DIR/environment.txt" <<META
start_local=$(date -Is)
project_dir=$PROJECT_DIR
run_bin=$RUN_BIN
model=$MODEL
repeats=$REPEATS
warmups=$WARMUPS
steps=$STEPS
temperature=$TEMPERATURE
seed=$SEED
prompt=$PROMPT
order_seed=$ORDER_SEED
perf=$PERF
perf_repeats=$PERF_REPEATS
perf_events=$PERF_EVENTS
uname=$(uname -a)
allowed_cpus=$(taskset -pc $$ 2>&1)
META
if command -v git >/dev/null && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  { git rev-parse HEAD; git status --short; } > "$OUT_DIR/git_revision_and_status.txt"
fi
printf 'case\trun\tthreads\tcpus\tcluster_a\tcluster_b\tlog\tstdout\tsha256\n' > "$OUT_DIR/manifest.tsv"

# A different reproducible random permutation of all 25 cases per block.
# Interleaving all cases per block limits the influence of drift in clock,
# temperature and background contention compared with grouped runs.
random_cases() {
  python3 - "$OUT_DIR/config.tsv" "$ORDER_SEED" "$1" <<'PY'
import csv, random, sys
with open(sys.argv[1], newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
random.Random(int(sys.argv[2]) + int(sys.argv[3])).shuffle(rows)
for row in rows:
    print('\t'.join(row[key] for key in ('case','threads','cpus','cluster_a','cluster_b')))
PY
}

run_in_affinity() {
  local threads="$1" cpus="$2" stdout="$3" stderr="$4"
  OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE \
    OMP_PLACES=threads taskset -c "$cpus" \
    "$RUN_BIN" "$MODEL" -n "$STEPS" -t "$TEMPERATURE" -s "$SEED" -i "$PROMPT" \
    >"$stdout" 2>"$stderr"
}

printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Conditions: %s cases x %s measured runs (+ %s warm-up rounds)\n' \
  "$(($(wc -l < "$OUT_DIR/config.tsv") - 1))" "$REPEATS" "$WARMUPS"

for ((round=1; round<=WARMUPS; round++)); do
  echo "Warm-up block $round/$WARMUPS"
  while IFS=$'\t' read -r name threads cpus a b; do
    echo "  warm-up $name ($cpus)"
    run_in_affinity "$threads" "$cpus" /dev/null /dev/null
  done < <(random_cases "$(( -round ))")
done

for ((round=1; round<=REPEATS; round++)); do
  echo "Measured block $round/$REPEATS"
  while IFS=$'\t' read -r name threads cpus a b; do
    stem="${name}_r$(printf '%02d' "$round")"
    out="stdout/$stem.out"; log="logs/$stem.log"
    run_in_affinity "$threads" "$cpus" "$OUT_DIR/$out" "$OUT_DIR/$log"
    if ! grep -Eq '103:[[:alnum:]_-]+:{1,2}[[:space:]]*[0-9.]+' "$OUT_DIR/$log"; then
      echo "ERROR: no stage-103 timing found in $OUT_DIR/$log" >&2
      echo 'The run executable must contain the existing stage-timing instrumentation.' >&2
      exit 1
    fi
    sha="$(sha256sum "$OUT_DIR/$out" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$round" "$threads" "$cpus" "$a" "$b" "$log" "$out" "$sha" \
      >> "$OUT_DIR/manifest.tsv"
    printf '  %-10s CPUs=%-22s run=%d\n' "$name" "$cpus" "$round"
  done < <(random_cases "$round")
done

# Optional second, separate pass: perf overhead is NOT mixed into timing trials.
# For hybrid Intel cores, generic counters may not count identically across
# PMUs, and cache-misses is not the same as L2 misses.
if [[ "$PERF" == 1 ]]; then
  if ! command -v perf >/dev/null; then
    echo 'WARNING: perf unavailable; skipping hardware counters.' >&2
  elif ! perf stat -x, -e "$PERF_EVENTS" -- /bin/true >/dev/null 2>&1; then
    echo 'WARNING: perf permission denied or event unsupported; skipping hardware counters.' >&2
  else
    for ((round=1; round<=PERF_REPEATS; round++)); do
      echo "Perf block $round/$PERF_REPEATS"
      while IFS=$'\t' read -r name threads cpus a b; do
        stem="${name}_p$(printf '%02d' "$round")"
        if ! perf stat -x, -o "$OUT_DIR/perf/$stem.csv" -e "$PERF_EVENTS" -- \
          env OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads \
          taskset -c "$cpus" "$RUN_BIN" "$MODEL" -n "$STEPS" -t "$TEMPERATURE" \
          -s "$SEED" -i "$PROMPT" > "$OUT_DIR/perf/$stem.out" 2> "$OUT_DIR/perf/$stem.log"; then
          echo "WARNING: perf failed for $stem (see $OUT_DIR/perf/$stem.log)" >&2
          rm -f "$OUT_DIR/perf/$stem.csv"
        fi
      done < <(random_cases "$((ORDER_SEED + round + REPEATS))")
    done
  fi
fi

echo "end_local=$(date -Is)" >> "$OUT_DIR/environment.txt"
python3 "$ANALYZER" --dir "$OUT_DIR"
echo "Saved: $OUT_DIR"
