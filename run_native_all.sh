#!/usr/bin/env bash
# Native-Linux experiment kit: runs every memory-placement experiment from the WSL study
# in one go and packs the results into results_<host>_<time>.tar.gz.
#   E1 tlbreach     : does a 2MB huge page give a 2MB TLB entry? (WSL: no)
#   E2 chunkshuffle : same pages, different order -> how much slower? (WSL: matvec 0.70x at 4KB)
#   E3 run.c kinds  : run.c with the weights in 5 kinds of memory (WSL: fresh file/anon 4KB ~0.9x)
#   E4 recache order: the file read first gets the scattered pages? (WSL: yes)
#   E5 bench_weights_layout.sh : the -w 0..3 comparison, P-cores only
#   E6 perf counters (only if perf is allowed without sudo)
# usage: ./run_native_all.sh            (FORCE=1 to run without AC power)
# Run from the repository root (needs gcc, python3, taskset; no sudo).
set -uo pipefail
export LC_ALL=C
cd "$(dirname "${BASH_SOURCE[0]}")"
KIT="$PWD"
OUT="$KIT/results_$(hostname)_$(date +%Y%m%d_%H%M%S)"
WORK="$KIT/work"
PCPU="${PCPU:-2}" # logical CPU for single-thread runs; 0-3 are the P-cores on a Core Ultra 5 225U
MODEL="${MODEL:-$KIT/stories15M.bin}"
# the team branches keep the model in build/
[[ -f "$MODEL" || ! -f "$KIT/build/stories15M.bin" ]] || MODEL="$KIT/build/stories15M.bin"
mkdir -p "$OUT" "$WORK"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
step() { # run one experiment; a failure is logged and the remaining experiments still run
  local name="$1"; shift
  log "START $name"
  if "$@" > "$OUT/$name.txt" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.txt)"; fi
}
pin() { taskset -c "$PCPU" "$@"; }
drop_cache() { sync; for f in "$@"; do dd if="$f" iflag=nocache count=0 status=none; done; }

# ---------------------------------------------------------------- checks
ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2
  exit 1
fi
for tool in gcc python3 taskset dd tar; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

env_info() {
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "energy_performance_preference=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"
  echo "thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag)"
  echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
  echo "pinned_cpu=$PCPU"
  echo; lscpu; echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; for c in /sys/devices/system/cpu/cpu[0-9]*; do
    for i in "$c"/cache/index*; do
      [[ "$(cat "$i/level")" == 2 ]] && echo "${c##*/} L2=$(cat "$i/size") shared=$(cat "$i/shared_cpu_list")"
    done
  done | sort -V
  echo; free -m; echo; cat /proc/buddyinfo
  echo; gcc --version | head -1; python3 --version
}
step 00_environment env_info

# ---------------------------------------------------------------- build
build() {
  set -e
  gcc -Ofast -fopenmp -march=native run.c -lm -o run
  ldd run | grep -E 'gomp|omp'
  gcc -O2 -o tlbreach memlab/tlbreach.c
  gcc -Ofast -march=native -o probe memlab/probe.c
  gcc -Ofast -march=native -o chunkshuffle memlab/chunkshuffle.c
  gcc -O2 -o resident memlab/resident.c
  OMP_NUM_THREADS=1 ./run "$MODEL" -n 16 -t 0 -s 42 -i "Once upon a time" -w 3
}
step 01_build build
[[ -x ./run && -x ./probe ]] || { log "build failed, stopping"; exit 1; }

# ---------------------------------------------------------------- E1
e1() { for i in 1 2; do echo "### P-core cpu$PCPU run $i"; pin ./tlbreach; done; echo "### E-core cpu4"; taskset -c 4 ./tlbreach; }
step E1_tlbreach e1

# ---------------------------------------------------------------- E2
e2() { for i in 1 2; do pin ./chunkshuffle 15; echo; pin ./chunkshuffle 15 nofill; echo; done; }
step E2_chunkshuffle e2

# ---------------------------------------------------------------- E3
e3() {
  local fresh="$WORK/model_fresh.bin" shm="/dev/shm/model_shm_$$.bin"
  cp "$MODEL" "$fresh"; cp "$MODEL" "$shm"
  ./resident "$MODEL" "$fresh" "$shm"
  declare -A C=([A_file_orig]="$MODEL 0" [B_file_fresh]="$fresh 0" [C_tmpfs]="$shm 0" [D_anon4k]="$MODEL 1" [E_anonTHP]="$MODEL 3")
  for r in $(seq 1 20); do
    for k in $(shuf -e "${!C[@]}"); do
      set -- ${C[$k]}
      t=$(OMP_NUM_THREADS=1 taskset -c "$PCPU" ./run "$1" -n 256 -t 0 -s 42 -i "Once upon a time" -w "$2" 2>&1 >/dev/null | sed -n 's/achieved tok\/s: //p')
      echo "RAW $r $k $t"
    done
  done
  rm -f "$shm"
}
summarize_e3() {
  python3 - "$OUT/E3_runc_kinds.txt" <<'PY'
import sys, statistics as st
from collections import defaultdict
d = defaultdict(dict)
for l in open(sys.argv[1]):
    if l.startswith("RAW "):
        _, r, k, v = l.split(); d[k][r] = float(v)
print(f"{'case':<14}{'median':>8}{'min':>8}{'max':>8}   vs A (paired median, wins)")
for k in sorted(d):
    v = list(d[k].values())
    rs = [d[k][r] / d["A_file_orig"][r] for r in d[k]]
    print(f"{k:<14}{st.median(v):8.1f}{min(v):8.1f}{max(v):8.1f}   {st.median(rs):.3f}x {sum(x > 1 for x in rs)}/{len(rs)}")
PY
}
step E3_runc_kinds e3
step E3_runc_kinds_summary summarize_e3

# ---------------------------------------------------------------- E4
e4() {
  local fresh="$WORK/model_fresh.bin" filler="$WORK/filler.bin"
  [[ -f "$fresh" ]] || cp "$MODEL" "$fresh"
  dd if=/dev/urandom of="$filler" bs=1M count=400 status=none
  for order in orig_first orig_first orig_first fresh_first fresh_first fresh_first filler_first filler_first filler_first; do
    drop_cache "$MODEL" "$fresh" "$filler"
    echo "### $order"; grep Normal /proc/buddyinfo
    case $order in
      orig_first) cat "$MODEL" "$fresh" > /dev/null ;;
      fresh_first) cat "$fresh" "$MODEL" > /dev/null ;;
      filler_first) cat "$filler" "$MODEL" "$fresh" > /dev/null ;;
    esac
    ./resident "$MODEL" "$fresh"
    pin ./probe 7 "$MODEL" "$fresh" | tail -3
  done
  drop_cache "$filler"; rm -f "$filler"
}
step E4_recache_order e4

# ---------------------------------------------------------------- E5
e5() { taskset -c 0-3 env MODEL="$MODEL" REPEATS=10 THREADS="1 2 4" REBUILD=0 OUT_ROOT="$OUT/bench_weights_layout" ./bench_weights_layout.sh; }
step E5_bench_weights_layout e5

# ---------------------------------------------------------------- E6
e6() {
  command -v perf >/dev/null || { echo "perf not installed"; return 0; }
  if ! perf stat -e cycles -- true >/dev/null 2>&1; then
    echo "perf not permitted (perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)); skipped"
    return 0
  fi
  for w in 0 1 3; do
    echo "### run -w $w"
    OMP_NUM_THREADS=1 perf stat -e cycles,instructions,dTLB-load-misses,LLC-load-misses,LLC-loads -- \
      taskset -c "$PCPU" ./run "$MODEL" -n 256 -t 0 -s 42 -i "Once upon a time" -w "$w" 2>&1 >/dev/null
  done
}
step E6_perf e6

# ---------------------------------------------------------------- pack
rm -rf "$WORK"
cp run.c "$OUT/"
tar czf "$OUT.tar.gz" -C "$KIT" "$(basename "$OUT")"
log "DONE. Bring this file back: $OUT.tar.gz"
