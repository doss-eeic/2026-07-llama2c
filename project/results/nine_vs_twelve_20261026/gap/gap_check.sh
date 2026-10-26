#!/usr/bin/env bash
# Why does the team find "9 CPUs is fastest" while rigorous_check.sh found "all 12 threads is fastest"?
# Same pinned binaries, all candidates in every block (random order), temperature under control:
#   phase cool    : before every block, rest until the CPU is back near its idle temperature
#   phase hot     : heat the CPU with continuous load first, then run blocks back to back
#   phase recover : rest until cool again, then cool blocks once more (does the ranking come back?)
# Candidates (all must print the same text; checked by sha256):
#   S_*  schedule sweep on the team's 9 CPUs (0,2,4-10) and on all 12 threads (0-11), master, ACTIVE, -g 288
#   G_*  -g 144 instead of 288          X_*  other CPU sets the team used (0,2,4-11 / check_top #2 / with LP-E)
#   W_*  default wait policy            P_*  the team's persistent OpenMP team (dev/wataru2 f0e4ffbb)
#   B_up original code on the P-cores (baseline)
# Every run records temperature, frequency, throttle counters and package energy (if readable);
# trace.tsv samples temperature/frequency every 2 s for the whole session, blocks.tsv the time spent per block.
# Power profile "performance" for the whole run, restored at the end. No clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/gap_check.sh -o gap_check.sh && bash gap_check.sh
# about 30-45 min with the defaults (19 candidates x 50 blocks; most of the time is resting)
# options (env): PHASES="cool hot recover" COOL_BLOCKS=20 HOT_BLOCKS=20 RECOVER_BLOCKS=10
#                INITIAL_REST=120 (s, before measuring the idle temperature)  COOL_TEMP=idle+COOL_MARGIN (C)
#                COOL_MARGIN=8  REST_MIN=15 (s, minimum rest before a cool block)  COOL_MAX=60 (s, longest wait)
#                RECOVER_WAIT=300 (s, longest wait before the first recover block)
#                HEAT_SECS=180 (s of continuous load before the hot phase)  NOTE=free text  MODEL=path DIR=work dir
#                FORCE=1 (run without AC power / without switching the profile)
set -uo pipefail
export LC_ALL=C
RAWGH="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c"
C_MASTER=256b0b2445ecd2c8c88d085fdf616313d5df91f3
C_PT=f0e4ffbb58ec8e15243509a3f0604674a2fb7379
C_SCRIPT=dev/sakurada
DIR="${DIR:-$HOME/llama_gap}"
PHASES="${PHASES:-cool hot recover}"
COOL_BLOCKS="${COOL_BLOCKS:-20}"
HOT_BLOCKS="${HOT_BLOCKS:-20}"
RECOVER_BLOCKS="${RECOVER_BLOCKS:-10}"
INITIAL_REST="${INITIAL_REST:-120}"
COOL_MARGIN="${COOL_MARGIN:-8}"
REST_MIN="${REST_MIN:-15}"
COOL_MAX="${COOL_MAX:-60}"
RECOVER_WAIT="${RECOVER_WAIT:-300}"
HEAT_SECS="${HEAT_SECS:-180}"
mkdir -p "$DIR/src/upstream" "$DIR/src/team/run" "$DIR/src/team/src" "$DIR/src/pt/src"
cd "$DIR"
OUT="$DIR/results_$(hostname)_gap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
RAW="$OUT/raw.tsv"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND OMP_NUM_THREADS GOMP_SPINCOUNT \
      PERSISTENT_OPENMP OUTPUT_THREADS LLAMA_OUT_THREADS LLAMA_OUT_P LLAMA_OUT_WP

N9=0,2,4,5,6,7,8,9,10            # master-threads-and-output-split.md "9 cores"
PE12=0,1,2,3,4,5,6,7,8,9,10,11   # every P/E thread except LP-E
# id | binary | cpus (placement order) | threads (- = number of cpus) | wait policy | schedule | -g | note
CANDIDATES=(
  "B_up_P4|up|0,1,2,3|-|default|static|-|original code, P-cores (baseline)"
)
for s in static dynamic,16 guided,1 guided,4 guided,16; do
  t="${s/dynamic,/dyn}"; t="${t/guided,/gd}"
  CANDIDATES+=("S_N9_$t|team|$N9|-|ACTIVE|$s|288|team 9 CPUs, schedule sweep")
  CANDIDATES+=("S_PE12_$t|team|$PE12|-|ACTIVE|$s|288|all 12 threads, schedule sweep")
done
CANDIDATES+=(
  "G_PE12_static_g144|team|$PE12|-|ACTIVE|static|144|-g 144 (rigorous_check C07)"
  "X_PE10_gd4|team|0,2,4,5,6,7,8,9,10,11|-|ACTIVE|guided,4|288|P x1 + all 8 E"
  "X_T9b_gd16|team|1,2,3,5,6,7,8,10,11|-|ACTIVE|guided,16|288|check_top #2 (omp-tmp 1575)"
  "X_LP12_gd4|team|0,2,4,5,6,7,8,9,10,11,12,13|-|ACTIVE|guided,4|288|12 physical cores incl. LP-E (team -10.6%)"
  "W_N9_gd4_def|team|$N9|-|default|guided,4|288|9 CPUs without ACTIVE"
  "W_PE12_static_def|team|$PE12|-|default|static|288|12 threads without ACTIVE"
  "P_9to10_gd4|pt|0,2,4,5,6,7,8,9,10,11|9|ACTIVE|guided,4|288|team recommendation: persistent, layers 9 -> output 10"
  "P_PE12_static|pt|$PE12|12|ACTIVE|static|288|persistent team with all 12 threads"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$RAWGH/$1/$2" -o "$3" 2>/dev/null || wget -q "$RAWGH/$1/$2" -O "$3"; }

# ---------------------------------------------------------------- checks
ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2; exit 1
fi
for tool in gcc taskset python3 sha256sum shuf; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
SAMPLER=""
orig="$(powerprofilesctl get 2>/dev/null || true)"
cleanup() {
  [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null
  [[ -n "$orig" ]] && powerprofilesctl set "$orig" 2>/dev/null && log "power profile restored to $orig"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
if [[ -n "$orig" ]] && powerprofilesctl set performance 2>/dev/null; then
  sleep 3
  log "power profile: performance (was $orig)"
elif [[ "${FORCE:-0}" == 1 ]]; then
  orig=""
  log "WARNING: could not switch to the performance profile; measuring with the current setting"
else
  echo "Could not switch the power profile to performance (powerprofilesctl). Run with FORCE=1 to measure anyway." >&2; exit 1
fi

# ---------------------------------------------------------------- sources (pinned) and build
log "downloading pinned sources"
fetch "$C_MASTER" run/run_upstream.c src/upstream/run.c &&
  fetch "$C_MASTER" run/run.c src/team/run/run.c &&
  fetch "$C_MASTER" src/fused.h src/team/src/fused.h &&
  fetch "$C_MASTER" src/classifier.h src/team/src/classifier.h &&
  fetch "$C_PT" run.c src/pt/run.c &&
  fetch "$C_PT" src/omp_pool.h src/pt/src/omp_pool.h &&
  fetch "$C_PT" src/classifier.h src/pt/src/classifier.h &&
  fetch "$C_PT" src/fused.h src/pt/src/fused.h &&
  fetch "$C_PT" src/attention.h src/pt/src/attention.h &&
  fetch "$C_MASTER" tokenizer.bin tokenizer.bin || { log "download failed"; exit 1; }
fetch "$C_SCRIPT" gap_check.sh "$OUT/gap_check.sh" 2>/dev/null || cp "$0" "$OUT/gap_check.sh" 2>/dev/null
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/llama_rigorous/stories15M.bin" "$HOME/Downloads/native_kit/stories15M.bin" \
           "$HOME/llama_nine_check/stories15M.bin" "$HOME/llama_omp_check/stories15M.bin" \
           "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
    [[ -f "$c" && $(stat -c %s "$c") == 60816028 ]] && { MODEL="$c"; break; }
  done
fi
if [[ -z "${MODEL:-}" ]]; then
  log "downloading model (60MB)"
  fetch "$C_MASTER" build/stories15M.bin stories15M.bin || { log "model download failed"; exit 1; }
  MODEL="$DIR/stories15M.bin"
fi
log "model: $MODEL"
log "building (gcc -Ofast -fopenmp -march=native)"
CFLAGS=(-Ofast -fopenmp -march=native)
gcc "${CFLAGS[@]}" src/upstream/run.c -lm -o bin-up &&
  (cd src/team && gcc -I. "${CFLAGS[@]}" run/run.c -lm -o ../../bin-team) &&
  (cd src/pt && gcc "${CFLAGS[@]}" run.c -lm -o ../../bin-pt) || { log "build failed"; exit 1; }

# ---------------------------------------------------------------- monitoring helpers (no forks: read builtin)
ZONES=(/sys/class/thermal/thermal_zone*/temp)
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
max_temp() { local m=0 t z; for z in "${ZONES[@]}"; do read -r t < "$z" 2>/dev/null || t=0; (( t > m )) && m=$t; done; echo $((m / 1000)); }
mean_freq() { local s=0 n=0 c f; for c in ${1//,/ }; do read -r f < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || f=0; s=$((s + f)); n=$((n + 1)); done; echo $((s / n / 1000)); }
throttle() { local s=0 f v; for f in /sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count; do read -r v < "$f" 2>/dev/null || v=0; s=$((s + v)); done; echo $s; }
energy() { local e; read -r e < "$RAPL" 2>/dev/null && echo "$e" || echo NA; }
supplies() { local f v o=""; for f in /sys/class/power_supply/*/online; do read -r v < "$f" 2>/dev/null || v=?; o+="$(basename "$(dirname "$f")")=$v,"; done; echo "${o%,}"; }
now_s() { date +%s.%N; }

# ---------------------------------------------------------------- environment record
{
  echo "start=$(date -Is) host=$(hostname) note=${NOTE:-}"
  echo "phases=$PHASES cool_blocks=$COOL_BLOCKS hot_blocks=$HOT_BLOCKS recover_blocks=$RECOVER_BLOCKS"
  echo "initial_rest=$INITIAL_REST cool_margin=$COOL_MARGIN cool_temp_given=${COOL_TEMP:-} rest_min=$REST_MIN cool_max=$COOL_MAX recover_wait=$RECOVER_WAIT heat_secs=$HEAT_SECS"
  echo "ac_online=$ac supplies=$(supplies) original_profile=${orig:-n/a} profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)" \
       "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
       "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)" "rapl_readable=$([[ $(energy) != NA ]] && echo yes || echo no)"
  for f in /sys/class/powercap/intel-rapl:0/constraint_*_{name,power_limit_uw,time_window_us}; do
    [[ -r "$f" ]] && echo "rapl $(basename "$f")=$(cat "$f")"
  done
  uname -a; head -2 /etc/os-release; gcc --version | head -1
  echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -c1-16)"
  for f in src/upstream/run.c src/team/run/run.c src/team/src/fused.h src/team/src/classifier.h \
           src/pt/run.c src/pt/src/omp_pool.h src/pt/src/classifier.h src/pt/src/fused.h src/pt/src/attention.h tokenizer.bin; do
    echo "source $f sha256=$(sha256sum "$f" | cut -c1-16)"
  done
  echo "commits: master=$C_MASTER persistent=$C_PT"
  echo; printf 'cand %s\n' "${CANDIDATES[@]}"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; uptime; free -m
  echo; echo "other processes using CPU:"; ps -eo pcpu,comm --sort=-pcpu | head -8
} > "$OUT/environment.txt"

printf 'phase\tblock\torder\tid\tbinary\tcpus\tthreads\tpolicy\tschedule\tg\ttok_s\tsha\ttemp_before\ttemp_after\tfreq_before\tfreq_after\tthrottle_delta\tenergy_uj\twall_s\tclock\n' > "$RAW"
printf 'phase\tblock\tstart\twait_s\ttemp_start\ttemp_end\trun_s\tthrottle_total\tsupplies\tprofile\n' > "$OUT/blocks.tsv"
printf 'clock\tphase\ttemp\tfreq_pe\tenergy_uj\n' > "$OUT/trace.tsv"
echo setup > "$OUT/.phase"
sampler() {
  local p
  while :; do
    read -r p < "$OUT/.phase" 2>/dev/null || p=?
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$p" "$(max_temp)" "$(mean_freq $PE12)" "$(energy)" >> "$OUT/trace.tsv"
    sleep 2
  done
}
sampler &
SAMPLER=$!
taskset -cp 13 "$SAMPLER" > /dev/null 2>&1   # keep the sampler off the measured cores (LP-E)

run_one() { # phase block order spec
  local ph="$1" blk="$2" ord="$3" spec="$4" id bin cpus thr pol sch g note n places sha tok tb ta fb fa th0 th1 e0 e1 de t0 t1
  IFS='|' read -r id bin cpus thr pol sch g note <<< "$spec"
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  [[ "$thr" == - ]] && thr=$n
  places=$(tr ',' '\n' <<< "$cpus" | sed 's/.*/{&}/' | paste -sd,)
  local envs=(OMP_NUM_THREADS="$thr" OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_SCHEDULE="$sch")
  [[ "$pol" != default ]] && envs+=(OMP_WAIT_POLICY="$pol")
  local args
  case "$bin" in
    up)   args=(./bin-up "$MODEL" -z tokenizer.bin) ;;
    team) args=(./bin-team "$MODEL" -z tokenizer.bin -w 1 -a 1 -g "$g") ;;
    pt)   args=(./bin-pt "$MODEL" -z tokenizer.bin -w 1 -a 1 -g "$g"); envs+=(PERSISTENT_OPENMP=1 OUTPUT_THREADS="$n") ;;
  esac
  tb=$(max_temp); fb=$(mean_freq "$cpus"); th0=$(throttle); e0=$(energy); t0=$(now_s)
  sha=$(env "${envs[@]}" taskset -c "$cpus" "${args[@]}" -n 256 -t 0 -s 42 -i "$PROMPT" 2>"$OUT/last.log" | sha256sum | cut -c1-16)
  t1=$(now_s); e1=$(energy); ta=$(max_temp); fa=$(mean_freq "$cpus"); th1=$(throttle)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  awk -v t="${tok:-0}" 'BEGIN{exit !(t > 0)}' || tok=NA
  de=NA; [[ "$e0" != NA && "$e1" != NA ]] && (( e1 >= e0 )) && de=$((e1 - e0))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ph" "$blk" "$ord" "$id" "$bin" "$cpus" "$thr" "$pol" "$sch" "$g" \
    "$tok" "$sha" "$tb" "$ta" "$fb" "$fa" "$((th1 - th0))" "$de" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b - a}')" "$t1" >> "$RAW"
}

run_block() { # phase block wait_s
  local ph="$1" b="$2" w="$3" ord=0 tstart t0 spec
  tstart=$(max_temp); t0=$(date +%s)
  while IFS= read -r spec; do
    ord=$((ord + 1)); run_one "$ph" "$b" "$ord" "$spec"
  done < <(printf '%s\n' "${CANDIDATES[@]}" | shuf)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ph" "$b" "$t0" "$w" "$tstart" "$(max_temp)" "$(( $(date +%s) - t0 ))" \
    "$(throttle)" "$(supplies)" "$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/blocks.tsv"
}

WAITED=0
wait_cool() { # rest REST_MIN, then until two readings in a row are <= COOL_TEMP (at most $1 s, default COOL_MAX)
  local start below=0 t max="${1:-$COOL_MAX}"
  start=$(date +%s)
  sleep "$REST_MIN"
  while :; do
    t=$(max_temp)
    if (( t <= COOL_TEMP )); then below=$((below + 1)); else below=0; fi
    (( below >= 2 )) && break
    if (( $(date +%s) - start >= max )); then log "    still ${t} C after ${max}s (target ${COOL_TEMP} C); measuring anyway"; break; fi
    sleep 3
  done
  WAITED=$(( $(date +%s) - start ))
}

heat() { # continuous load: all 12 threads, ACTIVE, back to back
  local end=$(( $(date +%s) + HEAT_SECS )) places
  places=$(tr ',' '\n' <<< "$PE12" | sed 's/.*/{&}/' | paste -sd,)
  while (( $(date +%s) < end )); do
    OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_WAIT_POLICY=ACTIVE OMP_SCHEDULE=static \
      taskset -c "$PE12" ./bin-team "$MODEL" -z tokenizer.bin -w 1 -a 1 -g 288 -n 256 -t 0 -s 42 -i "$PROMPT" > /dev/null 2>&1
  done
}

phase_blocks() { case "$1" in cool) echo "$COOL_BLOCKS" ;; hot) echo "$HOT_BLOCKS" ;; recover) echo "$RECOVER_BLOCKS" ;; esac; }

# ---------------------------------------------------------------- warm-up, idle temperature
cat "$MODEL" > /dev/null
log "warm-up: every candidate once (not recorded)"
for spec in "${CANDIDATES[@]}"; do run_one warm 0 0 "$spec"; done
if [[ $(awk -F'\t' 'NR > 1 {print $12}' "$RAW" | sort -u | wc -l) != 1 ]]; then
  log "ERROR: candidates print different text in the warm-up:"; awk -F'\t' 'NR > 1 {print $4, $12}' "$RAW" | tee -a "$OUT/progress.log"; exit 1
fi
sed -i '/^warm\t/d' "$RAW"
echo idle > "$OUT/.phase"
log "resting ${INITIAL_REST}s to measure the idle temperature (do not use the PC from now on)"
sleep "$INITIAL_REST"
s=0; for i in 1 2 3 4 5 6 7 8 9 10; do s=$((s + $(max_temp))); sleep 1; done
IDLE_TEMP=$((s / 10))
COOL_TEMP="${COOL_TEMP:-$((IDLE_TEMP + COOL_MARGIN))}"
echo "idle_temp=$IDLE_TEMP cool_temp=$COOL_TEMP" >> "$OUT/environment.txt"
log "idle temperature ${IDLE_TEMP} C -> cool blocks start at <= ${COOL_TEMP} C"

total=0; for ph in $PHASES; do total=$((total + $(phase_blocks "$ph"))); done
done_blocks=0; T_START=$(date +%s)
log "${#CANDIDATES[@]} candidates per block, $total blocks in phases: $PHASES (about 30-45 min from here)"

# ---------------------------------------------------------------- phases
for ph in $PHASES; do
  nb=$(phase_blocks "$ph")
  case "$ph" in
    cool|recover)
      echo "$ph" > "$OUT/.phase"
      log "phase $ph: $nb blocks, each after resting to <= ${COOL_TEMP} C"
      for ((b = 1; b <= nb; b++)); do
        if [[ "$ph" == recover && $b == 1 ]]; then wait_cool "$RECOVER_WAIT"; else wait_cool; fi
        run_block "$ph" "$b" "$WAITED"
        done_blocks=$((done_blocks + 1))
        el=$(( $(date +%s) - T_START ))
        log "  $ph block $b/$nb (waited ${WAITED}s, now $(max_temp) C)  elapsed $((el / 60)) min, rough ETA $(( el * (total - done_blocks) / done_blocks / 60 )) min"
      done ;;
    hot)
      echo heat > "$OUT/.phase"
      log "phase hot: heating with continuous load for ${HEAT_SECS}s (now $(max_temp) C)"
      heat
      echo hot > "$OUT/.phase"
      log "phase hot: $nb blocks back to back, no rest (now $(max_temp) C)"
      for ((b = 1; b <= nb; b++)); do
        run_block hot "$b" 0
        done_blocks=$((done_blocks + 1))
        el=$(( $(date +%s) - T_START ))
        log "  hot block $b/$nb (now $(max_temp) C)  elapsed $((el / 60)) min, rough ETA $(( el * (total - done_blocks) / done_blocks / 60 )) min"
      done ;;
    *) log "unknown phase $ph (skipped)" ;;
  esac
done
echo done > "$OUT/.phase"
rm -f "$OUT/last.log"
echo "end=$(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a) supplies=$(supplies)" >> "$OUT/environment.txt"

# ---------------------------------------------------------------- analysis (also saved for combining sessions)
cat > "$OUT/analyze_gap.py" <<'PY'
#!/usr/bin/env python3
"""usage: python3 analyze_gap.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows, cand, blocks = [], {}, {}
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
    for r in csv.DictReader(open(os.path.join(d, "blocks.tsv")), delimiter="\t"):
        blocks[(s, r["phase"], int(r["block"]))] = r
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("cand "):
            f = l[5:].rstrip("\n").split("|")
            cand.setdefault(f[0], f)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
for r in ok:
    r["tok"] = float(r["tok_s"])
PH = [p for p in ("cool", "hot", "recover") if any(r["phase"] == p for r in ok)]
ids = [i for i in cand if any(r["id"] == i for r in ok)]
by = defaultdict(lambda: defaultdict(dict))  # phase -> id -> (session, block) -> tok/s
for r in ok:
    by[r["phase"]][r["id"]][(r["session"], int(r["block"]))] = r["tok"]
rng = random.Random(42)
nan = float("nan")


def med(ph, i):
    v = list(by[ph][i].values())
    return st.median(v) if v else nan


def ci_median(v, n=3000):
    m = sorted(st.median(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def paired(ph, a, b, n=3000):
    A, B = by[ph].get(a, {}), by[ph].get(b, {})
    rs = [A[k] / B[k] for k in A if k in B]
    if not rs:
        return None
    lo, hi = ci_median(rs, n)
    return st.median(rs), lo, hi, sum(x > 1 for x in rs), len(rs)


def fmt_pair(p):
    if p is None:
        return f"{'-':>27}"
    r, lo, hi, w, n = p
    mark = "*" if lo > 1 or hi < 1 else " "
    return f"{r:6.3f}{mark}[{lo:.3f},{hi:.3f}] {w:>2}/{n:<2}"


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows:
    shas[r["sha"]].add(r["id"])
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else
      "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
print("\n== conditions per phase ==")
for ph in PH:
    R = [r for r in rows if r["phase"] == ph]
    tb = [int(r["temp_before"]) for r in R if r["temp_before"].isdigit()]
    th = sum(int(r["throttle_delta"]) for r in R if r["throttle_delta"].lstrip("-").isdigit())
    W = [float(r["energy_uj"]) / 1e6 / float(r["wall_s"]) for r in R if r["energy_uj"] not in ("NA", "") and float(r["wall_s"]) > 0]
    B = [b for k, b in blocks.items() if k[1] == ph]
    waits = [int(b["wait_s"]) for b in B]
    print(f"  {ph:<8} blocks {len(B):>3}  temp before run min/median/max {min(tb)}/{st.median(tb):.0f}/{max(tb)} C"
          f"  throttle events {th}" + (f"  package W median {st.median(W):.1f}" if W else "")
          + (f"  rest per block median {st.median(waits):.0f}s" if any(waits) else ""))

base = "B_up_P4"
for ph in PH:
    order = sorted(ids, key=lambda i: -med(ph, i))
    print(f"\n== ranking, phase {ph} (tok/s; 'team' = 1000 / mean ms per token, the team's statistic) ==")
    print(f"  {'id':<20}{'n':>4}{'median':>8}{'95% CI':>14}{'mean':>7}{'team':>7}{'min':>7}{'CV%':>6}{'xbase':>7}{'drops':>6}   note")
    cm = {i: med("cool", i) if "cool" in by else med(ph, i) for i in ids}
    for i in order:
        v = list(by[ph][i].values())
        if not v:
            continue
        lo, hi = ci_median(v, 2000)
        hm = len(v) / sum(1 / x for x in v)
        drops = sum(x < 0.85 * cm[i] for x in v)
        print(f"  {i:<20}{len(v):>4}{med(ph, i):>8.0f}{f'[{lo:.0f},{hi:.0f}]':>14}{st.mean(v):>7.0f}{hm:>7.0f}{min(v):>7.0f}"
              f"{st.pstdev(v) / st.mean(v) * 100:>6.1f}{med(ph, i) / med(ph, base):>7.2f}{drops:>6}   {cand[i][7]}")
print("  (drops = runs below 0.85 x the candidate's cool-phase median)")

Q = [
    ("Q1 team's own test: 9 / 12, both guided,4", "S_N9_gd4", "S_PE12_gd4"),
    ("Q2 each at my best: 12 static / 9 guided,4", "S_PE12_static", "S_N9_gd4"),
    ("Q3 12: guided,4 / static", "S_PE12_gd4", "S_PE12_static"),
    ("Q3 9: guided,4 / static", "S_N9_gd4", "S_N9_static"),
    ("Q3 12: guided,16 / static", "S_PE12_gd16", "S_PE12_static"),
    ("Q4 12 static: g144 / g288", "G_PE12_static_g144", "S_PE12_static"),
    ("Q5 LP-E 12 guided,4 / 9 guided,4", "X_LP12_gd4", "S_N9_gd4"),
    ("Q6 ACTIVE effect, 9 guided,4", "S_N9_gd4", "W_N9_gd4_def"),
    ("Q6 ACTIVE effect, 12 static", "S_PE12_static", "W_PE12_static_def"),
    ("Q6 no ACTIVE: 12 static / 9 guided,4", "W_PE12_static_def", "W_N9_gd4_def"),
    ("Q7 persistent 9->10 / normal 9 g4", "P_9to10_gd4", "S_N9_gd4"),
    ("Q7 persistent 9->10 / normal 12 static", "P_9to10_gd4", "S_PE12_static"),
    ("Q7 persistent 12 / normal 12 (static)", "P_PE12_static", "S_PE12_static"),
    ("Q8 check_top#2 guided,16 / 9 guided,4", "X_T9b_gd16", "S_N9_gd4"),
    ("Q8 0,2,4-11 guided,4 / 9 guided,4", "X_PE10_gd4", "S_N9_gd4"),
]
print("\n== questions: paired ratio first/second within the same block, median [95% CI] first-faster blocks ==")
print("   (* = CI excludes 1)")
print(f"  {'':<42}" + "".join(f"{ph:^28}" for ph in PH))
for label, a, b in Q:
    print(f"  {label:<42}" + "".join(fmt_pair(paired(ph, a, b)) + " " for ph in PH))
for ph in PH:
    b12 = max((i for i in ids if i.startswith("S_PE12_")), key=lambda i: med(ph, i), default=None)
    b9 = max((i for i in ids if i.startswith("S_N9_")), key=lambda i: med(ph, i), default=None)
    if b12 and b9:
        print(f"  best schedule each ({ph}): {b12} / {b9}  {fmt_pair(paired(ph, b12, b9))}")

print("\n== schedule curve: median tok/s relative to the same CPU set with static ==")
scheds = ["static", "dyn16", "gd1", "gd4", "gd16"]
for ph in PH:
    print(f"  phase {ph}: {'':<6}" + "".join(f"{s:>8}" for s in scheds))
    for cs in ("N9", "PE12"):
        ref = med(ph, f"S_{cs}_static")
        print(f"    {cs:<14}" + "".join(f"{med(ph, f'S_{cs}_{s}') / ref:>8.3f}" for s in scheds) + f"   (static = {ref:.0f})")

print("\n== temperature: paired ratio by the temperature just before the pair (all phases pooled) ==")
temp = {(r["session"], r["phase"], int(r["block"]), r["id"]): int(r["temp_before"]) for r in rows if r["temp_before"].isdigit()}
bins = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 200)]
for label, a, b in (("12 static / 9 guided,4", "S_PE12_static", "S_N9_gd4"), ("12 guided,4 / 9 guided,4", "S_PE12_gd4", "S_N9_gd4"),
                    ("persistent 9->10 / 12 static", "P_9to10_gd4", "S_PE12_static")):
    cells = []
    for lo, hi in bins:
        rs = []
        for ph in PH:
            A, B = by[ph].get(a, {}), by[ph].get(b, {})
            for k in A:
                if k in B:
                    t = (temp.get((k[0], ph, k[1], a), 0) + temp.get((k[0], ph, k[1], b), 0)) / 2
                    if lo <= t < hi:
                        rs.append(A[k] / B[k])
        cells.append(f"{st.median(rs):.3f} (n{len(rs)})" if rs else "-")
    print(f"  {label:<30}" + "".join(f"{f'{lo}-{hi}C' if hi < 200 else f'>={lo}C':>8}: {c:<14}" for (lo, hi), c in zip(bins, cells)))
print("  tok/s median by temperature before the run:")
for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4", "X_T9b_gd16"):
    if i not in ids:
        continue
    cells = []
    for lo, hi in bins:
        v = [r["tok"] for r in ok if r["id"] == i and r["temp_before"].isdigit() and lo <= int(r["temp_before"]) < hi]
        cells.append(f"{st.median(v):.0f} (n{len(v)})" if v else "-")
    print(f"  {i:<30}" + "".join(f"{f'{lo}-{hi}C' if hi < 200 else f'>={lo}C':>8}: {c:<14}" for (lo, hi), c in zip(bins, cells)))

if "hot" in by:
    print("\n== hot phase block by block: tok/s relative to the cool median ==")
    show = [i for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4") if i in ids]
    print(f"  {'block':>5}{'temp':>6}" + "".join(f"{i:>16}" for i in show))
    for (s, ph, b), blk in sorted(blocks.items()):
        if ph != "hot":
            continue
        print(f"  {b:>5}{blk['temp_start']:>6}" + "".join(
            f"{by['hot'][i].get((s, b), nan) / (med('cool', i) if 'cool' in by else nan):>16.3f}" for i in show))

if len(sys.argv) > 2:
    print("\n== per-session median tok/s (reproducibility) ==")
    for ph in PH:
        for i in ("S_PE12_static", "S_N9_gd4", "S_PE12_gd4", "P_9to10_gd4"):
            if i in ids:
                print(f"  {ph:<8}{i:<18}" + "  ".join(
                    f"s{s}: {st.median([v for k, v in by[ph][i].items() if k[0] == s]):.0f}"
                    for s in range(1, len(sys.argv)) if any(k[0] == s for k in by[ph][i])))
PY
python3 "$OUT/analyze_gap.py" "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
