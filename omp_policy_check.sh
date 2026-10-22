#!/usr/bin/env bash
# Native-Linux experiment: OpenMP wait policy (OMP_WAIT_POLICY) and loop schedule (OMP_SCHEDULE)
#   W  wait policy {default, PASSIVE, ACTIVE} x core configurations x {w0a0, w5a0, w5a1}, REPEATS x ROUNDS
#      (static schedule). Does ACTIVE help, and does the gain of -w 5 (fewer barriers) shrink with it?
#   C  loop schedule {static, dynamic,16, dynamic,4, guided} x {default, ACTIVE} on configurations that mix
#      P- and E-cores (plus P4_HT as a reference), w5a1. Can dynamic scheduling make mixed cores pay off?
# Power profile is set to "performance" and restored at the end. No clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/omp_policy_check.sh -o omp_policy_check.sh && bash omp_policy_check.sh
# options (env): ROUNDS=10 REPEATS=2 SECTIONS="W C" MODEL=path DIR=work dir LOCAL_SRC=checkout dir
#                FORCE=1 (run without AC power / without switching the profile)
set -uo pipefail
export LC_ALL=C
BASE="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada"
DIR="${DIR:-$HOME/llama_omp_check}"
ROUNDS="${ROUNDS:-10}"
REPEATS="${REPEATS:-2}"
SECTIONS="${SECTIONS:-W C}"
mkdir -p "$DIR"
cd "$DIR"
OUT="$DIR/results_$(hostname)_omp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
RAW="$OUT/raw.txt"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE   # set per run below

# name | threads | cpus. Core Ultra 5 225U: CPU 0-3 = 2 P-cores x 2 threads, 4-7 / 8-11 = two E-core clusters
CASES=(
  "P1|1|0"
  "P2|2|0,2"
  "P4_HT|4|0-3"
  "E4|4|4-7"
  "X4_2_2|4|4,5,8,9"
  "PE10|10|0,2,4-11"       # both P-cores (1 thread each) + all 8 E-cores, as in the team's run-fused example
  "PE12|12|0-11"           # every P/E thread except the LP E-cores
)
MIXED=(
  "P4_HT|4|0-3"
  "P2E4|6|0,2,4,5,8,9"
  "PE10|10|0,2,4-11"
  "PE12|12|0-11"
)
POLICIES=(default PASSIVE ACTIVE)
COMBOS=("0|0" "5|0" "5|1")
SCHEDS=("static" "dynamic,16" "dynamic,4" "guided")

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() {
  if [[ -n "${LOCAL_SRC:-}" ]]; then cp "$LOCAL_SRC/$1" "$2"; return; fi
  curl -fsSL "$BASE/$1" -o "$2" 2>/dev/null || wget -q "$BASE/$1" -O "$2"
}
has() { [[ " $SECTIONS " == *" $1 "* ]]; }

# ---------------------------------------------------------------- checks
ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2
  exit 1
fi
for tool in gcc taskset python3 sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
command -v curl >/dev/null || command -v wget >/dev/null || { echo "missing tool: curl or wget" >&2; exit 1; }

orig="$(powerprofilesctl get 2>/dev/null || true)"
if [[ -n "$orig" ]] && powerprofilesctl set performance 2>/dev/null; then
  trap 'powerprofilesctl set "$orig" 2>/dev/null; log "power profile restored to $orig"' EXIT
  sleep 3
  log "power profile: performance (was $orig)"
elif [[ "${FORCE:-0}" == 1 ]]; then
  log "WARNING: could not switch to the performance profile; measuring with the current setting"
else
  echo "Could not switch the power profile to performance (powerprofilesctl). Run with FORCE=1 to measure anyway." >&2
  exit 1
fi

# ---------------------------------------------------------------- files
log "downloading sources"
for f in run.c tokenizer.bin analyze_omp_policy.py; do
  fetch "$f" "$f" || { log "download failed: $f"; exit 1; }
done
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_full_check/stories15M.bin" \
           "$HOME/llama_core_check/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
    [[ -f "$c" && $(stat -c %s "$c") == 60816028 ]] && { MODEL="$c"; break; }
  done
fi
if [[ -z "${MODEL:-}" ]]; then
  log "downloading model (60MB)"
  fetch build/stories15M.bin stories15M.bin || { log "model download failed"; exit 1; }
  MODEL="$DIR/stories15M.bin"
fi
log "model: $MODEL"
log "building"
gcc -Ofast -fopenmp -march=native run.c -lm -o run || { log "build failed"; exit 1; }
# (grep without -q: it must read everything, or run gets SIGPIPE and pipefail reports a failure)
OMP_NUM_THREADS=1 OMP_SCHEDULE=guided ./run "$MODEL" -n 8 -t 0 -i "$PROMPT" -w 5 -a 1 2>&1 >/dev/null |
  grep '^openmp: .*schedule=guided' >/dev/null || { log "run.c does not support OMP_SCHEDULE (old version on GitHub?)"; exit 1; }

{
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac rounds=$ROUNDS repeats=$REPEATS sections=$SECTIONS model=$MODEL original_profile=${orig:-n/a}"
  echo "profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" \
       "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
       "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; printf '%s\n' "${CASES[@]}"; echo; printf '%s\n' "${MIXED[@]}"
  echo; gcc --version | head -1
} > "$OUT/environment.txt"

# run_case section repeat round name threads cpus w a policy sched -> one RAW line
run_case() {
  local sec="$1" rep="$2" r="$3" name="$4" threads="$5" cpus="$6" w="$7" a="$8" pol="$9" sch="${10}" sha tok st
  local envs=(OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads OMP_SCHEDULE="$sch")
  [[ "$pol" != default ]] && envs+=(OMP_WAIT_POLICY="$pol")
  sha=$(env "${envs[@]}" taskset -c "$cpus" ./run "$MODEL" -n 256 -t 0 -s 42 -i "$PROMPT" -w "$w" -a "$a" \
        2>"$OUT/last.log" | sha256sum | cut -c1-16)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  st=$(sed -n 's/^stage ms\/token: layers=\([0-9.]*\) final=\([0-9.]*\).*/\1 \2/p' "$OUT/last.log")
  echo "RAW $sec $rep $r $name $threads $cpus $w $a $pol $sch ${tok:-NA} ${st:-NA NA} $sha" >> "$RAW"
}
cat "$MODEL" > /dev/null

if has W; then
  log "W warm-up"
  for spec in "${CASES[@]}"; do IFS='|' read -r n th cp <<< "$spec"; run_case WARM 0 0 "$n" "$th" "$cp" 0 0 default static; done
  sed -i '/^RAW WARM /d' "$RAW"
  for ((rep = 1; rep <= REPEATS; rep++)); do
    log "W repeat $rep/$REPEATS: $(( ${#CASES[@]} * ${#POLICIES[@]} * ${#COMBOS[@]} )) combinations x $ROUNDS rounds"
    for ((r = 1; r <= ROUNDS; r++)); do
      for spec in $(for s in "${CASES[@]}"; do for p in "${POLICIES[@]}"; do for c in "${COMBOS[@]}"; do echo "$s|$p|$c"; done; done; done | shuf); do
        IFS='|' read -r n th cp pol w a <<< "$spec"
        run_case W "$rep" "$r" "$n" "$th" "$cp" "$w" "$a" "$pol" static
      done
      log "  repeat $rep round $r/$ROUNDS done"
    done
  done
fi

if has C; then
  log "C schedules: ${#MIXED[@]} configurations x ${#SCHEDS[@]} schedules x {default, ACTIVE} (w5 a1) x $ROUNDS rounds"
  for ((r = 1; r <= ROUNDS; r++)); do
    for spec in $(for s in "${MIXED[@]}"; do for sc in "${SCHEDS[@]}"; do for p in default ACTIVE; do echo "$s|$sc|$p"; done; done; done | shuf); do
      IFS='|' read -r n th cp sch pol <<< "$spec"
      run_case C 1 "$r" "$n" "$th" "$cp" 5 1 "$pol" "$sch"
    done
    log "  round $r/$ROUNDS done"
  done
fi
rm -f "$OUT/last.log"

echo "end: $(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"
python3 analyze_omp_policy.py "$RAW" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
cp run.c "$OUT/run.c"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
