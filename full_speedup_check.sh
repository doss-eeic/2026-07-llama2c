#!/usr/bin/env bash
# Native-Linux experiment: how much faster is llama2.c with everything combined?
#   huge pages (-w 3), fused layer matmuls (-w 5), int8 output layer with exact argmax (-a 1)
# Sections (all with the power profile set to "performance" unless noted; restored at the end):
#   T  tlbreach on a P-core and an E-core (address translation cost, for reference)
#   V  correctness: many prompts, int8 output layer vs fp32 (-a 1 / -a 3 verification / -a 2 approx)
#   M  main matrix: core configurations x {w0a0, w3a0, w5a0, w0a1, w3a1, w5a1, w5a2}, REPEATS x ROUNDS
#   G  int8 group size sweep (-g) on a few configurations
#   S  generated length (-n) for the original and the fully optimized version
#   B  power profiles (power-saver / balanced / performance) for the original and the optimized version
# Everything (sources, model) is fetched from this branch on GitHub; no clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/full_speedup_check.sh -o full_speedup_check.sh && bash full_speedup_check.sh
# options (env): ROUNDS=10 REPEATS=2 SECTIONS="T V M G S B" VN=12 (prompts in V) MODEL=path DIR=work dir LOCAL_SRC=checkout dir
#                FORCE=1 (run without AC power / without switching the profile)
set -uo pipefail
export LC_ALL=C
BASE="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada"
DIR="${DIR:-$HOME/llama_full_check}"
ROUNDS="${ROUNDS:-10}"
REPEATS="${REPEATS:-2}"
SECTIONS="${SECTIONS:-T V M G S B}"
mkdir -p "$DIR/memlab"
cd "$DIR"
OUT="$DIR/results_$(hostname)_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/outputs"
RAW="$OUT/raw.txt"
PROMPT="Once upon a time"

# name | threads | cpus. Core Ultra 5 225U: CPU 0-3 = 2 P-cores x 2 threads, 4-7 / 8-11 = two E-core clusters
CASES=(
  "P1|1|0"
  "P2|2|0,2"
  "P4_HT|4|0-3"
  "E4|4|4-7"
  "X4_2_2|4|4,5,8,9"
  "E1|1|4"
  "E2|2|4,5"
)
# w | a : weights mode | output layer mode
COMBOS=("0|0" "3|0" "5|0" "0|1" "3|1" "5|1" "5|2")
VPROMPTS=(
  "Once upon a time"
  "The little dog"
  "Lily wanted to"
  "One day, a boy named Tom"
  "The sun was shining"
  "Sara and Ben went to the park"
  "There was a big red ball"
  "Mom said"
  "The cat sat on"
  "In a small town, there lived"
  "Tim found a box"
  "It was a rainy day"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { # LOCAL_SRC=dir copies from a local checkout instead (for testing before a push)
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
for tool in gcc taskset python3 sha256sum cmp; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
command -v curl >/dev/null || command -v wget >/dev/null || { echo "missing tool: curl or wget" >&2; exit 1; }

# ---------------------------------------------------------------- power profile
orig="$(powerprofilesctl get 2>/dev/null || true)"
can_switch=0
if [[ -n "$orig" ]] && powerprofilesctl set performance 2>/dev/null; then
  can_switch=1
  trap 'powerprofilesctl set "$orig" 2>/dev/null; log "power profile restored to $orig"' EXIT
  sleep 3
  log "power profile: performance (was $orig)"
elif [[ "${FORCE:-0}" == 1 ]]; then
  log "WARNING: could not switch to the performance profile; measuring with the current setting"
else
  echo "Could not switch the power profile to performance (powerprofilesctl). Run with FORCE=1 to measure anyway." >&2
  exit 1
fi
profile_now() { if (( can_switch )); then powerprofilesctl get; else echo "performance"; fi; }

# ---------------------------------------------------------------- files
log "downloading sources"
for f in run.c tokenizer.bin memlab/tlbreach.c analyze_full_speedup.py; do
  fetch "$f" "$f" || { log "download failed: $f"; exit 1; }
done
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_core_check/stories15M.bin" \
           "$HOME/llama_power_check/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
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
gcc -Ofast -fopenmp -march=native run.c -lm -o run &&
  gcc -O2 -o tlbreach memlab/tlbreach.c || { log "build failed"; exit 1; }
# (grep without -q: it must read everything, or run gets SIGPIPE and pipefail reports a failure)
OMP_NUM_THREADS=1 ./run "$MODEL" -n 8 -t 0 -i "$PROMPT" -w 5 -a 1 2>&1 >/dev/null | grep '^classifier: mode=int8' >/dev/null ||
  { log "run.c has no int8 output layer (old version on GitHub?)"; exit 1; }

# ---------------------------------------------------------------- environment
cpu_state() {
  echo "profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" \
       "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
       "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)" \
       "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
}
{
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac rounds=$ROUNDS repeats=$REPEATS sections=$SECTIONS model=$MODEL original_profile=${orig:-n/a}"
  cpu_state
  echo; lscpu
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; for c in /sys/devices/system/cpu/cpu[0-9]*; do
    for i in "$c"/cache/index*; do
      [[ "$(cat "$i/level" 2>/dev/null)" == 2 ]] && echo "${c##*/} L2=$(cat "$i/size") shared=$(cat "$i/shared_cpu_list")"
    done
  done | sort -V
  echo; free -m; echo; cat /proc/buddyinfo
  echo; echo "thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo; printf '%s\n' "${CASES[@]}"
  echo; gcc --version | head -1
  echo; OMP_NUM_THREADS=1 ./run "$MODEL" -n 4 -t 0 -i "$PROMPT" -w 5 -a 1 2>&1 >/dev/null
} > "$OUT/environment.txt"

# one measured run -> one RAW line
#   run_case section repeat round name threads cpus w a g n
run_case() {
  local sec="$1" rep="$2" r="$3" name="$4" threads="$5" cpus="$6" w="$7" a="$8" g="$9" n="${10}" sha tok st cand prof
  sha=$(OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads \
        taskset -c "$cpus" ./run "$MODEL" -n "$n" -t 0 -s 42 -i "$PROMPT" -w "$w" -a "$a" -g "$g" \
        2>"$OUT/last.log" | sha256sum | cut -c1-16)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  st=$(sed -n 's/^stage ms\/token: layers=\([0-9.]*\) final=\([0-9.]*\).*/\1 \2/p' "$OUT/last.log")
  cand=$(sed -n 's/.*candidates mean=\([0-9.]*\) min=[0-9]* max=\([0-9]*\).*/\1 \2/p' "$OUT/last.log")
  prof=$(profile_now)
  echo "RAW $sec $rep $r $name $threads $cpus $w $a $g $n $prof ${tok:-NA} ${st:-NA NA} ${cand:-NA NA} $sha" >> "$RAW"
}
spec_of() { local want="$1" s; for s in "${CASES[@]}"; do [[ "${s%%|*}" == "$want" ]] && { echo "$s"; return; }; done; }

cat "$MODEL" > /dev/null

# ---------------------------------------------------------------- T
if has T; then
  log "T tlbreach (P-core cpu0, E-core cpu4)"
  { for c in 0 4; do echo "### cpu$c"; taskset -c "$c" ./tlbreach; done; } > "$OUT/T_tlbreach.txt" 2>&1
fi

# ---------------------------------------------------------------- V
if has V; then
  log "V correctness: ${#VPROMPTS[@]} prompts x {P1, P4_HT} x {-w 0, -w 5}"
  pid=0
  for prompt in "${VPROMPTS[@]:0:${VN:-${#VPROMPTS[@]}}}"; do
    pid=$((pid + 1))
    for cfg in P1 P4_HT; do
      IFS='|' read -r name threads cpus <<< "$(spec_of "$cfg")"
      for w in 0 5; do
        o="$OUT/outputs/p${pid}_${cfg}_w${w}"
        runv() { OMP_NUM_THREADS="$threads" OMP_PROC_BIND=TRUE OMP_PLACES=threads taskset -c "$cpus" \
                 ./run "$MODEL" -n 256 -t 0 -i "$prompt" -w "$w" "$@"; }
        runv -a 0 > "$o.a0.txt" 2>/dev/null
        runv -a 1 > "$o.a1.txt" 2>"$o.a1.log"
        runv -a 3 > "$o.a3.txt" 2>"$o.a3.log"
        runv -a 3 -g 32 > "$o.a3g32.txt" 2>"$o.a3g32.log"
        runv -a 2 > "$o.a2.txt" 2>/dev/null
        same() { cmp -s "$1" "$2" && echo 1 || echo 0; }
        vf() { local x; x=$(sed -n "s/.*$1=\([0-9]*\).*/\1/p" "$2"); echo "${x:-NA}"; }
        first=$(cmp "$o.a0.txt" "$o.a2.txt" 2>/dev/null | awk '{print $5}' | tr -d ',')
        cand=$(sed -n 's/.*candidates mean=\([0-9.]*\) min=[0-9]* max=\([0-9]*\).*/\1 \2/p' "$o.a1.log")
        echo "VER $pid $cfg $w $(same "$o.a0.txt" "$o.a1.txt") $(same "$o.a0.txt" "$o.a3.txt")" \
             "$(vf argmax_mismatch "$o.a3.log") $(vf value_mismatch "$o.a3.log") $(vf best_missed "$o.a3.log")" \
             "$(vf argmax_mismatch "$o.a3g32.log") $(vf best_missed "$o.a3g32.log")" \
             "$(same "$o.a0.txt" "$o.a2.txt") ${first:-NA} ${cand:-NA NA}" >> "$RAW"
      done
    done
  done
  printf '%s\n' "${VPROMPTS[@]}" > "$OUT/outputs/prompts.txt"
fi

# ---------------------------------------------------------------- M
if has M; then
  log "M warm-up"
  for spec in "${CASES[@]}"; do IFS='|' read -r n th cp <<< "$spec"; run_case WARM 0 0 "$n" "$th" "$cp" 0 0 144 256; done
  sed -i '/^RAW WARM /d' "$RAW"
  for ((rep = 1; rep <= REPEATS; rep++)); do
    log "M repeat $rep/$REPEATS: $(( ${#CASES[@]} * ${#COMBOS[@]} )) combinations x $ROUNDS rounds"
    for ((r = 1; r <= ROUNDS; r++)); do
      for spec in $(for s in "${CASES[@]}"; do for c in "${COMBOS[@]}"; do echo "$s|$c"; done; done | shuf); do
        IFS='|' read -r n th cp w a <<< "$spec"
        run_case M "$rep" "$r" "$n" "$th" "$cp" "$w" "$a" 144 256
      done
      log "  repeat $rep round $r/$ROUNDS done"
    done
  done
fi

# ---------------------------------------------------------------- G
if has G; then
  log "G group size sweep: {P1, P4_HT, X4_2_2} x g {16, 32, 48, 96, 144, 288} (+ fp32) x $ROUNDS rounds"
  for ((r = 1; r <= ROUNDS; r++)); do
    for spec in $(for cfg in P1 P4_HT X4_2_2; do for g in 16 32 48 96 144 288 fp32; do echo "$(spec_of $cfg)|$g"; done; done | shuf); do
      IFS='|' read -r n th cp g <<< "$spec"
      if [[ "$g" == fp32 ]]; then run_case G 1 "$r" "$n" "$th" "$cp" 5 0 144 256
      else run_case G 1 "$r" "$n" "$th" "$cp" 5 1 "$g" 256; fi
    done
  done
fi

# ---------------------------------------------------------------- S
if has S; then
  log "S generated length: P4_HT x -n {32, 64, 128, 256} x {w0a0, w5a1} x $ROUNDS rounds"
  IFS='|' read -r n th cp <<< "$(spec_of P4_HT)"
  for ((r = 1; r <= ROUNDS; r++)); do
    for spec in $(for len in 32 64 128 256; do echo "$len|0|0"; echo "$len|5|1"; done | shuf); do
      IFS='|' read -r len w a <<< "$spec"
      run_case S 1 "$r" "$n" "$th" "$cp" "$w" "$a" 144 "$len"
    done
  done
fi

# ---------------------------------------------------------------- B (last: changes the profile)
if has B; then
  if (( can_switch )); then
    log "B power profiles: {P1, P4_HT} x {w0a0, w5a1} x {power-saver, balanced, performance} x 6 rounds"
    for ((r = 1; r <= 6; r++)); do
      for prof in $(printf '%s\n' power-saver balanced performance | shuf); do
        powerprofilesctl set "$prof" 2>/dev/null || continue
        sleep 3
        for spec in $(for cfg in P1 P4_HT; do echo "$(spec_of $cfg)|0|0"; echo "$(spec_of $cfg)|5|1"; done | shuf); do
          IFS='|' read -r n th cp w a <<< "$spec"
          run_case B 1 "$r" "$n" "$th" "$cp" "$w" "$a" 144 256
        done
      done
    done
    powerprofilesctl set performance 2>/dev/null
  else
    log "B skipped (cannot switch power profiles)"
  fi
fi
rm -f "$OUT/last.log"

# ---------------------------------------------------------------- summary
echo "end: $(date -Is) $(cpu_state)" >> "$OUT/environment.txt"
python3 analyze_full_speedup.py "$RAW" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
cp run.c "$OUT/run.c"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
