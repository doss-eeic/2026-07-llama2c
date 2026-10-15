#!/usr/bin/env bash
# Native-Linux follow-up to project/docs/weights_memory_layout_report.md
#   E1: fixed tlbreach (does a 2MB huge page cover 2MB in the TLB on native Linux?)
#   P : power profile vs speed (memory bandwidth and run.c tok/s under each powerprofilesctl profile)
# Everything (sources, model) is fetched from this branch on GitHub; no clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/power_tlb_check.sh -o power_tlb_check.sh && bash power_tlb_check.sh
# options (env): ROUNDS=6  PCPU=2 (P-core for single-thread runs)  MODEL=path  DIR=work dir  FORCE=1 (no AC power)
set -uo pipefail
export LC_ALL=C
BASE="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada"
DIR="${DIR:-$HOME/llama_power_check}"
ROUNDS="${ROUNDS:-6}"
PCPU="${PCPU:-2}"
mkdir -p "$DIR/memlab"
cd "$DIR"
OUT="$DIR/results_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$BASE/$1" -o "$2" 2>/dev/null || wget -q "$BASE/$1" -O "$2"; }

# ---------------------------------------------------------------- checks
ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2
  exit 1
fi
for tool in gcc taskset python3; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
command -v curl >/dev/null || command -v wget >/dev/null || { echo "missing tool: curl or wget" >&2; exit 1; }

# ---------------------------------------------------------------- files
log "downloading sources"
for f in run.c tokenizer.bin memlab/tlbreach.c memlab/probe.c; do
  fetch "$f" "$f" || { log "download failed: $f"; exit 1; }
done
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
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
  gcc -O2 -o tlbreach memlab/tlbreach.c &&
  gcc -Ofast -march=native -o probe memlab/probe.c || { log "build failed"; exit 1; }

# ---------------------------------------------------------------- environment
cpu_state() {
  echo "governor=$(cat /sys/devices/system/cpu/cpu$PCPU/cpufreq/scaling_governor 2>/dev/null)" \
       "epp=$(cat /sys/devices/system/cpu/cpu$PCPU/cpufreq/energy_performance_preference 2>/dev/null)" \
       "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
}
{
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac pinned_cpu=$PCPU rounds=$ROUNDS model=$MODEL"
  echo "power_profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  cpu_state
  powerprofilesctl list 2>/dev/null
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
} > "$OUT/environment.txt"

# ---------------------------------------------------------------- E1 tlbreach (fixed)
log "E1 tlbreach (fixed version)"
{
  for i in 1 2; do echo "### P-core cpu$PCPU run $i"; taskset -c "$PCPU" ./tlbreach; done
  echo "### E-core cpu4"; taskset -c 4 ./tlbreach
} > "$OUT/E1_tlbreach_fixed.txt" 2>&1
cat "$OUT/E1_tlbreach_fixed.txt"

# ---------------------------------------------------------------- power profiles
orig="$(powerprofilesctl get 2>/dev/null || true)"
profiles=()
if [[ -n "$orig" ]]; then
  for p in $(powerprofilesctl list 2>/dev/null | sed -n 's/^[ *]*\([a-z-]*\):$/\1/p'); do
    if powerprofilesctl set "$p" 2>/dev/null; then profiles+=("$p"); else log "cannot switch to $p (skipped)"; fi
  done
  powerprofilesctl set "$orig" 2>/dev/null
  trap 'powerprofilesctl set "$orig" 2>/dev/null; log "power profile restored to $orig"' EXIT
fi
if (( ${#profiles[@]} == 0 )); then
  log "powerprofilesctl unavailable or not switchable: measuring the current setting only"
  profiles=("current")
fi
log "P power profiles: ${profiles[*]} (original: ${orig:-n/a}), $ROUNDS rounds"

measure() { # one round under the profile currently set
  local p="$1" r="$2" t
  echo "STATE $p $r $(cpu_state)"
  taskset -c "$PCPU" ./probe 5 "$MODEL" | tail -1 | awk -v p="$p" -v r="$r" '{print "BW", p, r, $(NF-2), $(NF-1)}'
  for w in 0 3; do
    t=$(OMP_NUM_THREADS=1 taskset -c "$PCPU" ./run "$MODEL" -n 256 -t 0 -s 42 -i "Once upon a time" -w $w 2>&1 >/dev/null | sed -n 's/achieved tok\/s: //p')
    echo "TOK $p $r t1_w$w $t"
    t=$(OMP_NUM_THREADS=4 OMP_PROC_BIND=TRUE OMP_PLACES=threads taskset -c 0-3 ./run "$MODEL" -n 256 -t 0 -s 42 -i "Once upon a time" -w $w 2>&1 >/dev/null | sed -n 's/achieved tok\/s: //p')
    echo "TOK $p $r t4P_w$w $t"
  done
}
cat "$MODEL" > /dev/null
for ((r = 1; r <= ROUNDS; r++)); do
  for p in $(printf '%s\n' "${profiles[@]}" | shuf); do
    [[ "$p" == current ]] || { powerprofilesctl set "$p"; sleep 3; }
    measure "$p" "$r" >> "$OUT/P_power_raw.txt" 2>&1
  done
  log "  round $r/$ROUNDS done"
done

python3 - "$OUT/P_power_raw.txt" > "$OUT/P_power_summary.txt" <<'PY'
import sys, statistics as st
from collections import defaultdict
v = defaultdict(lambda: defaultdict(list)); state = {}
for l in open(sys.argv[1]):
    f = l.split()
    if not f: continue
    if f[0] == "BW":
        v[f[1]]["seq GB/s"].append(float(f[3])); v[f[1]]["matvec GB/s"].append(float(f[4]))
    elif f[0] == "TOK" and len(f) == 5:
        v[f[1]][f[3] + " tok/s"].append(float(f[4]))
    elif f[0] == "STATE":
        state[f[1]] = " ".join(f[3:])
profs = list(v)
keys = ["seq GB/s", "matvec GB/s", "t1_w0 tok/s", "t1_w3 tok/s", "t4P_w0 tok/s", "t4P_w3 tok/s"]
print("median over rounds  (t1 = 1 thread on the pinned P-core, t4P = 4 threads on P-cores 0-3, w0 = mmap, w3 = aligned+hugepage)")
print(f"{'':<16}" + "".join(f"{p:>14}" for p in profs))
for k in keys:
    print(f"{k:<16}" + "".join(f"{st.median(v[p][k]):>14.1f}" if v[p][k] else f"{'-':>14}" for p in profs))
print()
for p in profs:
    print(f"{p}: {state.get(p, '')}")
PY
cat "$OUT/P_power_summary.txt"

tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
