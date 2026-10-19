#!/usr/bin/env bash
# Native-Linux follow-up: separate the two variables behind the huge-page (-w 3) effect,
# core type (P / E) and thread count (1 / 2 / 4), with the power profile fixed to "performance".
#   T : tlbreach (address-translation cost, 4KB vs huge pages) on a P-core and an E-core
#   M : run.c tok/s for every (core type x thread count) case, with -w 0 (mmap) and -w 3 (huge pages)
# Everything (sources, model) is fetched from this branch on GitHub; no clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/core_thread_check.sh -o core_thread_check.sh && bash core_thread_check.sh
# options (env): ROUNDS=10  MODEL=path  DIR=work dir  FORCE=1 (run without AC power / without switching the profile)
set -uo pipefail
export LC_ALL=C
BASE="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada"
DIR="${DIR:-$HOME/llama_core_check}"
ROUNDS="${ROUNDS:-10}"
mkdir -p "$DIR/memlab"
cd "$DIR"
OUT="$DIR/results_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# name | threads | cpus. On a Core Ultra 5 225U: CPU 0-3 = 2 P-cores x 2 threads,
# CPU 4-7 and 8-11 = two E-core clusters (each shares one L2), CPU 12-13 = LP E-cores.
CASES=(
  "P1|1|2"           # one P-core
  "P2|2|0,2"         # two P-cores, one thread each
  "P4|4|0-3"         # two P-cores, two threads each (hyper-threading)
  "E1|1|4"           # one E-core
  "E2|2|4,5"         # two E-cores in the same cluster
  "E4|4|4,5,8,9"     # four E-cores, 2+2 over both clusters (fastest 4-thread E layout in the team's runs)
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$BASE/$1" -o "$2" 2>/dev/null || wget -q "$BASE/$1" -O "$2"; }

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

# ---------------------------------------------------------------- power profile: performance
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
for f in run.c tokenizer.bin memlab/tlbreach.c; do
  fetch "$f" "$f" || { log "download failed: $f"; exit 1; }
done
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_power_check/stories15M.bin" \
           "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
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

# ---------------------------------------------------------------- environment
cpu_state() {
  echo "profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" \
       "governor=$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor 2>/dev/null)" \
       "epp=$(cat /sys/devices/system/cpu/cpu2/cpufreq/energy_performance_preference 2>/dev/null)" \
       "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
}
{
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac rounds=$ROUNDS model=$MODEL original_profile=${orig:-n/a}"
  cpu_state
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; for c in /sys/devices/system/cpu/cpu[0-9]*; do
    for i in "$c"/cache/index*; do
      [[ "$(cat "$i/level" 2>/dev/null)" == 2 ]] && echo "${c##*/} L2=$(cat "$i/size") shared=$(cat "$i/shared_cpu_list")"
    done
  done | sort -V
  echo; printf '%s\n' "${CASES[@]}"
} > "$OUT/environment.txt"
# the case table assumes the 225U layout; warn (but keep going) if this machine differs
maxp=$(cat /sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
maxe=$(cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
(( maxp > maxe )) || log "WARNING: CPU 2 is not faster than CPU 4 here; the P/E labels may be wrong (see environment.txt)"

# ---------------------------------------------------------------- T: tlbreach on P and E
log "T tlbreach (P-core cpu2 x2, E-core cpu4 x2)"
{
  for c in 2 4; do for i in 1 2; do echo "### cpu$c run $i"; taskset -c "$c" ./tlbreach; done; done
} > "$OUT/T_tlbreach.txt" 2>&1
cat "$OUT/T_tlbreach.txt"

# ---------------------------------------------------------------- M: core type x threads x -w
run_case() { # name threads cpus mode round -> one RAW line
  local name="$1" threads="$2" cpus="$3" w="$4" r="$5" t sha
  t=$(OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads \
      taskset -c "$cpus" ./run "$MODEL" -n 256 -t 0 -s 42 -i "Once upon a time" -w "$w" \
      2>"$OUT/last.log" | sha256sum | cut -c1-16)
  sha="$t"
  t=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  echo "RAW $r $name $threads $cpus w$w ${t:-NA} $sha"
}
cat "$MODEL" > /dev/null
log "M warm-up round"
for spec in "${CASES[@]}"; do IFS='|' read -r n th cp <<< "$spec"; run_case "$n" "$th" "$cp" 0 0 > /dev/null; done
log "M $(( ${#CASES[@]} * 2 )) combinations x $ROUNDS rounds (random order each round)"
for ((r = 1; r <= ROUNDS; r++)); do
  for spec in $(for s in "${CASES[@]}"; do echo "$s|0"; echo "$s|3"; done | shuf); do
    IFS='|' read -r n th cp w <<< "$spec"
    run_case "$n" "$th" "$cp" "$w" "$r" >> "$OUT/M_raw.txt"
  done
  log "  round $r/$ROUNDS done"
done
rm -f "$OUT/last.log"

python3 - "$OUT/M_raw.txt" > "$OUT/M_summary.txt" <<'PY'
import sys, statistics as st
from collections import defaultdict
tok = defaultdict(dict); shas = set()
for l in open(sys.argv[1]):
    f = l.split()
    if f[0] != "RAW" or f[6] == "NA": continue
    _, r, name, th, cpus, w, t, sha = f
    tok[(name, w)][r] = float(t); shas.add(sha)
print(f"outputs: {len(shas)} distinct " + ("(OK, all identical)" if len(shas) == 1 else "(MISMATCH!)"))
print("\nmedian tok/s, and huge pages vs mmap (paired median ratio, wins)")
print(f"{'case':<6}{'w0 (mmap)':>11}{'w3 (huge)':>11}{'w3/w0':>9}{'wins':>8}")
names = ["P1", "P2", "P4", "E1", "E2", "E4"]
for n in names:
    a, b = tok.get((n, "w0"), {}), tok.get((n, "w3"), {})
    if not a or not b: continue
    rs = [b[r] / a[r] for r in a if r in b]
    print(f"{n:<6}{st.median(a.values()):>11.1f}{st.median(b.values()):>11.1f}"
          f"{st.median(rs):>8.3f}x{sum(x > 1 for x in rs):>5}/{len(rs)}")
med = {k: st.median(v.values()) for k, v in tok.items() if v}
print("\nsame core type, more threads (w0):  P1->P2->P4  ",
      " -> ".join(f"{med.get((n,'w0'),0):.0f}" for n in names[:3]),
      "   E1->E2->E4  ", " -> ".join(f"{med.get((n,'w0'),0):.0f}" for n in names[3:]))
print("same thread count, P vs E (w0):     " + "   ".join(
      f"{t}T: P/E={med.get(('P'+t,'w0'),0)/max(med.get(('E'+t,'w0'),1),1e-9):.2f}x" for t in "124"))
PY
cat "$OUT/M_summary.txt"
echo "end: $(date -Is) $(cpu_state)" >> "$OUT/environment.txt"

tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
