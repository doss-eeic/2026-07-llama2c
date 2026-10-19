#!/usr/bin/env bash
# Native-Linux follow-up: how much do huge pages (-w 3) speed up each stage (transformer layers /
# output layer) under the core configurations of the team's Day4 plan-2 table, with the power
# profile fixed to "performance". Also estimates the best "layers on config X + output layer on
# config Y" combination (the team's plan 2A) with and without huge pages.
#   T : tlbreach (address-translation cost, 4KB vs huge pages) on a P-core and an E-core
#   M : run.c, every core configuration x {-w 0 (mmap), -w 3 (huge pages)}; tok/s and ms/token
#       of the layers and of the output layer (final rmsnorm + classifier)
# Everything (sources, model) is fetched from this branch on GitHub; no clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/core_thread_check.sh -o core_thread_check.sh && bash core_thread_check.sh
# options (env): ROUNDS=10  MODEL=path  DIR=work dir  LOCAL_SRC=checkout dir  FORCE=1 (no AC power / no profile switch)
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
# The first five are the configurations of the team's Day4 plan-2 table; E1/E2 are added so that
# core type and thread count can be compared separately.
CASES=(
  "P1|1|0"           # one P-core
  "P2|2|0,2"         # two P-cores, one thread each
  "P4_HT|4|0-3"      # two P-cores, two threads each (hyper-threading)
  "E4|4|4-7"         # four E-cores in one cluster
  "X4_2_2|4|4,5,8,9" # four E-cores, 2+2 over both clusters
  "E1|1|4"           # one E-core
  "E2|2|4,5"         # two E-cores in the same cluster
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { # LOCAL_SRC=dir copies from a local checkout instead (for testing before a push)
  if [[ -n "${LOCAL_SRC:-}" ]]; then cp "$LOCAL_SRC/$1" "$2"; return; fi
  curl -fsSL "$BASE/$1" -o "$2" 2>/dev/null || wget -q "$BASE/$1" -O "$2"
}

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
OMP_NUM_THREADS=1 ./run "$MODEL" -n 8 -t 0 -s 42 -i "Once upon a time" 2>&1 >/dev/null | grep -q '^stage ms/token' ||
  { log "run.c has no stage timing (old version on GitHub?)"; exit 1; }

# ---------------------------------------------------------------- environment
cpu_state() {
  echo "profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" \
       "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
       "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)" \
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
maxp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
maxe=$(cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
(( maxp > maxe )) || log "WARNING: CPU 0 is not faster than CPU 4 here; the P/E labels may be wrong (see environment.txt)"

# ---------------------------------------------------------------- T: tlbreach on P and E
log "T tlbreach (P-core cpu0 x2, E-core cpu4 x2)"
{
  for c in 0 4; do for i in 1 2; do echo "### cpu$c run $i"; taskset -c "$c" ./tlbreach; done; done
} > "$OUT/T_tlbreach.txt" 2>&1
cat "$OUT/T_tlbreach.txt"

# ---------------------------------------------------------------- M: configuration x -w
run_case() { # name threads cpus mode round -> one RAW line
  local name="$1" threads="$2" cpus="$3" w="$4" r="$5" sha tok st
  sha=$(OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads \
        taskset -c "$cpus" ./run "$MODEL" -n 256 -t 0 -s 42 -i "Once upon a time" -w "$w" \
        2>"$OUT/last.log" | sha256sum | cut -c1-16)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  st=$(sed -n 's/^stage ms\/token: layers=\([0-9.]*\) final=\([0-9.]*\).*/\1 \2/p' "$OUT/last.log")
  echo "RAW $r $name $threads $cpus w$w ${tok:-NA} ${st:-NA NA} $sha"
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

python3 - "$OUT/M_raw.txt" "$OUT/environment.txt" > "$OUT/M_summary.txt" <<'PY'
import sys, statistics as st
from collections import defaultdict
# v[(name, w)][metric][round]
v = defaultdict(lambda: defaultdict(dict)); shas = set(); cpus = {}
for l in open(sys.argv[1]):
    f = l.split()
    if len(f) != 10 or f[0] != "RAW" or "NA" in f: continue
    _, r, name, th, cpu, w, tok, lay, fin, sha = f
    d = v[(name, w)]
    d["tok"][r], d["layers"][r], d["final"][r] = float(tok), float(lay), float(fin)
    d["forward"][r] = float(lay) + float(fin)
    cpus[name] = cpu; shas.add(sha)
names = [n for n in ["P1", "P2", "P4_HT", "E4", "X4_2_2", "E1", "E2"] if (n, "w0") in v and (n, "w3") in v]
med = lambda n, w, m: st.median(v[(n, w)][m].values())
def paired(n, m):  # speedup of -w 3 over -w 0 for metric m (time: w0/w3, tok/s: w3/w0)
    a, b = v[(n, "w0")][m], v[(n, "w3")][m]
    rs = [(b[r] / a[r]) if m == "tok" else (a[r] / b[r]) for r in a if r in b]
    return st.median(rs), sum(x > 1 for x in rs), len(rs)

print(f"outputs: {len(shas)} distinct " + ("(OK, all identical)" if len(shas) == 1 else "(MISMATCH!)"))
print("[" + next((l.strip() for l in open(sys.argv[2]) if l.startswith("profile=")), "") + "]")

print("\n1) ms/token per stage (median), same layout as the team's Day4 plan-2 table")
print(f"{'config':<8}{'CPU':<10}{'layers w0':>10}{'w3':>8}{'output w0':>11}{'w3':>8}{'forward w0':>12}{'w3':>8}{'tok/s w0':>10}{'w3':>8}")
for n in names:
    print(f"{n:<8}{cpus[n]:<10}" + "".join(
        f"{med(n,'w0',m):>{wd}.3f}{med(n,'w3',m):>8.3f}" if m != "tok" else f"{med(n,'w0',m):>{wd}.0f}{med(n,'w3',m):>8.0f}"
        for m, wd in (("layers", 10), ("final", 11), ("forward", 12), ("tok", 10))))

print("\n2) huge-page speedup per stage (-w 3 vs -w 0, paired median over rounds, wins)")
print(f"{'config':<8}{'layers':>16}{'output layer':>18}{'forward':>16}")
for n in names:
    print(f"{n:<8}" + "".join(f"{paired(n,m)[0]:>9.3f}x {paired(n,m)[1]:>2}/{paired(n,m)[2]:<2}" for m in ("layers", "final", "forward")))

print("\n3) estimate for plan 2A: layers on the best config + output layer on the best config")
print("   (sum of separately measured stage medians; the cost of switching cores between stages is NOT included)")
best = {}
for w in ("w0", "w3"):
    bl = min(names, key=lambda n: med(n, w, "layers")); bf = min(names, key=lambda n: med(n, w, "final"))
    bs = min(names, key=lambda n: med(n, w, "forward"))
    est = med(bl, w, "layers") + med(bf, w, "final")
    best[w] = (est, med(bs, w, "forward"))
    print(f"   {w}: layers {bl} {med(bl,w,'layers'):.3f} + output {bf} {med(bf,w,'final'):.3f} = {est:.3f} ms/token"
          f" ({1000/est:.0f} tok/s)   best single config: {bs} {med(bs,w,'forward'):.3f} ms ({1000/med(bs,w,'forward'):.0f} tok/s)")
e0, s0 = best["w0"]; e3, s3 = best["w3"]
print(f"   vs best single config without huge pages ({s0:.3f} ms):")
print(f"     huge pages only            x{s0/s3:.3f}")
print(f"     stage-wise configs only    x{s0/e0:.3f}  (estimate)")
print(f"     both                       x{s0/e3:.3f}  (estimate)")

print("\n4) same core type, more threads (forward w0, tok/s):  "
      + "P1->P2->P4_HT " + " -> ".join(f"{med(n,'w0','tok'):.0f}" for n in ["P1", "P2", "P4_HT"] if n in names)
      + "   E1->E2->E4/X4_2_2 " + " -> ".join(f"{med(n,'w0','tok'):.0f}" for n in ["E1", "E2", "E4", "X4_2_2"] if n in names))
PY
cat "$OUT/M_summary.txt"
echo "end: $(date -Is) $(cpu_state)" >> "$OUT/environment.txt"

tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
