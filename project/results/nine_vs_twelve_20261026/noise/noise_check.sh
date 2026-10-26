#!/usr/bin/env bash
# Does other activity on the PC explain why the team measured "0-11 is 9% slower than 9 CPUs" while
# gap_check.sh found them within 1%?  With 12 ACTIVE threads on CPUs 0-11 no CPU of the compute tile is idle,
# so anything else that wakes up (the benchmark runner reading stdout, the desktop, an agent) has to share a CPU
# with an OpenMP thread and the whole team waits for it at the next barrier. 9 CPUs (0,2,4-10) leave 1,3,11 idle.
# Conditions, each with the same 4 candidates in every block (random order):
#   file   stdout to a file (no reader wakes up while generating)
#   sha    stdout piped to sha256sum (gap_check.sh / rigorous_check.sh)
#   py     Python subprocess.run(capture_output=True), like the team's benchmark scripts (unpinned parent)
#   bg10   sha + one unpinned background process busy 10% of the time (1 ms every 10 ms)
#   bg50   sha + one unpinned background process busy 50% of the time (5 ms every 10 ms)
# master 256b0b24, ACTIVE, -w 1 -a 1 -g 288. About 15 min with the defaults.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/noise_check.sh -o noise_check.sh && bash noise_check.sh
# options (env): BLOCKS=25 INITIAL_REST=60 COOL_MARGIN=8 REST_MIN=8 COOL_MAX=30 NOTE=text MODEL=path DIR=work dir FORCE=1
set -uo pipefail
export LC_ALL=C
RAWGH="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c"
C_MASTER=256b0b2445ecd2c8c88d085fdf616313d5df91f3
C_SCRIPT=dev/sakurada
DIR="${DIR:-$HOME/llama_noise}"
BLOCKS="${BLOCKS:-25}"
INITIAL_REST="${INITIAL_REST:-60}"
COOL_MARGIN="${COOL_MARGIN:-8}"
REST_MIN="${REST_MIN:-8}"
COOL_MAX="${COOL_MAX:-30}"
mkdir -p "$DIR/src/team/run" "$DIR/src/team/src"
cd "$DIR"
OUT="$DIR/results_$(hostname)_noise_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
RAW="$OUT/raw.tsv"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND OMP_NUM_THREADS GOMP_SPINCOUNT PERSISTENT_OPENMP OUTPUT_THREADS

N9=0,2,4,5,6,7,8,9,10
PE12=0,1,2,3,4,5,6,7,8,9,10,11
LP12=0,2,4,5,6,7,8,9,10,11,12,13
# id | cpus (placement order) | schedule | note
CANDIDATES=(
  "N9_gd4|$N9|guided,4|team 9 CPUs, team setting"
  "PE12_gd4|$PE12|guided,4|0-11, team setting (master-threads doc)"
  "PE12_static|$PE12|static|0-11, static"
  "LP12_gd4|$LP12|guided,4|12 physical cores incl. LP-E (leaves 1,3 idle)"
)
CONDITIONS=(file sha py bg10 bg50)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$RAWGH/$1/$2" -o "$3" 2>/dev/null || wget -q "$RAWGH/$1/$2" -O "$3"; }

ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2; exit 1
fi
for tool in gcc taskset python3 sha256sum shuf; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
BGPID=""
orig="$(powerprofilesctl get 2>/dev/null || true)"
cleanup() {
  [[ -n "$BGPID" ]] && kill "$BGPID" 2>/dev/null
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

log "downloading pinned sources"
fetch "$C_MASTER" run/run.c src/team/run/run.c &&
  fetch "$C_MASTER" src/fused.h src/team/src/fused.h &&
  fetch "$C_MASTER" src/classifier.h src/team/src/classifier.h &&
  fetch "$C_MASTER" tokenizer.bin tokenizer.bin || { log "download failed"; exit 1; }
fetch "$C_SCRIPT" noise_check.sh "$OUT/noise_check.sh" 2>/dev/null || cp "$0" "$OUT/noise_check.sh" 2>/dev/null
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_gap/stories15M.bin" "$HOME/llama_rigorous/stories15M.bin" \
           "$HOME/llama_nine_check/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
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
(cd src/team && gcc -I. -Ofast -fopenmp -march=native run/run.c -lm -o ../../bin-team) || { log "build failed"; exit 1; }

# the team's way of running: python subprocess.run(capture_output=True, text=True), parent not pinned
cat > "$OUT/pyrun.py" <<'PY'
import hashlib, os, subprocess, sys
i = sys.argv.index("--")
env = dict(os.environ, **dict(a.split("=", 1) for a in sys.argv[2:i]))
p = subprocess.run(sys.argv[i + 1:], env=env, capture_output=True, text=True, timeout=300)
open(sys.argv[1], "w").write(p.stderr)
print(hashlib.sha256(p.stdout.encode()).hexdigest()[:16])
PY
# unpinned background load: busy FRACTION of every 10 ms
cat > "$OUT/bg.py" <<'PY'
import sys, time
busy = float(sys.argv[1]) * 0.010
while True:
    t = time.perf_counter()
    while time.perf_counter() - t < busy:
        pass
    time.sleep(0.010 - busy)
PY

ZONES=(/sys/class/thermal/thermal_zone*/temp)
max_temp() { local m=0 t z; for z in "${ZONES[@]}"; do read -r t < "$z" 2>/dev/null || t=0; (( t > m )) && m=$t; done; echo $((m / 1000)); }
mean_freq() { local s=0 n=0 c f; for c in ${1//,/ }; do read -r f < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || f=0; s=$((s + f)); n=$((n + 1)); done; echo $((s / n / 1000)); }
throttle() { local s=0 f v; for f in /sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count; do read -r v < "$f" 2>/dev/null || v=0; s=$((s + v)); done; echo $s; }

{
  echo "start=$(date -Is) host=$(hostname) note=${NOTE:-} blocks=$BLOCKS"
  echo "initial_rest=$INITIAL_REST cool_margin=$COOL_MARGIN rest_min=$REST_MIN cool_max=$COOL_MAX"
  echo "ac_online=$ac original_profile=${orig:-n/a} profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)" \
       "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  uname -a; head -2 /etc/os-release; gcc --version | head -1; python3 --version
  echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -c1-16)"
  for f in src/team/run/run.c src/team/src/fused.h src/team/src/classifier.h tokenizer.bin; do echo "source $f sha256=$(sha256sum "$f" | cut -c1-16)"; done
  echo "commit: master=$C_MASTER"
  echo; printf 'cand %s\n' "${CANDIDATES[@]}"
  echo "conditions: ${CONDITIONS[*]}"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; echo "cache sharing (shared_cpu_list of the last level cache per CPU):"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    idx=$(ls -d "$c"/cache/index* 2>/dev/null | tail -1); [[ -n "$idx" ]] && echo "  $(basename "$c"): $(cat "$idx/shared_cpu_list")"
  done
  echo; echo "other processes using CPU:"; ps -eo pcpu,comm --sort=-pcpu | head -8
} > "$OUT/environment.txt"

printf 'block\torder\tid\tcond\tcpus\tschedule\ttok_s\tsha\ttemp_before\ttemp_after\tfreq_before\tthrottle_delta\tclock\n' > "$RAW"

run_one() { # block order spec cond
  local blk="$1" ord="$2" spec="$3" cond="$4" id cpus sch note n places sha tok tb ta fb th0 th1
  IFS='|' read -r id cpus sch note <<< "$spec"
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  places=$(tr ',' '\n' <<< "$cpus" | sed 's/.*/{&}/' | paste -sd,)
  local envs=(OMP_NUM_THREADS="$n" OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_SCHEDULE="$sch" OMP_WAIT_POLICY=ACTIVE)
  local cmd=(taskset -c "$cpus" ./bin-team "$MODEL" -z tokenizer.bin -w 1 -a 1 -g 288 -n 256 -t 0 -s 42 -i "$PROMPT")
  case "$cond" in
    bg10) python3 "$OUT/bg.py" 0.1 & BGPID=$!; sleep 0.3 ;;
    bg50) python3 "$OUT/bg.py" 0.5 & BGPID=$!; sleep 0.3 ;;
  esac
  tb=$(max_temp); fb=$(mean_freq "$cpus"); th0=$(throttle)
  case "$cond" in
    file) env "${envs[@]}" "${cmd[@]}" > "$OUT/out.txt" 2> "$OUT/last.log"; sha=$(sha256sum < "$OUT/out.txt" | cut -c1-16) ;;
    py)   sha=$(python3 "$OUT/pyrun.py" "$OUT/last.log" "${envs[@]}" -- "${cmd[@]}") ;;
    *)    sha=$(env "${envs[@]}" "${cmd[@]}" 2> "$OUT/last.log" | sha256sum | cut -c1-16) ;;
  esac
  ta=$(max_temp); th1=$(throttle)
  if [[ -n "$BGPID" ]]; then kill "$BGPID" 2>/dev/null; wait "$BGPID" 2>/dev/null; BGPID=""; fi
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  awk -v t="${tok:-0}" 'BEGIN{exit !(t > 0)}' || tok=NA
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$blk" "$ord" "$id" "$cond" "$cpus" "$sch" "${tok}" "$sha" \
    "$tb" "$ta" "$fb" "$((th1 - th0))" "$(date +%s.%N)" >> "$RAW"
}

wait_cool() {
  local start below=0 t
  start=$(date +%s)
  sleep "$REST_MIN"
  while :; do
    t=$(max_temp)
    if (( t <= COOL_TEMP )); then below=$((below + 1)); else below=0; fi
    (( below >= 2 )) && break
    (( $(date +%s) - start >= COOL_MAX )) && break
    sleep 2
  done
  WAITED=$(( $(date +%s) - start ))
}

combos() { local s c; for s in "${CANDIDATES[@]}"; do for c in "${CONDITIONS[@]}"; do echo "$c|$s"; done; done; }

cat "$MODEL" > /dev/null
log "warm-up: every combination once (not recorded)"
while IFS= read -r x; do run_one 0 0 "${x#*|}" "${x%%|*}"; done < <(combos)
if [[ $(awk -F'\t' 'NR > 1 {print $8}' "$RAW" | sort -u | wc -l) != 1 ]]; then
  log "ERROR: different outputs in the warm-up:"; awk -F'\t' 'NR > 1 {print $3, $4, $7, $8}' "$RAW" | tee -a "$OUT/progress.log"; exit 1
fi
sed -i '/^0\t/d' "$RAW"
log "resting ${INITIAL_REST}s to measure the idle temperature (do not use the PC from now on)"
sleep "$INITIAL_REST"
s=0; for i in 1 2 3 4 5; do s=$((s + $(max_temp))); sleep 1; done
IDLE_TEMP=$((s / 5))
COOL_TEMP="${COOL_TEMP:-$((IDLE_TEMP + COOL_MARGIN))}"
echo "idle_temp=$IDLE_TEMP cool_temp=$COOL_TEMP" >> "$OUT/environment.txt"
log "idle ${IDLE_TEMP} C; ${#CANDIDATES[@]} candidates x ${#CONDITIONS[@]} conditions x $BLOCKS blocks, each block after resting to <= ${COOL_TEMP} C"
T0=$(date +%s)
for ((b = 1; b <= BLOCKS; b++)); do
  wait_cool
  ord=0
  while IFS= read -r x; do
    ord=$((ord + 1)); run_one "$b" "$ord" "${x#*|}" "${x%%|*}"
  done < <(combos | shuf)
  el=$(( $(date +%s) - T0 ))
  log "  block $b/$BLOCKS (waited ${WAITED}s, now $(max_temp) C)  elapsed $((el / 60)) min, rough ETA $(( el * (BLOCKS - b) / b / 60 )) min"
done
rm -f "$OUT/last.log" "$OUT/out.txt"
echo "end=$(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"

cat > "$OUT/analyze_noise.py" <<'PY'
#!/usr/bin/env python3
"""usage: python3 analyze_noise.py RESULT_DIR [RESULT_DIR ...]"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows = []
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
by = defaultdict(dict)  # (id, cond) -> (session, block) -> tok/s
for r in ok:
    by[(r["id"], r["cond"])][(r["session"], int(r["block"]))] = float(r["tok_s"])
IDS = ["N9_gd4", "PE12_gd4", "PE12_static", "LP12_gd4"]
CONDS = ["file", "sha", "py", "bg10", "bg50"]
rng = random.Random(42)


def paired(a, b, n=3000):
    A, B = by.get(a, {}), by.get(b, {})
    rs = [A[k] / B[k] for k in A if k in B]
    if not rs:
        return None
    m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), m[int(0.025 * n)], m[int(0.975 * n)], sum(x > 1 for x in rs), len(rs)


def fp(p):
    if p is None:
        return f"{'-':>26}"
    r, lo, hi, w, n = p
    return f"{r:6.3f}{'*' if lo > 1 or hi < 1 else ' '}[{lo:.3f},{hi:.3f}] {w:>2}/{n:<2}"


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows:
    shas[r["sha"]].add((r["id"], r["cond"]))
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
tb = [int(r["temp_before"]) for r in rows if r["temp_before"].isdigit()]
th = sum(int(r["throttle_delta"]) for r in rows if r["throttle_delta"].lstrip("-").isdigit())
print(f"temperature before runs min/median/max {min(tb)}/{st.median(tb):.0f}/{max(tb)} C   throttle events {th}")

for label, f in (("median tok/s", st.median), ("min tok/s", min), ("CV %", lambda v: st.pstdev(v) / st.mean(v) * 100)):
    print(f"\n== {label} ==")
    print(f"  {'':<14}" + "".join(f"{c:>9}" for c in CONDS))
    for i in IDS:
        print(f"  {i:<14}" + "".join(f"{f(list(by[(i, c)].values())):>9.{1 if label.startswith('CV') else 0}f}" if by.get((i, c)) else f"{'-':>9}" for c in CONDS))

print("\n== each condition relative to 'file' (paired within block, median [95% CI] faster-blocks) ==")
for i in IDS:
    print(f"  {i:<14}" + "".join(f"{c}: {fp(paired((i, c), (i, 'file')))}  " for c in CONDS[1:]))

print("\n== 9 CPUs vs the others under each condition: ratio N9_gd4 / other (team saw 1.10 for 0-11, 1.12 for LP-E) ==")
print(f"  {'':<18}" + "".join(f"{c:^28}" for c in CONDS))
for o in ("PE12_gd4", "PE12_static", "LP12_gd4"):
    print(f"  N9 / {o:<13}" + "".join(fp(paired(("N9_gd4", c), (o, c))) + "  " for c in CONDS))
PY
python3 "$OUT/analyze_noise.py" "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
