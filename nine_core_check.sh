#!/usr/bin/env bash
# Native-Linux check of the 9-CPU configurations found by the team's topology search
#   T9_search  (0,2,1,4,5,6,7,8,9)    : fastest fixed configuration of the 1..14-CPU search (P logical 3 + E 6)
#   T9_top1    (0,1,2,3,6,7,9,10,11)  : master src/cpuprofiler/benchmark.py check_top() #1 (P4 + E5)
#   T9_top2    (1,2,3,5,6,7,8,10,11)  : master check_top() #2 (P3 + E6)
#   references P4_HT (0-3) and PE12 (0-11)
# Each with two binaries: this branch's run.c (-w 5 -a 1) and master's run/run.c (-w 1 -a 1 -g 144),
# x wait policy {default, ACTIVE} x schedule {static, dynamic,16, guided,16}, plus the original code path
# (this branch, -w 0 -a 0) on P4_HT as the baseline. CPUs are placed in the listed order (OMP_PLACES),
# as the team's BenchmarkRunner does. Power profile "performance", restored at the end. No clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/nine_core_check.sh -o nine_core_check.sh && bash nine_core_check.sh
# options (env): ROUNDS=10 REPEATS=2 MODEL=path DIR=work dir LOCAL_SRC=checkout dir FORCE=1
set -uo pipefail
export LC_ALL=C
BASE="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada"
TEAM="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/master"
DIR="${DIR:-$HOME/llama_nine_check}"
ROUNDS="${ROUNDS:-10}"
REPEATS="${REPEATS:-2}"
mkdir -p "$DIR/team/run" "$DIR/team/src"
cd "$DIR"
OUT="$DIR/results_$(hostname)_nine_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
RAW="$OUT/raw.txt"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND

CASES=(
  "T9_search|0,2,1,4,5,6,7,8,9"
  "T9_top1|0,1,2,3,6,7,9,10,11"
  "T9_top2|1,2,3,5,6,7,8,10,11"
  "P4_HT|0,1,2,3"
  "PE12|0,1,2,3,4,5,6,7,8,9,10,11"
)
BINS=(ours team)
POLICIES=(default ACTIVE)
SCHEDS=(static dynamic,16 guided,16)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { # fetch URLBASE path dest; LOCAL_SRC=dir copies this branch's files from a checkout instead
  if [[ -n "${LOCAL_SRC:-}" && "$1" == "$BASE" ]]; then cp "$LOCAL_SRC/$2" "$3"; return; fi
  curl -fsSL "$1/$2" -o "$3" 2>/dev/null || wget -q "$1/$2" -O "$3"
}

ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2
  exit 1
fi
for tool in gcc taskset python3 sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
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

log "downloading sources (this branch + master)"
fetch "$BASE" run.c run.c && fetch "$BASE" tokenizer.bin tokenizer.bin &&
  fetch "$TEAM" run/run.c team/run/run.c && fetch "$TEAM" src/fused.h team/src/fused.h &&
  fetch "$TEAM" src/classifier.h team/src/classifier.h || { log "download failed"; exit 1; }
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_omp_check/stories15M.bin" \
           "$HOME/llama_full_check/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
    [[ -f "$c" && $(stat -c %s "$c") == 60816028 ]] && { MODEL="$c"; break; }
  done
fi
if [[ -z "${MODEL:-}" ]]; then
  log "downloading model (60MB)"
  fetch "$BASE" build/stories15M.bin stories15M.bin || { log "model download failed"; exit 1; }
  MODEL="$DIR/stories15M.bin"
fi
log "model: $MODEL"
log "building"
gcc -Ofast -fopenmp -march=native run.c -lm -o run-ours &&
  (cd team && gcc -I. -Ofast -fopenmp -march=native run/run.c -lm -o ../run-team) || { log "build failed"; exit 1; }
{
  date -Is; hostname; uname -a; head -2 /etc/os-release
  echo "ac_online=$ac rounds=$ROUNDS repeats=$REPEATS model=$MODEL original_profile=${orig:-n/a}"
  echo "profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" \
       "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"
  echo "ours: run.c sha256 $(sha256sum run.c | cut -c1-16)  team: run/run.c sha256 $(sha256sum team/run/run.c | cut -c1-16)"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ; echo; printf '%s\n' "${CASES[@]}"
} > "$OUT/environment.txt"

# run_case section repeat round name cpus bin policy sched -> one RAW line
run_case() {
  local sec="$1" rep="$2" r="$3" name="$4" cpus="$5" bin="$6" pol="$7" sch="$8" n places sha tok args
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  places=$(tr ',' '\n' <<< "$cpus" | sed 's/.*/{&}/' | paste -sd,)
  case "$bin" in
    ours) args=(./run-ours "$MODEL" -z tokenizer.bin -w 5 -a 1) ;;
    base) args=(./run-ours "$MODEL" -z tokenizer.bin -w 0 -a 0) ;;
    team) args=(./run-team "$MODEL" -z tokenizer.bin -w 1 -a 1 -g 144) ;;
  esac
  local envs=(OMP_NUM_THREADS="$n" OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_SCHEDULE="$sch")
  [[ "$pol" != default ]] && envs+=(OMP_WAIT_POLICY="$pol")
  sha=$(env "${envs[@]}" taskset -c "$cpus" "${args[@]}" -n 256 -t 0 -s 42 -i "$PROMPT" 2>"$OUT/last.log" | sha256sum | cut -c1-16)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  awk -v t="${tok:-0}" 'BEGIN{exit !(t > 0)}' || tok=NA   # a wall-clock jump can make it negative
  echo "RAW $sec $rep $r $name $n $cpus $bin $pol $sch ${tok:-NA} $sha" >> "$RAW"
}
cat "$MODEL" > /dev/null

log "warm-up"
for spec in "${CASES[@]}"; do IFS='|' read -r nm cp <<< "$spec"; run_case WARM 0 0 "$nm" "$cp" ours default static; done
sed -i '/^RAW WARM /d' "$RAW"
for ((rep = 1; rep <= REPEATS; rep++)); do
  log "repeat $rep/$REPEATS: $(( ${#CASES[@]} * ${#BINS[@]} * ${#POLICIES[@]} * ${#SCHEDS[@]} + 1 )) combinations x $ROUNDS rounds"
  for ((r = 1; r <= ROUNDS; r++)); do
    for spec in $( { for s in "${CASES[@]}"; do for b in "${BINS[@]}"; do for p in "${POLICIES[@]}"; do for sc in "${SCHEDS[@]}"; do
                       echo "$s|$b|$p|$sc"; done; done; done; done; echo "P4_HT|0,1,2,3|base|default|static"; } | shuf); do
      IFS='|' read -r nm cp b p sc <<< "$spec"
      run_case M "$rep" "$r" "$nm" "$cp" "$b" "$p" "$sc"
    done
    log "  repeat $rep round $r/$ROUNDS done"
  done
done
rm -f "$OUT/last.log"
echo "end: $(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"

python3 - "$RAW" > "$OUT/summary.txt" <<'PY'
import sys, statistics as st
from collections import defaultdict
v = defaultdict(list); rv = defaultdict(list); shas = set(); bad = 0
for l in open(sys.argv[1]):
    f = l.split()
    if len(f) != 12 or f[0] != "RAW": continue
    _, sec, rep, r, name, n, cpus, b, pol, sch, tok, sha = f
    shas.add(sha)
    if tok == "NA": bad += 1; continue
    v[(name, b, pol, sch)].append(float(tok)); rv[(name, b, pol, sch, rep)].append(float(tok))
med = lambda k: st.median(v[k]) if v.get(k) else float("nan")
names = ["T9_search", "T9_top1", "T9_top2", "P4_HT", "PE12"]
scheds = ["static", "dynamic,16", "guided,16"]
print(f"outputs: {len(shas)} distinct " + ("(OK, all identical)" if len(shas) == 1 else "(check!)") + f"   invalid tok/s: {bad}")
base = med(("P4_HT", "base", "default", "static"))
print(f"baseline: this branch -w 0 -a 0 (original code path), P4_HT, default, static = {base:.0f} tok/s\n")
for b, label in (("ours", "this branch run.c (-w 5 -a 1)"), ("team", "master run/run.c (-w 1 -a 1 -g 144)")):
    print(f"== {label}: tok/s median ==")
    print(f"  {'config':<10}{'policy':<9}" + "".join(f"{s:>12}" for s in scheds))
    for nm in names:
        for pol in ("default", "ACTIVE"):
            print(f"  {nm:<10}{pol:<9}" + "".join(f"{med((nm, b, pol, s)):>12.0f}" for s in scheds))
    print()
ranked = sorted(((med(k), k) for k in v if k[1] != "base"), reverse=True)
print("== top 10 (tok/s median, x baseline) ==")
for t, k in ranked[:10]:
    print(f"  {t:7.0f}  x{t / base:.2f}  {k[0]:<10} {k[1]:<5} {k[2]:<8} {k[3]}")
reps = sorted({k[4] for k in rv})
if len(reps) > 1:
    print("\n== reproducibility of the top 5 (median tok/s per repeat) ==")
    for t, k in ranked[:5]:
        print(f"  {k[0]:<10} {k[1]:<5} {k[2]:<8} {k[3]:<11}" + "  ".join(f"rep{rp}: {st.median(rv[k + (rp,)]):.0f}" for rp in reps if rv.get(k + (rp,))))
PY
cat "$OUT/summary.txt"
cp run.c "$OUT/run_ours.c"; cp team/run/run.c "$OUT/run_team.c"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
