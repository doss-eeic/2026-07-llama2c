#!/usr/bin/env bash
# Rigorous final comparison of the fastest known llama2.c configurations on this machine.
#   - sources pinned to commits (and their sha256 recorded):
#       upstream : master 256b0b24 run/run_upstream.c (= untouched karpathy 350e04fe)
#       team     : master 256b0b24 run/run.c (QKV/FFN fusion, -w 1 hugepage, -a 1 -g 144 exact int8 argmax)
#       ours     : dev/sakurada a49a36f7 run.c (-w 5 fused + hugepage, -a 1 -g 144)
#       omptmp   : dev/omp-tmp 3bd572ed run.c (fp32, weighted P/E output split)
#   - randomized complete blocks: every candidate once per block, new random order each block (BLOCKS blocks)
#   - before/after every run: max temperature, mean frequency of the used CPUs, thermal throttle counters
#   - analysis: medians with block-bootstrap 95% CI, paired comparison against the fastest, drift over blocks
#   - run it twice (e.g. after a 10-15 min break) and analyze both sessions together for reproducibility
# Power profile "performance" for the whole run, restored at the end. No clone, no sudo.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/rigorous_check.sh -o rigorous_check.sh && bash rigorous_check.sh
# options (env): BLOCKS=30 COOLDOWN=0 (seconds between blocks) MODEL=path DIR=work dir FORCE=1
set -uo pipefail
export LC_ALL=C
RAWGH="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c"
C_MASTER=256b0b2445ecd2c8c88d085fdf616313d5df91f3
C_OURS=a49a36f7181e906026e533c47e001fd7c77a37a1
C_OMPTMP=3bd572ed59901fda549601953b4d8ab7361e25e0
C_SCRIPT=dev/sakurada
DIR="${DIR:-$HOME/llama_rigorous}"
BLOCKS="${BLOCKS:-30}"
COOLDOWN="${COOLDOWN:-0}"
mkdir -p "$DIR/src/upstream" "$DIR/src/team/run" "$DIR/src/team/src" "$DIR/src/ours" "$DIR/src/omptmp"
cd "$DIR"
OUT="$DIR/results_$(hostname)_rigorous_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
RAW="$OUT/raw.tsv"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND OMP_NUM_THREADS GOMP_SPINCOUNT \
      LLAMA_OUT_THREADS LLAMA_OUT_P LLAMA_OUT_WP

# id | binary | cpus (in placement order) | wait policy | schedule | note
CANDIDATES=(
  "C01_up_P4|upstream|0,1,2,3|default|static|original code, P-cores (baseline)"
  "C02_up_P4_act|upstream|0,1,2,3|ACTIVE|static|original code + ACTIVE"
  "C03_team_T9b|team|1,2,3,5,6,7,8,10,11|ACTIVE|guided,16|team best 9 CPUs (check_top #2, omp-tmp 1575 tok/s)"
  "C04_team_T9c|team|0,2,3,5,6,7,8,10,11|ACTIVE|guided,16|omp-tmp extra 9 CPUs (1560 tok/s)"
  "C05_team_T9a|team|0,1,2,3,6,7,9,10,11|ACTIVE|guided,16|check_top #1 placement"
  "C06_team_T9b_def|team|1,2,3,5,6,7,8,10,11|default|guided,16|team best 9 CPUs without ACTIVE"
  "C07_team_PE12|team|0,1,2,3,4,5,6,7,8,9,10,11|ACTIVE|static|all P+E threads"
  "C08_team_PE12_g|team|0,1,2,3,4,5,6,7,8,9,10,11|ACTIVE|guided,16|all P+E threads, guided"
  "C09_team_PE10|team|0,2,4,5,6,7,8,9,10,11|ACTIVE|static|P-cores 1 thread each + all E"
  "C10_team_P4|team|0,1,2,3|ACTIVE|guided,16|P-cores only"
  "C11_ours_PE12|ours|0,1,2,3,4,5,6,7,8,9,10,11|ACTIVE|static|sakurada -w 5 -a 1, all P+E"
  "C12_ours_T9b|ours|1,2,3,5,6,7,8,10,11|ACTIVE|static|sakurada -w 5 -a 1, team best 9 CPUs"
  "C13_omptmp_PE10|omptmp|0,2,4,5,6,7,8,9,10,11|ACTIVE|static|omp-tmp fp32 weighted output (P x2)"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$RAWGH/$1/$2" -o "$3" 2>/dev/null || wget -q "$RAWGH/$1/$2" -O "$3"; }

# ---------------------------------------------------------------- checks
ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2; exit 1
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
  echo "Could not switch the power profile to performance (powerprofilesctl). Run with FORCE=1 to measure anyway." >&2; exit 1
fi

# ---------------------------------------------------------------- sources (pinned) and build
log "downloading pinned sources"
fetch "$C_MASTER" run/run_upstream.c src/upstream/run.c &&
  fetch "$C_MASTER" run/run.c src/team/run/run.c &&
  fetch "$C_MASTER" src/fused.h src/team/src/fused.h &&
  fetch "$C_MASTER" src/classifier.h src/team/src/classifier.h &&
  fetch "$C_OURS" run.c src/ours/run.c &&
  fetch "$C_OMPTMP" run.c src/omptmp/run.c &&
  fetch "$C_MASTER" tokenizer.bin tokenizer.bin || { log "download failed"; exit 1; }
fetch "$C_SCRIPT" rigorous_check.sh "$OUT/rigorous_check.sh" 2>/dev/null || cp "$0" "$OUT/rigorous_check.sh" 2>/dev/null
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_nine_check/stories15M.bin" \
           "$HOME/llama_omp_check/stories15M.bin" "$HOME/native_kit/stories15M.bin" "$DIR/stories15M.bin"; do
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
gcc "${CFLAGS[@]}" src/upstream/run.c -lm -o bin-upstream &&
  (cd src/team && gcc -I. "${CFLAGS[@]}" run/run.c -lm -o ../../bin-team) &&
  gcc "${CFLAGS[@]}" src/ours/run.c -lm -o bin-ours &&
  gcc "${CFLAGS[@]}" src/omptmp/run.c -lm -o bin-omptmp || { log "build failed"; exit 1; }

# ---------------------------------------------------------------- environment record
{
  echo "start=$(date -Is) host=$(hostname) blocks=$BLOCKS cooldown=$COOLDOWN"
  echo "ac_online=$ac original_profile=${orig:-n/a} profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)" \
       "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
       "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  uname -a; head -2 /etc/os-release; gcc --version | head -1
  echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -c1-16)"
  for f in src/upstream/run.c src/team/run/run.c src/team/src/fused.h src/team/src/classifier.h src/ours/run.c src/omptmp/run.c tokenizer.bin; do
    echo "source $f sha256=$(sha256sum "$f" | cut -c1-16)"
  done
  echo "commits: master=$C_MASTER ours=$C_OURS omptmp=$C_OMPTMP"
  echo; printf '%s\n' "${CANDIDATES[@]}"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
  echo; uptime; free -m
} > "$OUT/environment.txt"

# ---------------------------------------------------------------- monitoring helpers
max_temp() { local m=0 t; for z in /sys/class/thermal/thermal_zone*/temp; do t=$(cat "$z" 2>/dev/null || echo 0); (( t > m )) && m=$t; done; echo $((m / 1000)); }
mean_freq() { # mean current MHz of the given cpus
  local s=0 n=0 c f
  for c in ${1//,/ }; do f=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || echo 0); s=$((s + f)); n=$((n + 1)); done
  echo $((s / n / 1000))
}
throttle() { local s=0 f; for f in /sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count; do s=$((s + $(cat "$f" 2>/dev/null || echo 0))); done; echo $s; }

printf 'block\torder\tid\tbinary\tcpus\tpolicy\tschedule\ttok_s\tsha\ttemp_before\ttemp_after\tfreq_before\tfreq_after\tthrottle_delta\tclock\n' > "$RAW"

run_one() { # block order spec
  local blk="$1" ord="$2" spec="$3" id bin cpus pol sch note n places sha tok tb ta fb fa th0 th1
  IFS='|' read -r id bin cpus pol sch note <<< "$spec"
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  places=$(tr ',' '\n' <<< "$cpus" | sed 's/.*/{&}/' | paste -sd,)
  local envs=(OMP_NUM_THREADS="$n" OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_SCHEDULE="$sch")
  [[ "$pol" != default ]] && envs+=(OMP_WAIT_POLICY="$pol")
  local args
  case "$bin" in
    upstream) args=(./bin-upstream "$MODEL" -z tokenizer.bin) ;;
    team)     args=(./bin-team "$MODEL" -z tokenizer.bin -w 1 -a 1 -g 144) ;;
    ours)     args=(./bin-ours "$MODEL" -z tokenizer.bin -w 5 -a 1 -g 144) ;;
    omptmp)   args=(./bin-omptmp "$MODEL" -z tokenizer.bin); envs+=(LLAMA_OUT_THREADS="$n" LLAMA_OUT_P=2 LLAMA_OUT_WP=2) ;;
  esac
  tb=$(max_temp); fb=$(mean_freq "$cpus"); th0=$(throttle)
  sha=$(env "${envs[@]}" taskset -c "$cpus" "${args[@]}" -n 256 -t 0 -s 42 -i "$PROMPT" 2>"$OUT/last.log" | sha256sum | cut -c1-16)
  ta=$(max_temp); fa=$(mean_freq "$cpus"); th1=$(throttle)
  tok=$(sed -n 's/achieved tok\/s: //p' "$OUT/last.log")
  awk -v t="${tok:-0}" 'BEGIN{exit !(t > 0)}' || tok=NA
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$blk" "$ord" "$id" "$bin" "$cpus" "$pol" "$sch" \
    "$tok" "$sha" "$tb" "$ta" "$fb" "$fa" "$((th1 - th0))" "$(date +%s.%N)" >> "$RAW"
}

cat "$MODEL" > /dev/null
log "warm-up: every candidate once (not recorded)"
for spec in "${CANDIDATES[@]}"; do run_one 0 0 "$spec"; done
sed -i '/^0\t/d' "$RAW"
log "measuring: ${#CANDIDATES[@]} candidates x $BLOCKS blocks (random order inside each block)"
for ((b = 1; b <= BLOCKS; b++)); do
  ord=0
  while IFS= read -r spec; do
    ord=$((ord + 1)); run_one "$b" "$ord" "$spec"
  done < <(printf '%s\n' "${CANDIDATES[@]}" | shuf)
  log "  block $b/$BLOCKS done (max temp $(max_temp) C)"
  (( COOLDOWN > 0 )) && sleep "$COOLDOWN"
done
rm -f "$OUT/last.log"
echo "end=$(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"

# ---------------------------------------------------------------- analysis (also saved for combining sessions)
cat > "$OUT/analyze_rigorous.py" <<'PY'
#!/usr/bin/env python3
"""usage: python3 analyze_rigorous.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)"""
import csv, os, random, statistics as st, sys
from collections import defaultdict

rows = []
for s, d in enumerate(sys.argv[1:], 1):
    for r in csv.DictReader(open(os.path.join(d, "raw.tsv")), delimiter="\t"):
        r["session"] = s
        rows.append(r)
ok = [r for r in rows if r["tok_s"] not in ("NA", "")]
for r in ok:
    r["tok"] = float(r["tok_s"])
ids = sorted({r["id"] for r in ok})
note = {}
for d in sys.argv[1:]:
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("C") and "|" in l:
            f = l.strip().split("|"); note[f[0]] = f"{f[1]} {f[3]} {f[4]} cpus={f[2]}"
by = defaultdict(dict)  # id -> (session, block) -> tok
for r in ok:
    by[r["id"]][(r["session"], r["block"])] = r["tok"]
blocks = sorted({k for v in by.values() for k in v})
rng = random.Random(42)

def boot_median(id_, n=4000):
    keys = list(by[id_]); vals = [by[id_][k] for k in keys]
    meds = sorted(st.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]

def paired(a, b, n=4000):
    keys = [k for k in by[a] if k in by[b]]
    rs = [by[a][k] / by[b][k] for k in keys]
    if not rs: return float("nan"), (float("nan"), float("nan")), 0, 0
    meds = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), (meds[int(0.025 * n)], meds[int(0.975 * n)]), sum(x > 1 for x in rs), len(rs)

print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   invalid tok/s: {len(rows) - len(ok)}")
shas = defaultdict(set)
for r in rows: shas[r["sha"]].add(r["id"])
print(f"distinct outputs: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK: " + "; ".join(f"{k}:{sorted(v)}" for k, v in shas.items()) + ")"))
thr = sum(int(r["throttle_delta"]) for r in rows if r["throttle_delta"].lstrip("-").isdigit())
temps = [int(r["temp_after"]) for r in rows if r["temp_after"].isdigit()]
print(f"thermal throttle events during runs: {thr}   max temperature seen: {max(temps) if temps else 'n/a'} C")

med = {i: st.median(by[i].values()) for i in ids}
order = sorted(ids, key=lambda i: -med[i])
best = order[0]
base = "C01_up_P4" if "C01_up_P4" in by else order[-1]
print(f"\n== ranking (median tok/s, block-bootstrap 95% CI, x baseline {base}) ==")
print(f"  {'id':<18}{'n':>4}{'median':>9}{'95% CI':>18}{'CV%':>6}{'x base':>8}   setting")
for i in order:
    v = list(by[i].values()); lo, hi = boot_median(i)
    cv = st.pstdev(v) / st.mean(v) * 100
    print(f"  {i:<18}{len(v):>4}{med[i]:>9.0f}{f'[{lo:.0f}, {hi:.0f}]':>18}{cv:>6.1f}{med[i] / med[base]:>8.2f}   {note.get(i, '')}")

print(f"\n== paired comparison with the fastest ({best}): ratio of other / best within the same block ==")
for i in order[1:]:
    r, (lo, hi), w, n = paired(i, best)
    verdict = "slower (CI below 1)" if hi < 1 else ("faster?" if lo > 1 else "not distinguishable")
    print(f"  {i:<18} {r:.3f}  CI [{lo:.3f}, {hi:.3f}]  faster than best in {w}/{n} blocks  -> {verdict}")

pairs = [("C03_team_T9b", "C07_team_PE12", "team 9 CPUs vs all 12"), ("C11_ours_PE12", "C07_team_PE12", "ours vs team (PE12)"),
         ("C12_ours_T9b", "C03_team_T9b", "ours vs team (9 CPUs)"), ("C03_team_T9b", "C06_team_T9b_def", "ACTIVE effect (team 9 CPUs)"),
         ("C02_up_P4_act", "C01_up_P4", "ACTIVE effect (original code)"), ("C08_team_PE12_g", "C07_team_PE12", "guided vs static (PE12)")]
print("\n== specific questions (paired, ratio first/second) ==")
for a, b, label in pairs:
    if a in by and b in by:
        r, (lo, hi), w, n = paired(a, b)
        print(f"  {label:<32} {r:.3f}  CI [{lo:.3f}, {hi:.3f}]  first faster in {w}/{n}")

print("\n== drift: median tok/s per quarter of the blocks, relative to the candidate's own median ==")
for s in sorted({k[0] for k in blocks}):
    bl = sorted(int(k[1]) for k in blocks if k[0] == s); q = max(1, len(bl) // 4)
    quarters = [bl[i:i + q] for i in range(0, len(bl), q)][:4]
    cells = []
    for qb in quarters:
        rel = [by[i][(s, str(b))] / med[i] for i in ids for b in qb if (s, str(b)) in by[i]]
        cells.append(f"{st.median(rel):.3f}")
    print(f"  session {s}: " + "  ".join(cells))
if len(sys.argv) > 2:
    print("\n== per-session median tok/s (reproducibility) ==")
    for i in order:
        print(f"  {i:<18}" + "  ".join(f"s{s}: {st.median([v for k, v in by[i].items() if k[0] == s]):.0f}" for s in range(1, len(sys.argv))))
PY
python3 "$OUT/analyze_rigorous.py" "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
