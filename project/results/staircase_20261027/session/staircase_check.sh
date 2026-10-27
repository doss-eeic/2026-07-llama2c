#!/usr/bin/env bash
# Speed of every optimization step, measured in the same randomized blocks on a quiet PC, for the final slides.
#   main steps (S1..S8):
#     S1 original code (upstream)               P-cores 0-3, 4 threads
#     S2 + hugepage                             (dev/sakurada run.c -w 3: aligned copy + MADV_HUGEPAGE, no fusion)
#     S3 + fused QKV / W1|W3 matmuls            (master run/run.c -w 1)
#     S4 + int8 two-stage exact argmax          (master -w 1 -a 1 -g 144)
#     S5 + OMP_WAIT_POLICY=ACTIVE               (still P-cores 0-3)
#     S6 9 CPUs 0,2,4-10, guided,4, g288        (the team's setting, master-threads doc)
#     S7 kasikoi2's pick                         (9 CPUs, guided,16, ACTIVE, GOMP_SPINCOUNT=10000, g144; kasikoi2_quiet_20261026)
#     S8 12 threads 0-11, static, ACTIVE, g288   (fastest known)
#   factor checks (F_*): fusion without hugepage, sakurada's -w 5 implementation, ACTIVE / guided on 9 CPUs,
#     guided,4 on 12 threads, all 14 CPUs with default settings (kasikoi2's start), upstream on 0-11
# Sources pinned to commits: master c3130ec9 (run/run_upstream.c, run/run.c, src/fused.h, src/classifier.h) and
# dev/sakurada a49a36f7 (run.c). Every run: stdout to a file (nothing wakes up while generating), tok/s, sha256 of the
# text, involuntary context switches of llama2c, CPU time used by other processes, temperature before/after.
# Power profile "performance" for the whole run, restored at the end. No sudo. About 12-15 min.
# Close every other program first and do not touch the PC until it ends.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/staircase_check.sh -o staircase_check.sh && bash staircase_check.sh
# options (env): BLOCKS=25 INITIAL_REST=60 COOL_MARGIN=8 REST_MIN=10 COOL_MAX=45 STRICT=1 NOTE=text MODEL=path DIR=work dir FORCE=1
set -uo pipefail
export LC_ALL=C
RAWGH="https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c"
C_MASTER=c3130ec98fc434359d1b713a61a841cca5727243
C_OURS=a49a36f7181e906026e533c47e001fd7c77a37a1
C_SCRIPT=dev/sakurada
DIR="${DIR:-$HOME/llama_staircase}"
BLOCKS="${BLOCKS:-25}"
INITIAL_REST="${INITIAL_REST:-60}"
COOL_MARGIN="${COOL_MARGIN:-8}"
REST_MIN="${REST_MIN:-10}"
COOL_MAX="${COOL_MAX:-45}"
mkdir -p "$DIR/src/upstream" "$DIR/src/master/run" "$DIR/src/master/src" "$DIR/src/ours"
cd "$DIR"
OUT="$DIR/results_$(hostname)_staircase_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
PROMPT="Once upon a time"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND OMP_NUM_THREADS OMP_DYNAMIC GOMP_SPINCOUNT \
      PERSISTENT_OPENMP OUTPUT_THREADS

P4=0,1,2,3
N9=0,2,4,5,6,7,8,9,10
PE12=0,1,2,3,4,5,6,7,8,9,10,11
ALL=0,1,2,3,4,5,6,7,8,9,10,11,12,13
# id | binary | args | cpus (placement order) | wait policy | schedule | spincount | label
CANDIDATES=(
  "S1_upstream|up|-|$P4|default|static|-|original code, P-cores 4 threads"
  "S2_hugepage|ours|-w 3 -a 0|$P4|default|static|-|+ hugepage"
  "S3_fused|master|-w 1 -a 0|$P4|default|static|-|+ fused QKV and W1/W3 matmuls"
  "S4_argmax|master|-w 1 -a 1 -g 144|$P4|default|static|-|+ int8 two-stage exact argmax"
  "S5_active|master|-w 1 -a 1 -g 144|$P4|ACTIVE|static|-|+ OMP_WAIT_POLICY=ACTIVE"
  "S6_nine|master|-w 1 -a 1 -g 288|$N9|ACTIVE|guided,4|-|9 CPUs 0,2,4-10, guided,4 (team setting)"
  "S7_kasikoi2|master|-w 1 -a 1 -g 144|$N9|ACTIVE|guided,16|10000|kasikoi2 pick: 9 CPUs, guided,16, spin 10000"
  "S8_twelve|master|-w 1 -a 1 -g 288|$PE12|ACTIVE|static|-|12 threads 0-11, static"
  "F_fused_only|master|-w 0 -a 0|$P4|default|static|-|fused matmuls without hugepage"
  "F_ours_w5|ours|-w 5 -a 0|$P4|default|static|-|sakurada -w 5 (fused + hugepage)"
  "F_ours_w5a1|ours|-w 5 -a 1 -g 144|$P4|default|static|-|sakurada -w 5 -a 1"
  "F_nine_default|master|-w 1 -a 1 -g 288|$N9|default|guided,4|-|9 CPUs guided,4 without ACTIVE"
  "F_nine_static|master|-w 1 -a 1 -g 288|$N9|ACTIVE|static|-|9 CPUs static ACTIVE"
  "F_twelve_guided4|master|-w 1 -a 1 -g 288|$PE12|ACTIVE|guided,4|-|12 threads guided,4 ACTIVE"
  "F_all14_default|master|-w 1 -a 1 -g 144|$ALL|default|dynamic,16|-|all 14 CPUs, default settings (kasikoi2 start)"
  "F_upstream_twelve|up|-|$PE12|default|static|-|original code on 0-11"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fetch() { curl -fsSL "$RAWGH/$1/$2" -o "$3" 2>/dev/null || wget -q "$RAWGH/$1/$2" -O "$3"; }

ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2; exit 1
fi
for tool in gcc taskset python3 sha256sum shuf; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
orig="$(powerprofilesctl get 2>/dev/null || true)"
cleanup() { [[ -n "$orig" ]] && powerprofilesctl set "$orig" 2>/dev/null && log "power profile restored to $orig"; }
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
fetch "$C_MASTER" run/run_upstream.c src/upstream/run.c &&
  fetch "$C_MASTER" run/run.c src/master/run/run.c &&
  fetch "$C_MASTER" src/fused.h src/master/src/fused.h &&
  fetch "$C_MASTER" src/classifier.h src/master/src/classifier.h &&
  fetch "$C_OURS" run.c src/ours/run.c &&
  fetch "$C_MASTER" tokenizer.bin tokenizer.bin || { log "download failed"; exit 1; }
fetch "$C_SCRIPT" staircase_check.sh "$OUT/staircase_check.sh" 2>/dev/null || cp "$0" "$OUT/staircase_check.sh" 2>/dev/null
if [[ -z "${MODEL:-}" ]]; then
  for c in "$HOME/Downloads/native_kit/stories15M.bin" "$HOME/llama_gap/stories15M.bin" "$HOME/llama_rigorous/stories15M.bin" \
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
log "building (gcc -Ofast -fopenmp -march=native, as make runomp)"
CFLAGS=(-Ofast -fopenmp -march=native)
gcc "${CFLAGS[@]}" src/upstream/run.c -lm -o bin-up &&
  gcc "${CFLAGS[@]}" src/master/run/run.c -lm -o bin-master &&
  gcc "${CFLAGS[@]}" src/ours/run.c -lm -o bin-ours || { log "build failed"; exit 1; }

# one run: stdout/stderr to files, rusage of llama2c, other processes' CPU time, temperature -> one JSON line
cat > "$OUT/runone.py" <<'PY'
import hashlib, json, os, re, sys, tempfile, time
from pathlib import Path
log, block, order, cid, i = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv.index("--")
env = dict(os.environ, **dict(a.split("=", 1) for a in sys.argv[5:i]))
cmd = sys.argv[i + 1:]
tick = os.sysconf("SC_CLK_TCK")
def max_temp():
    m = 0
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try: m = max(m, int(z.read_text()))
        except (OSError, ValueError): pass
    return m // 1000
def busy():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return (v[0] + v[1] + v[2] + v[5] + v[6] + (v[7] if len(v) > 7 else 0)) / tick
def freq(cpus):
    f = []
    for c in cpus:
        try: f.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()))
        except (OSError, ValueError): pass
    return round(sum(f) / len(f) / 1000) if f else 0
cpus = [int(c) for c in cmd[2].split(",")]  # cmd = taskset -c CPUS binary ...
out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
t0, f0, b0, w0 = max_temp(), freq(cpus), busy(), time.time()
pid = os.fork()
if pid == 0:
    os.dup2(out.fileno(), 1); os.dup2(err.fileno(), 2)
    try: os.execvpe(cmd[0], cmd, env)
    finally: os._exit(127)
_, status, ru = os.wait4(pid, 0)
w1, b1, t1 = time.time(), busy(), max_temp()
out.seek(0); err.seek(0); o, e = out.read(), err.read()
m = re.search(rb"achieved tok/s:\s*(-?[\d.]+)", e)
child = ru.ru_utime + ru.ru_stime
rec = {"block": block, "order": order, "id": cid, "tok_s": float(m.group(1)) if m else None,
       "sha": hashlib.sha256(o).hexdigest()[:16], "exit": os.waitstatus_to_exitcode(status),
       "nivcsw": ru.ru_nivcsw, "other_cpu_s": round(b1 - b0 - child, 3), "wall_s": round(w1 - w0, 3),
       "temp_before": t0, "temp_after": t1, "freq_before": f0, "time": w0,
       "env": {k: v for k, v in env.items() if k.startswith(("OMP_", "GOMP_"))}, "cmd": cmd}
with open(log, "a") as f:
    f.write(json.dumps(rec) + "\n")
if rec["exit"] != 0:
    sys.stderr.write(e.decode(errors="replace"))
PY

ZONES=(/sys/class/thermal/thermal_zone*/temp)
max_temp() { local m=0 t z; for z in "${ZONES[@]}"; do read -r t 2>/dev/null < "$z" || t=0; (( t > m )) && m=$t; done; echo $((m / 1000)); }
quiet_check() { # label -> logs the busiest other processes over 5 s; returns 1 if they use more than 3% of one CPU
  python3 -I - "$1" "$$" >> "$OUT/quiet_check.txt" <<'PYQ'
import os, sys, time
label, me = sys.argv[1], int(sys.argv[2])
tick = os.sysconf("SC_CLK_TCK")
def snap():
    s = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                f = open(f"/proc/{p}/stat").read()
                v = f[f.rindex(")") + 2:].split()
                s[int(p)] = (f[f.index("(") + 1:f.rindex(")")], int(v[11]) + int(v[12]))
            except (OSError, ValueError, IndexError):
                pass
    return s
a = snap(); time.sleep(5); b = snap()
use = sorted(((b[p][1] - a[p][1]) / tick / 5 * 100, b[p][0], p) for p in b if p in a and p not in (me, os.getpid()))[::-1]
total = sum(u for u, _, _ in use)
print(f"[{time.strftime('%H:%M:%S')}] {label}: other processes used {total:.1f}% of one CPU over 5 s; top: "
      + ", ".join(f"{n}({p}) {u:.1f}%" for u, n, p in use[:5] if u > 0))
sys.exit(1 if total > 3 else 0)
PYQ
}

run_one() { # block order spec
  local blk="$1" ord="$2" spec="$3" id bin args cpus pol sch spin label n places
  IFS='|' read -r id bin args cpus pol sch spin label <<< "$spec"
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  places=$(tr ',' '\n' <<< "$cpus" | sed 's/.*/{&}/' | paste -sd,)
  local envs=(OMP_NUM_THREADS="$n" OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES="$places" OMP_SCHEDULE="$sch")
  [[ "$pol" != default ]] && envs+=(OMP_WAIT_POLICY="$pol")
  [[ "$spin" != - ]] && envs+=(GOMP_SPINCOUNT="$spin")
  local extra=(); [[ "$args" != - ]] && read -r -a extra <<< "$args"
  python3 -I "$OUT/runone.py" "$OUT/runs.jsonl" "$blk" "$ord" "$id" "${envs[@]}" -- \
    taskset -c "$cpus" "./bin-$bin" "$MODEL" -z tokenizer.bin -n 256 -t 0 -s 42 -i "$PROMPT" "${extra[@]}" 2>> "$OUT/errors.log"
}

{
  echo "start=$(date -Is) host=$(hostname) note=${NOTE:-} blocks=$BLOCKS"
  echo "commits: master=$C_MASTER ours=$C_OURS"
  echo "ac_online=$ac original_profile=${orig:-n/a} profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "tty=$(tty 2>/dev/null) session=${XDG_SESSION_TYPE:-}"
  uname -a; head -2 /etc/os-release; gcc --version | head -1; python3 --version
  echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -c1-16)"
  for f in src/upstream/run.c src/master/run/run.c src/master/src/fused.h src/master/src/classifier.h src/ours/run.c tokenizer.bin \
           bin-up bin-master bin-ours; do
    echo "file $f sha256=$(sha256sum "$f" | cut -c1-16)"
  done
  echo; printf 'cand %s\n' "${CANDIDATES[@]}"
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
} > "$OUT/environment.txt"

cat "$MODEL" > /dev/null
log "warm-up: every candidate once (not recorded)"
for spec in "${CANDIDATES[@]}"; do run_one 0 0 "$spec"; done
python3 -I - "$OUT/runs.jsonl" <<'PY' || { log "ERROR: a candidate failed or printed different text in the warm-up (see runs.jsonl, errors.log)"; exit 1; }
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
bad = [r["id"] for r in rs if r["exit"] != 0 or not r["tok_s"]]
shas = {r["sha"] for r in rs}
print("warm-up:", len(rs), "runs, failed:", bad, "distinct texts:", len(shas))
sys.exit(1 if bad or len(shas) != 1 else 0)
PY
mv "$OUT/runs.jsonl" "$OUT/warmup.jsonl"

log "checking that nothing else is using the CPU (5 s)"
if ! quiet_check start; then
  tail -1 "$OUT/quiet_check.txt"
  if [[ "${STRICT:-0}" == 1 ]]; then log "other processes are busy (STRICT=1): close them and run again"; exit 1; fi
  log "WARNING: other processes are using the CPU (see quiet_check.txt). Close them now if you can; continuing in 20 s"
  sleep 20; quiet_check "start (after 20 s)" || log "WARNING: still busy, measuring anyway"
fi
log "resting ${INITIAL_REST}s to measure the idle temperature. From now on do not touch the PC"
sleep "$INITIAL_REST"
s=0; for i in 1 2 3 4 5; do s=$((s + $(max_temp))); sleep 1; done
IDLE_TEMP=$((s / 5)); COOL_TEMP="${COOL_TEMP:-$((IDLE_TEMP + COOL_MARGIN))}"
echo "idle_temp=$IDLE_TEMP cool_temp=$COOL_TEMP" >> "$OUT/environment.txt"
log "idle ${IDLE_TEMP} C; ${#CANDIDATES[@]} candidates x $BLOCKS blocks, each block after resting to <= ${COOL_TEMP} C"
printf 'block\tstart\twait_s\ttemp_start\ttemp_end\trun_s\n' > "$OUT/blocks.tsv"
T0=$(date +%s)
for ((b = 1; b <= BLOCKS; b++)); do
  start=$(date +%s); below=0
  sleep "$REST_MIN"
  while :; do
    t=$(max_temp)
    if (( t <= COOL_TEMP )); then below=$((below + 1)); else below=0; fi
    (( below >= 2 )) && break
    (( $(date +%s) - start >= COOL_MAX )) && break
    sleep 2
  done
  waited=$(( $(date +%s) - start )); tstart=$(max_temp); r0=$(date +%s)
  ord=0
  while IFS= read -r spec; do
    ord=$((ord + 1)); run_one "$b" "$ord" "$spec"
  done < <(printf '%s\n' "${CANDIDATES[@]}" | shuf)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$b" "$r0" "$waited" "$tstart" "$(max_temp)" "$(( $(date +%s) - r0 ))" >> "$OUT/blocks.tsv"
  el=$(( $(date +%s) - T0 ))
  log "  block $b/$BLOCKS (rested ${waited}s from ${t} C)  elapsed $((el / 60)) min, rough ETA $(( el * (BLOCKS - b) / b / 60 )) min"
done
quiet_check end || true
echo "end=$(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"

cat > "$OUT/analyze_staircase.py" <<'PY'
#!/usr/bin/env python3
"""usage: python3 analyze_staircase.py RESULT_DIR [RESULT_DIR ...]   (several dirs = several sessions)
Prints summary tables and writes staircase.csv (one row per main step) into the first directory."""
import csv, json, os, random, statistics as st, sys
from collections import defaultdict

rows, label = [], {}
for s, d in enumerate(sys.argv[1:], 1):
    for l in open(os.path.join(d, "runs.jsonl")):
        r = json.loads(l); r["session"] = s; rows.append(r)
    for l in open(os.path.join(d, "environment.txt")):
        if l.startswith("cand "):
            f = l[5:].rstrip("\n").split("|"); label.setdefault(f[0], f[7])
ok = [r for r in rows if r["exit"] == 0 and r["tok_s"] and r["tok_s"] > 0]
by = defaultdict(dict)
for r in ok:
    by[r["id"]][(r["session"], r["block"])] = r["tok_s"]
rng = random.Random(42)
ids = [i for i in label if i in by]


def med(i):
    return st.median(by[i].values())


def paired(a, b, n=4000):
    rs = [by[a][k] / by[b][k] for k in by[a] if k in by[b]]
    m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(n))
    return st.median(rs), m[int(0.025 * n)], m[int(0.975 * n)], sum(x > 1 for x in rs), len(rs)


def ci(v, n=4000):
    m = sorted(st.median(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def fp(p):
    r, lo, hi, w, n = p
    return f"{r:6.3f} [{lo:.3f}, {hi:.3f}] {w:>2}/{n:<2}" + (" *" if lo > 1 or hi < 1 else "  ")


print(f"sessions: {len(sys.argv) - 1}   runs: {len(rows)}   failed or invalid: {len(rows) - len(ok)}")
shas = {r["sha"] for r in ok}
print(f"distinct generated texts: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK)"))
niv = [r["nivcsw"] for r in ok]; oth = [r["other_cpu_s"] for r in ok]
tb = [r["temp_before"] for r in ok]
print(f"quietness: involuntary context switches per run median {st.median(niv):.0f}, p90 {sorted(niv)[int(0.9 * len(niv))]}; "
      f"other processes' CPU s per run median {st.median(oth):.3f}; temperature before runs {min(tb)}-{max(tb)} C")

base = "S1_upstream"
print(f"\n== all candidates (tok/s; 'team' = 1000 / mean ms per token) ==")
print(f"  {'id':<19}{'n':>4}{'median':>8}{'95% CI':>14}{'mean':>7}{'team':>7}{'min':>7}{'CV%':>6}{'x S1':>7}   label")
for i in ids:
    v = list(by[i].values()); lo, hi = ci(v)
    print(f"  {i:<19}{len(v):>4}{med(i):>8.0f}{f'[{lo:.0f},{hi:.0f}]':>14}{st.mean(v):>7.0f}{len(v) / sum(1 / x for x in v):>7.0f}"
          f"{min(v):>7.0f}{st.pstdev(v) / st.mean(v) * 100:>6.1f}{med(i) / med(base):>7.2f}   {label[i]}")

steps = [i for i in ids if i.startswith("S")]
print("\n== staircase: each step against the previous one (paired within block, median [95% CI] faster-blocks, * = CI excludes 1) ==")
out = []
prev = None
for i in steps:
    v = list(by[i].values())
    step = paired(i, prev) if prev else None
    total = paired(i, base) if i != base else None
    print(f"  {i:<14}{med(i):>7.0f} tok/s   step {fp(step) if step else '-':<32}   x S1 {total[0] if total else 1:.2f}   {label[i]}")
    out.append({"step": i, "label": label[i], "median_tok_s": round(med(i), 1), "mean_tok_s": round(st.mean(v), 1),
                "step_ratio": round(step[0], 4) if step else "", "step_ci_low": round(step[1], 4) if step else "",
                "step_ci_high": round(step[2], 4) if step else "", "ratio_to_upstream": round(total[0], 4) if total else 1.0,
                "n": len(v)})
    prev = i
with open(os.path.join(sys.argv[1], "staircase.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

Q = [("fusion alone (no hugepage) vs original", "F_fused_only", "S1_upstream"),
     ("hugepage alone vs original", "S2_hugepage", "S1_upstream"),
     ("hugepage on top of fusion", "S3_fused", "F_fused_only"),
     ("sakurada -w 5 vs master -w 1", "F_ours_w5", "S3_fused"),
     ("sakurada -w 5 -a 1 vs master -w 1 -a 1", "F_ours_w5a1", "S4_argmax"),
     ("ACTIVE on 9 CPUs (guided,4)", "S6_nine", "F_nine_default"),
     ("guided,4 vs static on 9 CPUs", "S6_nine", "F_nine_static"),
     ("12 threads: static vs guided,4", "S8_twelve", "F_twelve_guided4"),
     ("team's 9 vs 12 test: 9 guided,4 / 12 guided,4", "S6_nine", "F_twelve_guided4"),
     ("kasikoi2 pick vs its start (all 14, defaults)", "S7_kasikoi2", "F_all14_default"),
     ("original code: 0-11 vs 0-3", "F_upstream_twelve", "S1_upstream"),
     ("code effect on 0-11: S8 vs original on 0-11", "S8_twelve", "F_upstream_twelve")]
print("\n== factor checks (paired ratio first / second) ==")
for lab, a, b in Q:
    if a in by and b in by:
        print(f"  {lab:<48} {fp(paired(a, b))}")
print(f"\nstaircase.csv written to {sys.argv[1]}")
PY
python3 -I "$OUT/analyze_staircase.py" "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
