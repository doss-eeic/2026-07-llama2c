#!/usr/bin/env bash
# Run the team's autotuner on dev/opt (`uv run cpuprofiler kasikoi2`) on a quiet PC, record every benchmark run
# it makes, and re-check its pick against known configurations in randomized blocks.
#   1. download dev/opt at a pinned commit, build build/run-fused from its run.c (gcc -Ofast -march=native -fopenmp)
#   2. put a recording wrapper at build/run-fused (the real binary is build/run-fused.real). For every run it logs
#      the OpenMP settings, tok/s, sha256 of the generated text, involuntary context switches of llama2c
#      (= times another task took its CPU), CPU time used by other processes during the run, and temperature.
#      stdout/stderr go to files during the run, so nothing wakes up while generating.
#   3. check that nothing else is using the CPU (the 5 busiest other processes are logged; STRICT=1 aborts)
#   4. run the unmodified `uv run cpuprofiler kasikoi2` SEARCHES times (rest until cool before each)
#   5. confirmation: kasikoi2's picks, its input configuration (all CPUs), 9 CPUs (0,2,4-10) guided,4 ACTIVE g288 and
#      12 threads (0-11) static ACTIVE g288, all through the team's BenchmarkRunner, BLOCKS blocks in random order
# Power profile "performance" for the whole run, restored at the end. No sudo. Installs uv into ~/.local/bin if missing.
# Recommended: close every other program, and start it from a text console (Ctrl+Alt+F3, log in, run, come back
# with Ctrl+Alt+F2 or F1 afterwards) so that the desktop stops drawing. About 15-25 min.
# usage:
#   curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/dev/sakurada/opt_quiet_check.sh -o opt_quiet_check.sh && bash opt_quiet_check.sh
# options (env): SEARCHES=2 BLOCKS=15 INITIAL_REST=60 COOL_MARGIN=8 REST_MIN=10 COOL_MAX=60 SEARCH_REST_MAX=300
#                STRICT=1 (abort if other processes use CPU)  NOTE=text  DIR=work dir  FORCE=1 (no AC / no profile)
set -uo pipefail
export LC_ALL=C
C_OPT=1d7c213622901900d9fb6980b5b24512726830da
C_SCRIPT=dev/sakurada
REPO=doss-eeic/2026-07-llama2c
DIR="${DIR:-$HOME/llama_opt}"
SEARCHES="${SEARCHES:-2}"
BLOCKS="${BLOCKS:-15}"
INITIAL_REST="${INITIAL_REST:-60}"
COOL_MARGIN="${COOL_MARGIN:-8}"
REST_MIN="${REST_MIN:-10}"
COOL_MAX="${COOL_MAX:-60}"
SEARCH_REST_MAX="${SEARCH_REST_MAX:-300}"
mkdir -p "$DIR"
cd "$DIR"
OUT="$DIR/results_$(hostname)_opt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
unset OMP_WAIT_POLICY OMP_SCHEDULE OMP_PLACES OMP_PROC_BIND OMP_NUM_THREADS OMP_DYNAMIC GOMP_SPINCOUNT \
      PERSISTENT_OPENMP OUTPUT_THREADS VIRTUAL_ENV PYTHONPATH

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }

ac="$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)"
if [[ "$ac" != 1 && "${FORCE:-0}" != 1 ]]; then
  echo "AC power is not connected (online=$ac). Plug in the charger, or run with FORCE=1." >&2; exit 1
fi
for tool in gcc taskset python3 sha256sum tar curl; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "uv not found: installing it into ~/.local/bin (official installer, no sudo)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >> "$OUT/progress.log" 2>&1 || { log "uv install failed"; exit 1; }
  command -v uv >/dev/null || { log "uv still not found"; exit 1; }
fi
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

# ---------------------------------------------------------------- source, build, wrapper
SRC="$DIR/src-$C_OPT"
if [[ ! -f "$SRC/pyproject.toml" ]]; then
  log "downloading dev/opt at ${C_OPT:0:8} (includes the 60MB model)"
  rm -rf "$DIR/dl.tmp" && mkdir -p "$DIR/dl.tmp" &&
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$C_OPT" | tar xz -C "$DIR/dl.tmp" &&
    mv "$DIR/dl.tmp"/*/ "$SRC" && rm -rf "$DIR/dl.tmp" || { log "download failed"; exit 1; }
fi
curl -fsSL "https://raw.githubusercontent.com/$REPO/$C_SCRIPT/opt_quiet_check.sh" -o "$OUT/opt_quiet_check.sh" 2>/dev/null ||
  cp "$0" "$OUT/opt_quiet_check.sh" 2>/dev/null
cd "$SRC"
[[ -f build/run-fused.committed ]] || cp build/run-fused build/run-fused.committed
log "building build/run-fused.real from run.c (gcc -Ofast -fopenmp -march=native)"
gcc -Ofast -fopenmp -march=native run.c -lm -o build/run-fused.real || { log "build failed"; exit 1; }

cat > build/run-fused <<'PY'
#!/usr/bin/env python3
# Recording wrapper installed by opt_quiet_check.sh: runs run-fused.real with the same arguments and appends one JSON
# line per run to $WRAP_LOG. stdout/stderr are kept in files until the run ends, then passed on unchanged.
import hashlib, json, os, re, sys, tempfile, time
from pathlib import Path

real = str(Path(__file__).resolve().with_name("run-fused.real"))
tick = os.sysconf("SC_CLK_TCK")


def max_temp():
    m = 0
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            m = max(m, int(z.read_text()))
        except (OSError, ValueError):
            pass
    return m // 1000


def busy_seconds():  # all CPUs: user+nice+system+irq+softirq+steal
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    return (v[0] + v[1] + v[2] + v[5] + v[6] + (v[7] if len(v) > 7 else 0)) / tick


def run_once(attempt):
    out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    t0, b0, w0 = max_temp(), busy_seconds(), time.time()
    pid = os.fork()
    if pid == 0:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            os.execv(real, [real] + sys.argv[1:])
        finally:
            os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    w1, b1, t1 = time.time(), busy_seconds(), max_temp()
    err.seek(0); e = err.read()
    out.seek(0); o = out.read()
    m = re.search(rb"achieved tok/s:\s*(-?[\d.]+)", e)
    tok = float(m.group(1)) if m else None
    child = ru.ru_utime + ru.ru_stime
    if os.environ.get("WRAP_LOG"):
        rec = {
            "time": w0, "phase": os.environ.get("WRAP_PHASE", ""), "attempt": attempt,
            "cpus": sorted(os.sched_getaffinity(0)),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "GOMP_"))}, "args": sys.argv[1:],
            "tok_s": tok, "sha": hashlib.sha256(o).hexdigest()[:16],
            "nivcsw": ru.ru_nivcsw, "nvcsw": ru.ru_nvcsw, "child_cpu_s": round(child, 4),
            "other_cpu_s": round(b1 - b0 - child, 3), "wall_s": round(w1 - w0, 4),
            "temp_before": t0, "temp_after": t1, "exit": os.waitstatus_to_exitcode(status),
        }
        with open(os.environ["WRAP_LOG"], "a") as f:
            f.write(json.dumps(rec) + "\n")
    return status, tok, o, e


# a wall-clock jump can make llama2c print a negative tok/s; such a run is measured again (at most twice more)
for attempt in range(3):
    status, tok, o, e = run_once(attempt)
    if os.waitstatus_to_exitcode(status) != 0 or (tok is not None and tok > 0):
        break
sys.stderr.buffer.write(e); sys.stderr.flush()
sys.stdout.buffer.write(o); sys.stdout.flush()
sys.exit(os.waitstatus_to_exitcode(status))
PY
chmod +x build/run-fused
export WRAP_LOG="$OUT/runs.jsonl"

cat > "$OUT/confirm.py" <<'PY'
"""Confirmation blocks through the team's BenchmarkRunner (run with `uv run python`)."""
import json, os, random, re, sys, time
from pathlib import Path, PosixPath
from cpuprofiler import BenchmarkConfig, BenchmarkRunner, CPUTopology

out, blocks = Path(sys.argv[1]), int(sys.argv[2])
cool, rest_min, cool_max = int(os.environ["COOL_TEMP"]), int(os.environ["REST_MIN"]), int(os.environ["COOL_MAX"])


def max_temp():
    m = 0
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            m = max(m, int(z.read_text()))
        except (OSError, ValueError):
            pass
    return m // 1000


cands = {}
for log in sorted(out.glob("search*.log")):
    m = re.search(r"^Best configuration: (BenchmarkConfig\(.*\))$", log.read_text(), re.M)
    if m:
        cfg = eval(m.group(1), {"BenchmarkConfig": BenchmarkConfig, "PosixPath": PosixPath})
        if cfg not in cands.values():
            cands[f"kasikoi2_{log.stem}"] = cfg
available = tuple(sorted(CPUTopology().cpus))
cands["kasikoi2_input_all"] = BenchmarkConfig(cpus=available, threads=len(available),
                                              optimize_weights=True, two_stage_argmax=True)
refs = {"ref_N9_guided4": ((0, 2, 4, 5, 6, 7, 8, 9, 10), "guided,4"), "ref_PE12_static": (tuple(range(12)), "static")}
for name, (cpus, sched) in refs.items():
    if set(cpus) <= set(available):
        cands[name] = BenchmarkConfig(cpus=cpus, threads=len(cpus), omp_schedule=sched, omp_wait_policy="ACTIVE",
                                      optimize_weights=True, two_stage_argmax=True, group_size=288)
with open(out / "confirm_candidates.txt", "w") as f:
    for name, cfg in cands.items():
        f.write(f"{name}\t{cfg!r}\n")
runner, rng = BenchmarkRunner(), random.Random(42)
os.environ["WRAP_PHASE"] = "confirm_warmup"
for cfg in cands.values():
    runner.run(cfg)
os.environ["WRAP_PHASE"] = "confirm"
with open(out / "confirm.jsonl", "w") as f:
    for b in range(1, blocks + 1):
        start = time.time()
        time.sleep(rest_min)
        below = 0
        while time.time() - start < cool_max:
            below = below + 1 if max_temp() <= cool else 0
            if below >= 2:
                break
            time.sleep(2)
        waited, temp = round(time.time() - start), max_temp()
        order = list(cands)
        rng.shuffle(order)
        for i, name in enumerate(order, 1):
            r = runner.run(cands[name])
            f.write(json.dumps({"block": b, "order": i, "name": name, "tok_s": r.tokens_per_second,
                                "wait_s": waited, "temp_start": temp}) + "\n")
            f.flush()
        print(f"  confirm block {b}/{blocks} done ({max_temp()} C)", flush=True)
PY

cat > "$OUT/analyze_opt.py" <<'PY'
#!/usr/bin/env python3
"""usage: python3 analyze_opt.py RESULT_DIR"""
import json, random, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
runs = [json.loads(l) for l in open(d / "runs.jsonl")]
rng = random.Random(42)
bad = [r for r in runs if r["exit"] != 0 or r["tok_s"] is None or r["tok_s"] <= 0]
print(f"benchmark runs recorded: {len(runs)}   failed or invalid tok/s (measured again): {len(bad)}")
runs = [r for r in runs if r not in bad]
shas = {r["sha"] for r in runs if r["phase"] != "setup"}
print(f"distinct generated texts: {len(shas)} " + ("(OK, all identical)" if len(shas) == 1 else "(CHECK)"))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


print("\n== quietness per phase (from the wrapper) ==")
print("  nivcsw = times another task took a CPU from llama2c during one run; other = CPU seconds used by everything else")
by_phase = defaultdict(list)
for r in runs:
    by_phase[r["phase"]].append(r)
for ph, rs in by_phase.items():
    niv = [r["nivcsw"] for r in rs]
    oth = [r["other_cpu_s"] for r in rs]
    print(f"  {ph:<16} runs {len(rs):>4}  nivcsw median {st.median(niv):>5.0f} p90 {q(niv, 0.9):>5}  "
          f"other CPU s median {st.median(oth):.3f} p90 {q(oth, 0.9):.3f}  temp max {max(r['temp_after'] for r in rs)} C")

print("\n== kasikoi2 searches ==")
for log in sorted(d.glob("search*.log")):
    t = log.read_text()
    best = re.search(r"^Best configuration: (.*)$", t, re.M)
    perf = re.search(r"^Best performance: ([\d.]+)", t, re.M)
    srch = re.search(r"^Search: ([\d.]+) sec, (\d+) measurements", t, re.M)
    print(f"  {log.stem}: " + (f"{float(perf.group(1)):.0f} tok/s, {srch.group(1)} s, {srch.group(2)} runs" if perf and srch else "did not finish"))
    if best:
        b = best.group(1)
        keep = [m.group(0) for m in re.finditer(r"(cpus=\([^)]*\)|threads=\d+|omp_\w+=(?:'[^']*'|[^,)]+)|gomp_spincount=[^,)]+|group_size=\d+)", b)]
        print("    " + ", ".join(keep))

cf = d / "confirm.jsonl"
if cf.exists():
    rows = [json.loads(l) for l in open(cf)]
    by = defaultdict(dict)
    for r in rows:
        by[r["name"]][r["block"]] = r["tok_s"]
    med = {n: st.median(v.values()) for n, v in by.items()}
    order = sorted(by, key=lambda n: -med[n])
    best = order[0]
    print(f"\n== confirmation ({len({r['block'] for r in rows})} blocks, random order; 'team' = 1000 / mean ms per token) ==")
    print(f"  {'name':<22}{'n':>4}{'median':>8}{'mean':>8}{'team':>8}{'min':>7}{'CV%':>6}   vs best (paired median [95% CI] faster-blocks)")
    for n in order:
        v = list(by[n].values())
        hm = len(v) / sum(1 / x for x in v)
        line = f"  {n:<22}{len(v):>4}{med[n]:>8.0f}{st.mean(v):>8.0f}{hm:>8.0f}{min(v):>7.0f}{st.pstdev(v) / st.mean(v) * 100:>6.1f}"
        if n != best:
            rs = [by[n][k] / by[best][k] for k in by[n] if k in by[best]]
            m = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(3000))
            line += f"   {st.median(rs):.3f} [{m[75]:.3f},{m[2924]:.3f}] {sum(x > 1 for x in rs)}/{len(rs)}"
        print(line)
    cands = d / "confirm_candidates.txt"
    if cands.exists():
        print("\n  candidates:")
        for l in open(cands):
            n, c = l.rstrip("\n").split("\t", 1)
            keep = [m.group(0) for m in re.finditer(r"(cpus=\([^)]*\)|threads=\d+|omp_\w+=(?:'[^']*'|[^,)]+)|gomp_spincount=[^,)]+|group_size=\d+)", c)]
            print(f"    {n:<22}" + ", ".join(keep))
PY

# ---------------------------------------------------------------- setup run (also creates uv's environment)
log "preparing the uv environment (downloads Python 3.14 if needed)"
WRAP_PHASE=setup uv run cpuprofiler topology > "$OUT/topology.txt" 2>&1 || { log "uv run failed (see topology.txt)"; exit 1; }
WRAP_PHASE=setup ./build/run-fused build/stories15M.bin -n 16 -t 0 -s 42 -i "Once upon a time" -w 1 -a 1 > /dev/null 2>&1 ||
  { log "wrapper test failed"; exit 1; }
cat build/stories15M.bin > /dev/null

ZONES=(/sys/class/thermal/thermal_zone*/temp)
max_temp() { local m=0 t z; for z in "${ZONES[@]}"; do read -r t 2>/dev/null < "$z" || t=0; (( t > m )) && m=$t; done; echo $((m / 1000)); }
quiet_check() { # label -> logs the busiest other processes over 5 s; returns 1 if they use more than 3% of a CPU
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
                name = f[f.index("(") + 1:f.rindex(")")]
                v = f[f.rindex(")") + 2:].split()
                s[int(p)] = (name, int(v[11]) + int(v[12]))
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
wait_cool() { # $1 = longest wait
  local start below=0 t
  start=$(date +%s); sleep "$REST_MIN"
  while :; do
    t=$(max_temp)
    if (( t <= COOL_TEMP )); then below=$((below + 1)); else below=0; fi
    (( below >= 2 )) && break
    (( $(date +%s) - start >= $1 )) && { log "  still ${t} C after $1 s (target ${COOL_TEMP} C)"; break; }
    sleep 3
  done
  log "  rested $(( $(date +%s) - start )) s, now $(max_temp) C"
}

{
  echo "start=$(date -Is) host=$(hostname) note=${NOTE:-} searches=$SEARCHES blocks=$BLOCKS"
  echo "commit dev/opt=$C_OPT"
  echo "ac_online=$ac original_profile=${orig:-n/a} profile=$(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  uname -a; head -2 /etc/os-release; gcc --version | head -1; uv --version; uv run python --version 2>&1
  echo "tty=$(tty 2>/dev/null) display=${DISPLAY:-} wayland=${WAYLAND_DISPLAY:-} session=${XDG_SESSION_TYPE:-}"
  for f in run.c src/classifier.h src/fused.h build/stories15M.bin tokenizer.bin build/run-fused.committed build/run-fused.real \
           src/cpuprofiler/autotuner.py src/cpuprofiler/benchmark.py src/cpuprofiler/topology.py src/cpuprofiler/__init__.py; do
    echo "file $f sha256=$(sha256sum "$f" | cut -c1-16)"
  done
  echo; lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
} > "$OUT/environment.txt"

log "checking that nothing else is using the CPU (5 s)"
if ! quiet_check start; then
  tail -1 "$OUT/quiet_check.txt"
  if [[ "${STRICT:-0}" == 1 ]]; then log "other processes are busy (STRICT=1): close them and run again"; exit 1; fi
  log "WARNING: other processes are using the CPU (see quiet_check.txt). Close them now if you can; continuing in 20 s"
  sleep 20; quiet_check "start (after 20 s)" || log "WARNING: still busy, measuring anyway"
fi
log "resting ${INITIAL_REST}s to measure the idle temperature. From now on do not touch the PC (no output until the end)"
sleep "$INITIAL_REST"
s=0; for i in 1 2 3 4 5; do s=$((s + $(max_temp))); sleep 1; done
IDLE_TEMP=$((s / 5)); COOL_TEMP="${COOL_TEMP:-$((IDLE_TEMP + COOL_MARGIN))}"
echo "idle_temp=$IDLE_TEMP cool_temp=$COOL_TEMP" >> "$OUT/environment.txt"
export COOL_TEMP REST_MIN COOL_MAX
log "idle ${IDLE_TEMP} C"

for ((i = 1; i <= SEARCHES; i++)); do
  wait_cool "$SEARCH_REST_MAX"
  quiet_check "before search $i" || log "  WARNING: other processes were using the CPU before search $i"
  log "search $i/$SEARCHES: uv run cpuprofiler kasikoi2"
  t0=$(date +%s)
  WRAP_PHASE="search$i" uv run cpuprofiler kasikoi2 > "$OUT/search$i.log" 2>&1 || log "  search $i failed (see search$i.log)"
  log "  search $i took $(( $(date +%s) - t0 )) s: $(grep '^Best performance' "$OUT/search$i.log")"
done
wait_cool "$SEARCH_REST_MAX"
quiet_check "before confirmation" || log "  WARNING: other processes were using the CPU before the confirmation"
log "confirmation: $BLOCKS blocks"
uv run python "$OUT/confirm.py" "$OUT" "$BLOCKS" >> "$OUT/progress.log" 2>&1 || log "confirmation failed (see progress.log)"
quiet_check end || true
echo "end=$(date -Is) profile=$(powerprofilesctl get 2>/dev/null || echo n/a)" >> "$OUT/environment.txt"

python3 -I "$OUT/analyze_opt.py" "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
tar czf "$OUT.tar.gz" -C "$DIR" "$(basename "$OUT")"
log "DONE. results: $OUT.tar.gz"
