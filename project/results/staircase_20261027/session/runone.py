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
