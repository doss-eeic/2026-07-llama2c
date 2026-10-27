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
