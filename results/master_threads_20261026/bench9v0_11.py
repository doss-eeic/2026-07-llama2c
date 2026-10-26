import json, os, random, re, subprocess, sys

S = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/denjo/2026-07-llama2c"
CASES = {
    "9core": [0, 2, 4, 5, 6, 7, 8, 9, 10],
    "12core": list(range(12)),
}
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
out = open(os.path.join(S, "bench9v0_11.jsonl"), "w")
rng = random.Random(42)

def run(name):
    cpus = CASES[name]
    env = dict(os.environ, OMP_NUM_THREADS=str(len(cpus)), OMP_SCHEDULE="guided,4",
               OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_WAIT_POLICY="ACTIVE",
               OMP_PLACES=",".join(f"{{{c}}}" for c in cpus))
    cmd = ["taskset", "-c", ",".join(map(str, cpus)), os.path.join(S, "run-master"),
           f"{ROOT}/build/stories15M.bin", "-z", f"{ROOT}/tokenizer.bin", "-n", "256", "-t", "0",
           "-s", "42", "-i", "Once upon a time", "-w", "1", "-a", "1", "-g", "288"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    m = re.search(r"achieved tok/s: ([\d.]+)", p.stderr + p.stdout)
    return float(m.group(1)), hash(p.stdout)

run("9core"); run("12core")  # warmup
for r in range(ROUNDS):
    order = list(CASES); rng.shuffle(order)
    for name in order:
        tps, h = run(name)
        out.write(json.dumps({"round": r, "case": name, "tps": tps, "out": h}) + "\n"); out.flush()
