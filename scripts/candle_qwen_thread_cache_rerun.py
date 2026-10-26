"""Benchmark Qwen2.5-1.5B-Instruct with three Candle thread-count conditions.

Usage: python3 scripts/candle_qwen_thread_cache_rerun.py UPSTREAM FIX MODEL_DIR ROUNDS OUT
"""

import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Once upon a time"
SAMPLE_LEN = 16


def run(binary, model_dir, threads):
    controlled = {"RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "CANDLE_CPU_MASK", "CANDLE_CPU_AFFINITY"}
    env = {key: value for key, value in os.environ.items() if key not in controlled}
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
    cmd = [binary, "--cpu", "--model", "2-1.5b", "--model-id", MODEL_ID,
           "--weight-path", model_dir, "--prompt", PROMPT, "--temperature", "0",
           "--repeat-penalty", "1", "--seed", "42", "--sample-len", str(SAMPLE_LEN)]
    start = time.perf_counter()
    # The qwen example resolves a single local model.safetensors relative to CWD.
    output = subprocess.run(cmd, env=env, cwd=model_dir, capture_output=True, text=True, check=True).stdout
    wall_s = time.perf_counter() - start
    matches = list(re.finditer(r"(\d+) tokens generated \(([\d.]+) token/s\)", output))
    if len(matches) != 1:
        raise RuntimeError(f"missing or duplicate throughput line: {output[-300:]}")
    n_tokens = int(matches[0].group(1))
    tok_s = float(matches[0].group(2))
    generated = output.split("loaded the model in ", 1)[1].split("\n", 1)[1].rsplit("\n", 2)[0]
    digest = hashlib.sha256(generated.encode()).hexdigest()[:16]
    return n_tokens, tok_s, wall_s, digest


def main():
    upstream, fix, model_dir, rounds, out = sys.argv[1:]
    rounds = int(rounds)
    configs = {
        "upstream_default": (upstream, None),
        "upstream_explicit_12": (upstream, 12),
        "cache_default": (fix, None),
    }
    for binary, threads in configs.values():
        run(binary, model_dir, threads)
    rng = random.Random(20261026)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with path.open("w") as f:
        for round_no in range(rounds):
            names = list(configs)
            rng.shuffle(names)
            for name in names:
                binary, threads = configs[name]
                n_tokens, tok_s, wall_s, digest = run(binary, model_dir, threads)
                row = {"round": round_no, "config": name, "tokens": n_tokens,
                       "tok_s": tok_s, "wall_s": round(wall_s, 3), "text_sha256_16": digest}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
    print(f"raw data: {path}")
    print(f"token counts: {sorted({row['tokens'] for row in rows})}")
    print(f"text hashes: {sorted({row['text_sha256_16'] for row in rows})}")
    print("| condition | mean token/s | median token/s | min–max |")
    print("|---|---:|---:|---:|")
    for name in configs:
        values = [row["tok_s"] for row in rows if row["config"] == name]
        print(f"| {name} | {statistics.mean(values):.2f} | {statistics.median(values):.2f} | "
              f"{min(values):.2f}–{max(values):.2f} |")


if __name__ == "__main__":
    main()
