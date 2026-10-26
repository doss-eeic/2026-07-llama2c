"""Repeat the Candle thread-count comparison with pinned local binaries.

Usage: python3 scripts/candle_thread_cache_rerun.py UPSTREAM FIX ARGMAX MODEL TOKENIZER ROUNDS OUT
Each condition gets one unrecorded warmup, then ROUNDS randomized paired runs.
"""

import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path


def run(binary, model, tokenizer, threads=None, argmax=False):
    controlled = {"RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "CANDLE_CPU_MASK", "CANDLE_CPU_AFFINITY"}
    env = {k: v for k, v in os.environ.items() if k not in controlled}
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
    cmd = [binary, "--cpu", "--repeat-penalty", "1", "--tokenizer", tokenizer,
           "inference", "--config", model, "--temperature", "0",
           "--prompt", "Once upon a time"]
    if argmax:
        cmd.append("--two-stage-argmax")
    output = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    match = re.search(r"\(([\d.]+) token/s\)", output)
    if match is None:
        raise RuntimeError(f"token/s missing from output of {cmd}")
    generated = output.split("starting the inference loop\n", 1)[1].rsplit("\n", 3)[0]
    return float(match.group(1)), hashlib.md5(generated.encode()).hexdigest()[:12]


def main():
    upstream, fix, argmax, model, tokenizer, rounds, out = sys.argv[1:]
    rounds = int(rounds)
    configs = {
        "upstream_default": (upstream, None, False),
        "upstream_explicit_12": (upstream, 12, False),
        "upstream_explicit_14": (upstream, 14, False),
        "cache_default": (fix, None, False),
        "cache_argmax_default": (argmax, None, True),
    }
    randomizer = random.Random(20261026)
    for binary, threads, two_stage in configs.values():
        run(binary, model, tokenizer, threads, two_stage)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with path.open("w") as f:
        for round_no in range(rounds):
            names = list(configs)
            randomizer.shuffle(names)
            for name in names:
                binary, threads, two_stage = configs[name]
                tok_s, digest = run(binary, model, tokenizer, threads, two_stage)
                row = {"round": round_no, "config": name, "tok_s": tok_s, "text_md5_12": digest}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    print(f"raw data: {path}")
    print(f"text hashes: {sorted({row['text_md5_12'] for row in rows})}")
    print("| condition | mean token/s | median token/s | min–max |")
    print("|---|---:|---:|---:|")
    for name in configs:
        values = [row["tok_s"] for row in rows if row["config"] == name]
        print(f"| {name} | {statistics.mean(values):.1f} | {statistics.median(values):.1f} | "
              f"{min(values):.1f}–{max(values):.1f} |")


if __name__ == "__main__":
    main()
