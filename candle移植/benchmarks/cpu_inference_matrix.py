"""Measure the three Candle builds and every valid Int8 group size on this host."""

import argparse
import csv
import os
import random
import re
import struct
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
CPU_ORDER = PHYSICAL + (1, 3)
ALL = tuple(range(14))
RATE = re.compile(r"(\d+) tokens generated \(([\d.]+) token/s\)")
FIELDS = ("phase", "condition", "revision", "round", "cpus", "threads",
          "group_size", "generated_tokens", "token_per_second", "wall_seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "data/stories15M.bin")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer.json")
    parser.add_argument("--upstream", type=Path,
                        default=ROOT.parent / "candle-upstream/target/release/examples/llama2-c")
    parser.add_argument("--cache", type=Path,
                        default=ROOT.parent / "candle-cache/target/release/examples/llama2-c")
    parser.add_argument("--optimized", type=Path,
                        default=ROOT / "target/release/examples/llama2-c")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/results/cpu-inference-2026-10-27.csv")
    parser.add_argument("--cache-rounds", type=int, default=20)
    parser.add_argument("--group-rounds", type=int, default=5)
    parser.add_argument("--scaling-rounds", type=int, default=5)
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--group-sizes", type=int, nargs="*")
    args = parser.parse_args()
    model = args.model.resolve()
    tokenizer = args.tokenizer.resolve()
    with model.open("rb") as source:
        dim, = struct.unpack("i", source.read(4))
    groups = [size for size in range(args.min_group_size, dim + 1) if dim % size == 0]
    if args.group_sizes is not None:
        groups = [size for size in groups if size in args.group_sizes]
    binaries = {"upstream": args.upstream.resolve(), "cache": args.cache.resolve(),
                "optimized": args.optimized.resolve()}
    rng = random.Random(20261027)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(phase, condition, revision, round_number, cpus, threads=None, group_size=None):
        mask = ",".join(map(str, cpus))
        command = ["taskset", "-c", mask, str(binaries[revision]), "--cpu",
                   "--tokenizer", str(tokenizer), "--repeat-penalty", "1",
                   "inference", "--config", str(model), "--temperature", "0",
                   "--prompt", "Once upon a time"]
        if group_size is not None:
            command += ["--two-stage-argmax", "--group-size", str(group_size)]
        env = os.environ.copy()
        for name in ("RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "CANDLE_CPU_MASK",
                     "CANDLE_CPU_AFFINITY"):
            env.pop(name, None)
        if threads is not None:
            env["RAYON_NUM_THREADS"] = str(threads)
            if phase != "cache_pr":
                env["CANDLE_NUM_THREADS"] = str(threads)
                env["CANDLE_CPU_MASK"] = mask
        start = time.perf_counter()
        output = subprocess.run(command, env=env, capture_output=True, text=True,
                                check=True, timeout=120)
        elapsed = time.perf_counter() - start
        match = RATE.search(output.stdout)
        if match is None:
            raise RuntimeError(f"No token rate in output: {output.stdout[-1000:]}")
        return dict(phase=phase, condition=condition, revision=revision,
                    round=round_number, cpus=mask, threads=threads or "default",
                    group_size=group_size or "", generated_tokens=int(match.group(1)),
                    token_per_second=float(match.group(2)), wall_seconds=round(elapsed, 4))

    cache_cases = [("upstream_default", "upstream", None),
                   ("upstream_12", "upstream", 12),
                   ("cache_default", "cache", None),
                   ("cache_12", "cache", 12)]
    group_cases = [("f32", None)] + [(f"int8_g{size}", size) for size in groups]
    scaling_cases = [(f"f32_{count}", "f32", count) for count in (1, 2, 4, 6, 8, 10, 12, 14)]
    scaling_cases += [(f"int8_g32_{count}", "int8", count)
                      for count in (1, 2, 4, 6, 8, 10, 12, 14)]

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        for round_number in range(1, args.cache_rounds + 1):
            cases = cache_cases.copy()
            rng.shuffle(cases)
            for condition, revision, threads in cases:
                writer.writerow(measure("cache_pr", condition, revision, round_number,
                                        ALL, threads))
            stream.flush()
            print(f"cache_pr round {round_number}/{args.cache_rounds}", flush=True)
        for round_number in range(1, args.group_rounds + 1):
            cases = group_cases.copy()
            rng.shuffle(cases)
            for condition, group_size in cases:
                writer.writerow(measure("groups_12", condition, "optimized", round_number,
                                        PHYSICAL, 12, group_size))
            stream.flush()
            print(f"groups_12 round {round_number}/{args.group_rounds}", flush=True)
        for round_number in range(1, args.scaling_rounds + 1):
            cases = scaling_cases.copy()
            rng.shuffle(cases)
            for condition, mode, count in cases:
                cpus = tuple(sorted(CPU_ORDER[:count]))
                writer.writerow(measure("scaling", condition, "optimized", round_number,
                                        cpus, count, 32 if mode == "int8" else None))
            stream.flush()
            print(f"scaling round {round_number}/{args.scaling_rounds}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
