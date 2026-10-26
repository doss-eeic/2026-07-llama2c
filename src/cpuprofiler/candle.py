"""Candle llama2-c CPU tuning, reusing the topology-aware search from AutoTuner.

Run with: PYTHONPATH=src python3 -m cpuprofiler.candle --help
"""
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import struct

from .autotuner import AutoTuner
from .benchmark import BenchmarkResult
from .topology import CPUTopology


@dataclass(frozen=True)
class CandleConfig:
    cpus: tuple[int, ...]
    threads: int
    prompt: str = "Once upon a time"
    group_size: int = 144
    two_stage_argmax: bool = True
    timeout: float = 120


class CandleRunner:
    def __init__(self, binary_path, model_path, tokenizer_path, results_path=None):
        self.binary_path = Path(binary_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.results_path = Path(results_path) if results_path else None
        for path in (self.binary_path, self.model_path, self.tokenizer_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with self.model_path.open("rb") as model:
            self.dim, = struct.unpack("<i", model.read(4))
        self.expected_hash = None

    def run(self, config):
        if not config.cpus or config.threads != len(config.cpus):
            raise ValueError("threads must equal the number of selected CPUs")
        if config.two_stage_argmax and (config.group_size < 1 or self.dim % config.group_size):
            raise ValueError(f"group size must divide model dimension {self.dim}")
        if not set(config.cpus) <= os.sched_getaffinity(0):
            raise ValueError("selected CPUs are outside the current process affinity")
        mask = ",".join(map(str, config.cpus))
        command = ["taskset", "-c", mask, str(self.binary_path), "--cpu",
                   "--tokenizer", str(self.tokenizer_path), "--repeat-penalty", "1",
                   "inference", "--config", str(self.model_path),
                   "--temperature", "0", "--prompt", config.prompt]
        if config.two_stage_argmax:
            command += ["--two-stage-argmax", "--group-size", str(config.group_size)]
        env = {k: v for k, v in os.environ.items()
               if k not in ("RAYON_NUM_THREADS", "CANDLE_CPU_MASK")}
        env.update(RAYON_NUM_THREADS=str(config.threads), CANDLE_CPU_MASK=mask)
        completed = subprocess.run(command, env=env, capture_output=True, text=True,
                                   check=True, timeout=config.timeout)
        match = re.search(r"\(([\d.]+) token/s\)", completed.stdout)
        if not match or "starting the inference loop\n" not in completed.stdout:
            raise RuntimeError(f"Candle benchmark output not found:\n{completed.stdout}\n{completed.stderr}")
        generated = completed.stdout.split("starting the inference loop\n", 1)[1].rsplit("\n", 3)[0]
        digest = hashlib.md5(generated.encode()).hexdigest()[:12]
        if self.expected_hash is None:
            self.expected_hash = digest
        elif digest != self.expected_hash:
            raise RuntimeError(f"generated text changed: expected {self.expected_hash}, got {digest}")
        result = BenchmarkResult(config, float(match.group(1)), completed.stderr)
        if self.results_path:
            with self.results_path.open("a") as output:
                output.write(json.dumps({"cpus": config.cpus, "threads": config.threads,
                                         "group_size": config.group_size,
                                         "two_stage_argmax": config.two_stage_argmax,
                                         "tok_s": result.tokens_per_second,
                                         "hash": digest}) + "\n")
        return result


class CandleTuner(AutoTuner):
    def _make_config(self, config, mask):
        return replace(config, cpus=mask, threads=len(mask))

    def _tuning_candidates(self, config, masks, groups):
        return [replace(self._make_config(config, mask), group_size=group)
                for mask in masks for group in dict.fromkeys(groups)]


def main():
    parser = argparse.ArgumentParser(description="Tune Candle llama2-c CPU affinity and int8 group size")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--cpus", help="initial CPU list, e.g. 0,2,4-10; default is current affinity")
    parser.add_argument("--group-size", type=int, default=144)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--budget", type=int, default=56)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    topology = CPUTopology()
    from .topology import parse_cpu_list
    cpus = tuple(parse_cpu_list(args.cpus)) if args.cpus else tuple(topology.cpus)
    runner = CandleRunner(args.binary, args.model, args.tokenizer, args.results)
    tuner = CandleTuner(topology, runner)
    config = CandleConfig(cpus, len(cpus), args.prompt, args.group_size, timeout=args.timeout)
    best = tuner.kasikoi_serch(config, args.budget, args.top_k, args.repeats, args.seed)
    print(f"best: {best.tokens_per_second:.2f} tok/s, CPUs={best.config.cpus}, "
          f"threads={best.config.threads}, group_size={best.config.group_size}")
    print(f"search: {tuner.last_search['measurements']} runs, "
          f"{tuner.last_search['seconds']:.1f} s, text hash={runner.expected_hash}")


if __name__ == "__main__":
    main()
