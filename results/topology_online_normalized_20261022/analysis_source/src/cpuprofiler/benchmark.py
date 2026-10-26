import re
import subprocess

def test_benchmark():
    # シンプルなテストを回すための関数。cpu4を使ってベンチマークを実行し、tpsを取得する。
    result = subprocess.run(
        [
            "taskset", "-c", "4",
            "./build/run-omp", "./build/stories15M.bin",
            "-n", "256",
            "-t", "0",
            "-s", "42",
            "-i", "Once upon a time",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"achieved tok/s:\s*([\d.]+)",
        result.stderr
    )

    if match is None:
        raise RuntimeError("Benchmark result not found")

    tps = float(match.group(1))
    print(f"{tps:.2f} tokens/s")

import os
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int

    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"

    omp_dynamic: bool = False
    omp_proc_bind: str = "TRUE"
    omp_places: str = "threads"
    output_threads: int = 8
    omp_schedule: str = "dynamic,128"


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    tokens_per_second: float
    raw_stderr: str

    @property
    def ms_per_token(self):
        return 1000.0 / self.tokens_per_second


class BenchmarkRunner:
    def __init__(
        self,
        binary_path: str = "./build/run-omp",
        model_path: str = "./build/stories15M.bin"
    ):
        self.binary_path = Path(binary_path).resolve()
        self.model_path = Path(model_path).resolve()

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        if not config.cpus:
            raise ValueError("CPU list must not be empty")

        if config.threads < 1:
            raise ValueError("threads must be positive")

        if config.output_threads < 1:
            raise ValueError("output_threads must be positive")

        # OpenMPの環境変数
        env = os.environ.copy()

        env["OMP_NUM_THREADS"] = str(config.threads)
        env["OUTPUT_THREADS"] = str(config.output_threads)
        env["OMP_SCHEDULE"] = config.omp_schedule
        env["OMP_DYNAMIC"] = (
            "TRUE" if config.omp_dynamic else "FALSE"
        )
        env["OMP_PROC_BIND"] = config.omp_proc_bind
        env["OMP_PLACES"] = config.omp_places

        # CPU affinity
        cpu_list = ",".join(map(str, config.cpus))

        command = [
            "taskset", "-c", cpu_list,
            str(self.binary_path),
            str(self.model_path),
            "-n", str(config.steps),
            "-t", str(config.temperature),
            "-s", str(config.seed),
            "-i", config.prompt
        ]

        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # llama2.cの標準出力から性能を取得
        match = re.search(
            r"achieved tok/s:\s*([\d.]+)",
            result.stderr
        )

        if match is None:
            raise RuntimeError(
                "Benchmark result not found:\n"
                + result.stderr
            )

        return BenchmarkResult(
            config=config,
            tokens_per_second=float(match.group(1)),
            raw_stderr=result.stderr
        )

if __name__ == "__main__":
    runner = BenchmarkRunner()

    config = BenchmarkConfig(
        cpus=(0, 1, 2, 3, 4, 5, 8, 9),
        threads=4,
        output_threads=8,
        omp_proc_bind="close",
        omp_places="{0},{1},{2},{3},{4},{5},{8},{9}"
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)

    config = BenchmarkConfig(
        cpus=(0, 1),
        threads=2,
        output_threads=2
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)

    
