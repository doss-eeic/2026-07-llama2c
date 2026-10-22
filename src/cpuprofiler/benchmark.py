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
    # 全体の cpu 指定（taskset）。body_cpus / cls_cpus を使うときは、指定しなかった区間がこれを使う。
    # 省略すると body_cpus と cls_cpus を合わせたものになる
    cpus: tuple[int, ...] = ()
    # 全体のスレッド数（OMP_NUM_THREADS）。省略すると len(cpus)
    threads: int = 0

    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"

    omp_dynamic: bool = False
    omp_proc_bind: str = "TRUE"
    omp_places: str = "threads"
    output_threads: int = 8
    omp_schedule: str = "dynamic,128"

    # 出力層以外のcpu指定（run.c の -B）。スレッド数は len(body_cpus)
    body_cpus: tuple[int, ...] | None = None
    # 出力層のcpu指定（run.c の -C）。スレッド数は len(cls_cpus)
    cls_cpus: tuple[int, ...] | None = None

    @property
    def uses_sections(self) -> bool:
        # 出力層とそれ以外で cpu を分けるか
        return self.body_cpus is not None or self.cls_cpus is not None

    @property
    def all_cpus(self) -> tuple[int, ...]:
        # taskset に渡す cpu
        if self.cpus:
            return self.cpus
        merged = set(self.body_cpus or ()) | set(self.cls_cpus or ())
        return tuple(sorted(merged))

    @property
    def all_threads(self) -> int:
        # OMP_NUM_THREADS に渡すスレッド数
        return self.threads if self.threads > 0 else len(self.all_cpus)


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    tokens_per_second: float
    raw_stderr: str

    # 区間ごとの時間 (ms/token)。区間タイマーのないバイナリでは None
    body_ms_per_token: float | None = None  # 出力層以外
    cls_ms_per_token: float | None = None   # 出力層

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
        cpus = config.all_cpus
        threads = config.all_threads

        if not cpus:
            raise ValueError("CPU list must not be empty")

        if threads < 1:
            raise ValueError("threads must be positive")

        if config.output_threads < 1:
            raise ValueError("output_threads must be positive")

        for section in (config.body_cpus, config.cls_cpus):
            if section is not None and not section:
                raise ValueError("body_cpus / cls_cpus must not be empty")

        # OpenMPの環境変数
        env = os.environ.copy()

        env["OMP_NUM_THREADS"] = str(threads)
        env["OUTPUT_THREADS"] = str(config.output_threads)
        env["OMP_SCHEDULE"] = config.omp_schedule
        env["OMP_DYNAMIC"] = (
            "TRUE" if config.omp_dynamic else "FALSE"
        )
        if config.uses_sections:
            # -B / -C では run.c がスレッドを cpu に固定するので、OpenMP には固定させない
            env["OMP_PROC_BIND"] = "FALSE"
            env.pop("OMP_PLACES", None)
        else:
            env["OMP_PROC_BIND"] = config.omp_proc_bind
            env["OMP_PLACES"] = config.omp_places

        # CPU affinity
        cpu_list = ",".join(map(str, cpus))

        command = [
            "taskset", "-c", cpu_list,
            str(self.binary_path),
            str(self.model_path),
            "-n", str(config.steps),
            "-t", str(config.temperature),
            "-s", str(config.seed),
            "-i", config.prompt
        ]

        # 区間ごとの cpu 指定（指定した区間だけ渡す）
        if config.body_cpus is not None:
            command += ["-B", ",".join(map(str, config.body_cpus))]
        if config.cls_cpus is not None:
            command += ["-C", ",".join(map(str, config.cls_cpus))]

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

        # 区間タイマーの結果（出力層以外 / 出力層）を取得
        section = re.search(
            r"section ms/tok:\s*body=([\d.]+)\s+cls=([\d.]+)",
            result.stderr
        )

        return BenchmarkResult(
            config=config,
            tokens_per_second=float(match.group(1)),
            raw_stderr=result.stderr,
            body_ms_per_token=float(section.group(1)) if section else None,
            cls_ms_per_token=float(section.group(2)) if section else None
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

    # 出力層とそれ以外で cpu を分ける
    config = BenchmarkConfig(
        body_cpus=(0, 2),
        cls_cpus=(4, 5, 6, 7)
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.body_ms_per_token, result.cls_ms_per_token)

    
