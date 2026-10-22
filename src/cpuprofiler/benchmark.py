import re
import subprocess

def test_benchmark():
    # シンプルなテストを回すための関数。cpu4を使ってベンチマークを実行し、tpsを取得する。
    result = subprocess.run(
        [
            "taskset", "-c", "4",
            "./build/run-fused", "./build/stories15M.bin",
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
from dataclasses import dataclass, replace


if __package__:
    from .benchmarkconfig import BenchmarkConfig
else:
    from benchmarkconfig import BenchmarkConfig


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
        binary_path: str = "./build/run-fused",
        model_path: str = "./build/stories15M.bin"
    ):
        self.binary_path = Path(binary_path).resolve()
        self.model_path = Path(model_path).resolve()

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        if not config.cpus:
            raise ValueError("CPU list must not be empty")

        if config.threads < 1:
            raise ValueError("threads must be positive")

        # OpenMPの環境変数
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMP_", "GOMP_", "OUTPUT_"))}

        env["OMP_NUM_THREADS"] = str(config.threads)
        env["OMP_SCHEDULE"] = config.omp_schedule
        env["OMP_DYNAMIC"] = (
            "TRUE" if config.omp_dynamic else "FALSE"
        )
        env["OMP_PROC_BIND"] = config.omp_proc_bind
        env["OMP_PLACES"] = config.omp_places or ",".join(f"{{{cpu}}}" for cpu in config.cpus)

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

        # upstreamは-w非対応。既定のmmapはオプションを渡さず実行する。
        if config.optimize_weights:
            command += ["-w", "1"]

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

def main():
    runner = BenchmarkRunner(binary_path="./build/run-upstream")
    print("\n==0,1,2,3,4,5,8,9を4threads==\n")
    config = BenchmarkConfig(
        cpus=(0, 1, 2, 3, 4, 5, 8, 9),
        threads=4,
        omp_proc_bind="close",
        omp_places="{0},{1},{2},{3},{4},{5},{8},{9}"
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)

    print("\n==0,1を2threads==\n")
    config = BenchmarkConfig(
        cpus=(0, 1),
        threads=2,
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)

    print("\n==0,2,4,5,6,7,8,9,10,11を10threads==\n")
    config = BenchmarkConfig(
        cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        threads=10,
        omp_proc_bind="close",
        omp_places="{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}",
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)

    print("\n==0,2,4,5,6,7,8,9,10,11を10threads==\n")
    config = BenchmarkConfig(
        cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        threads=10,
    )

    result = runner.run(config)

    print(result.tokens_per_second)
    print(result.ms_per_token)


def main2():
    #ここで，ffn,qkvの最適化のベンチマーク比較をする。
    runners = {
        "upstream": BenchmarkRunner(binary_path="./build/run-upstream"), #これが未改造
        "fused": BenchmarkRunner(binary_path="./build/run-fused"), #これがffn,qkvの融合，並列実行版
    }

    # 層内・出力層とも同じ10スレッドで比較
    config = BenchmarkConfig(
        cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        threads=10,
        omp_schedule="dynamic,16",  # 出力を含む全matmul共通
        omp_proc_bind="close",
        omp_places="{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}",
    )

    print(f"threads={config.threads}, OMP_SCHEDULE={config.omp_schedule}")

    # 各バイナリをウォームアップ
    for runner in runners.values():
        runner.run(config)

    results = {name: [] for name in runners}

    # 実行順を交互にして10回ずつ測定
    for repeat in range(10):
        order = (
            ("upstream", "fused")
            if repeat % 2 == 0
            else ("fused", "upstream")
        )

        for name in order:
            result = runners[name].run(config)
            results[name].append(result.tokens_per_second)
            print(f"{repeat + 1:02d} {name:8s}: "
                  f"{result.tokens_per_second:.2f} tok/s")

    means = {
        name: sum(values) / len(values)
        for name, values in results.items()
    }

    print("\n平均")
    for name, tps in means.items():
        print(f"{name:8s}: {tps:.2f} tok/s")

    gain = (means["fused"] / means["upstream"] - 1) * 100
    print(f"融合版の改善率: {gain:+.2f}%")

def main3():
    # 同じ融合版の重み最適化False/Trueを比較し、未改造版も参考に測る。
    fused = BenchmarkRunner(binary_path="./build/run-fused")
    config = BenchmarkConfig(
        cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        threads=10,
        omp_schedule="dynamic,16",
        omp_proc_bind="close",
        omp_places="{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}",
    )
    cases = {
        "upstream": (BenchmarkRunner(binary_path="./build/run-upstream"), config),
        "fused": (fused, config),
        "fused+weights": (fused, replace(config, optimize_weights=True)),
    }
    print(f"threads={config.threads}, OMP_SCHEDULE={config.omp_schedule}", flush=True)
    for runner, case_config in cases.values():
        runner.run(case_config)

    results = {name: [] for name in cases}
    names = list(cases)
    for repeat in range(10):
        # 各条件が先頭・中間・最後になるよう順番を回す。
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            runner, case_config = cases[name]
            result = runner.run(case_config)
            results[name].append(result.tokens_per_second)
            print(f"{repeat + 1:02d} {name:13s}: "
                  f"{result.tokens_per_second:.2f} tok/s", flush=True)

    means = {name: sum(values) / len(values) for name, values in results.items()}
    print("\n平均")
    for name, tps in means.items():
        print(f"{name:13s}: {tps:.2f} tok/s")
    gain = (means["fused+weights"] / means["fused"] - 1) * 100
    total = (means["fused+weights"] / means["upstream"] - 1) * 100
    print(f"重み最適化の追加改善率: {gain:+.2f}%")
    print(f"未改造版からの合計改善率: {total:+.2f}%")


if __name__ == "__main__":
    main3()
