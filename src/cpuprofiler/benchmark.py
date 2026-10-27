import math
import os
import re
import subprocess
from dataclasses import dataclass

from .benchmarkconfig import BenchmarkConfig


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    tokens_per_second: float
    raw_stderr: str

    @property
    def ms_per_token(self):
        return 1000.0 / self.tokens_per_second


def execution(config: BenchmarkConfig):
    """Shared command/environment for measurements and exported launchers."""
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

    if config.omp_wait_policy is not None:
        env["OMP_WAIT_POLICY"] = config.omp_wait_policy.upper()
    if config.gomp_spincount is not None:
        env["GOMP_SPINCOUNT"] = str(config.gomp_spincount)

    # CPU affinity
    cpu_list = ",".join(map(str, config.cpus))

    command = [
        "taskset", "-c", cpu_list,
        str(config.binary),
        str(config.model),
        "-n", str(config.steps),
        "-t", str(config.temperature),
        "-s", str(config.seed),
        "-i", config.prompt,
        "-z", str(config.tokenizer)
    ]

    # upstreamは-w非対応。既定のmmapはオプションを渡さず実行する。
    if config.optimize_weights:
        command += ["-w", "1"]
    if config.two_stage_argmax:
        command += ["-a", "1", "-g", str(config.group_size)]

    return command, env


class BenchmarkRunner:
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        command, env = execution(config)
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=config.timeout,
        )

        # llama2.cの標準エラー出力から性能を取得
        match = re.search(
            r"achieved tok/s:\s*([\d.]+)",
            result.stderr
        )

        if match is None:
            raise RuntimeError(
                "Benchmark result not found:\n"
                + result.stderr
            )

        tps = float(match.group(1))
        if not math.isfinite(tps) or tps <= 0:
            raise RuntimeError("Benchmark throughput must be finite and positive")

        return BenchmarkResult(
            config=config,
            tokens_per_second=tps,
            raw_stderr=result.stderr
        )
