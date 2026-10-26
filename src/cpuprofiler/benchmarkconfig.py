"""benchmarkとbenchmark_distributeで共用する実行設定。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int
    output: Path | None = None  # benchmark_distributeでは保存先を指定する
    binary: Path = ROOT / "build/run-fused"
    model: Path = ROOT / "build/stories15M.bin"
    tokenizer: Path = ROOT / "tokenizer.bin"
    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"
    omp_dynamic: bool = False
    omp_proc_bind: str = "close"
    omp_places: str | None = None  # Noneならcpusの順序で作る
    omp_schedule: str = "dynamic,16"  # 出力を含む全matmul共通
    omp_wait_policy: str | None = None  # None=未指定、ACTIVE/PASSIVE
    gomp_spincount: int | None = None  # None=未指定、0ならスピン待機なし
    persistent_openmp: bool = False  # 生成ループ全体で1つのOpenMPチームを使う
    output_threads: int | None = None  # 常駐時の出力人数。Noneならthreadsと同数
    optimize_weights: bool = False  # Trueなら重みを2MiB境界に揃えTHPを要求
    two_stage_argmax: bool = False  # -a 1、greedy生成用
    group_size: int = 144  # -g、モデルのdimを割り切る値
    timeout: float = 300
    build: bool = True  # 実行前に現在のrun.cをコンパイルする

    def __post_init__(self):
        if self.omp_wait_policy is not None and self.omp_wait_policy.upper() not in ("ACTIVE", "PASSIVE"):
            raise ValueError("omp_wait_policy must be ACTIVE, PASSIVE or None")
        if self.gomp_spincount is not None and (not isinstance(self.gomp_spincount, int) or self.gomp_spincount < 0):
            raise ValueError("gomp_spincount must be a nonnegative integer or None")
        if self.output_threads is not None and (not isinstance(self.output_threads, int) or self.output_threads < 1):
            raise ValueError("output_threads must be a positive integer or None")
        if self.output_threads is not None and not self.persistent_openmp:
            raise ValueError("output_threads requires persistent_openmp=True")
        if self.persistent_openmp and (self.omp_dynamic or self.threads < 1
                or len(set(self.cpus)) != len(self.cpus)
                or max(self.threads, self.output_threads or self.threads) > len(self.cpus)):
            raise ValueError("persistent_openmp needs fixed thread counts within config.cpus")
