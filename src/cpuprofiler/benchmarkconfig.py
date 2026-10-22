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
    optimize_weights: bool = False  # Trueなら重みを2MiB境界に揃えTHPを要求
    timeout: float = 300
    build: bool = True  # 実行前に現在のrun.cをコンパイルする
