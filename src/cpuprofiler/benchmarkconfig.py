"""benchmark_distributeへ渡す実行設定。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int
    output: Path
    binary: Path = ROOT / "build/run-omp"
    model: Path = ROOT / "build/stories15M.bin"
    tokenizer: Path = ROOT / "tokenizer.bin"
    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"
    omp_dynamic: bool = False
    omp_proc_bind: str = "close"
    omp_places: str | None = None  # Noneならcpusの順序で作る
    omp_schedule: str = "static"  # 通常部分用
    output_threads: int = 8
    output_schedule: str = "dynamic"
    output_chunk: int = 128  # staticで0ならOpenMP既定の分割
    timeout: float = 300
    build: bool = True  # 実行前に現在のrun.cをコンパイルする
