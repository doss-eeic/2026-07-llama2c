"""詳細benchmarkへ渡す1回分の実行設定。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int
    output: Path  # autotunerが実行ごとに別の保存先を指定する
    binary: Path = ROOT / "build/omp-run-detail"
    model: Path = ROOT / "build/stories15M.bin"
    tokenizer: Path = ROOT / "tokenizer.bin"
    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"
    omp_dynamic: bool = False
    omp_proc_bind: str = "close"
    omp_places: str | None = None  # Noneならcpusから作る
    skip_tokens: int = 1
    timeout: float = 300
    stages: tuple[str, ...] | None = None


@dataclass
class StayBenchmarkConfig:
    """外部のCPUポリシーでstayを1回実行する設定。"""

    policy: Path
    output: Path
    binary: Path = ROOT / "build/stay-run-detail"
    model: Path = ROOT / "build/stories15M.bin"
    tokenizer: Path = ROOT / "tokenizer.bin"
    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"
    skip_tokens: int = 1
    timeout: float = 300
    stages: tuple[str, ...] | None = None
    timing_only: bool = False  # Trueならforward_totalのみ計測
