"""benchmarkとautotunerの実行設定。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int
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
    optimize_weights: bool = False  # Trueなら重みを2MiB境界に揃えTHPを要求
    two_stage_argmax: bool = False  # -a 1、greedy生成用
    group_size: int = 144  # -g、モデルのdimを割り切る値
    timeout: float = 300

    def __post_init__(self):
        if not self.cpus or len(set(self.cpus)) != len(self.cpus) or any(c < 0 for c in self.cpus):
            raise ValueError("cpus must contain unique nonnegative CPU ids")
        if self.threads < 1 or self.threads > len(self.cpus):
            raise ValueError("threads must be between 1 and len(cpus)")
        if self.steps < 2 or self.timeout <= 0:
            raise ValueError("steps must be at least 2 and timeout must be positive")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.two_stage_argmax and self.temperature != 0:
            raise ValueError("two_stage_argmax requires temperature=0")
        if self.omp_wait_policy is not None and self.omp_wait_policy.upper() not in ("ACTIVE", "PASSIVE"):
            raise ValueError("omp_wait_policy must be ACTIVE, PASSIVE or None")
        if self.gomp_spincount is not None and (not isinstance(self.gomp_spincount, int) or self.gomp_spincount < 0):
            raise ValueError("gomp_spincount must be a nonnegative integer or None")
