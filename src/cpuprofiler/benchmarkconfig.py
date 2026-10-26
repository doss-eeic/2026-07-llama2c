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
    workers: str = "omp"  # -x、omp=OpenMP、pool=cpusのCPUごとに常駐スレッド（threadsとOMP_*は効かない）
    # -b / -c、出力層以外と出力層のCPU（両方を指定する。スレッド数はそれぞれのCPUの数、cpusは両方を合わせたもの）
    body_cpus: tuple[int, ...] | None = None
    cls_cpus: tuple[int, ...] | None = None
    timeout: float = 300

    def __post_init__(self):
        if self.omp_wait_policy is not None and self.omp_wait_policy.upper() not in ("ACTIVE", "PASSIVE"):
            raise ValueError("omp_wait_policy must be ACTIVE, PASSIVE or None")
        if self.gomp_spincount is not None and (not isinstance(self.gomp_spincount, int) or self.gomp_spincount < 0):
            raise ValueError("gomp_spincount must be a nonnegative integer or None")
        if self.workers not in ("omp", "pool"):
            raise ValueError("workers must be omp or pool")
        if (self.body_cpus is None) != (self.cls_cpus is None):
            raise ValueError("body_cpus and cls_cpus must be given together")
        if self.body_cpus is not None and not (self.body_cpus and self.cls_cpus
                                               and set(self.body_cpus) | set(self.cls_cpus) <= set(self.cpus)):
            raise ValueError("body_cpus and cls_cpus must be nonempty and within cpus")

    @property
    def uses_sections(self) -> bool:
        # 出力層とそれ以外で CPU を分けるか（-b / -c）
        return self.body_cpus is not None
