"""benchmark_distributeへ渡す実行設定。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Pinned upstream source; git show extracts the unmodified file for each build.
UPSTREAM_REVISION = "350e04fe35433e6d2941dce5a1f53308f87058eb"


@dataclass(frozen=True)
class BenchmarkConfig:
    cpus: tuple[int, ...]
    threads: int  # cpus先頭の何人を層内で使うか
    output: Path
    binary: Path = ROOT / "build/stay-run"
    model: Path = ROOT / "build/stories15M.bin"
    tokenizer: Path = ROOT / "tokenizer.bin"
    steps: int = 256
    temperature: float = 0.0
    seed: int = 42
    prompt: str = "Once upon a time"
    p_weight: int = 2  # outputで層内参加CPUに割り当てる重み（残りは1）
    timeout: float = 300
    build: bool = True  # 実行前に現在のrun.cをコンパイルする
    backend: str = "stay"  # stay / upstream（未改造）/ parallel（統合OpenMP）
    profile: bool = False  # Trueなら層・演算別計測を有効にする
    skip_tokens: int = 1
    omp_schedule: str = "dynamic,16"  # parallel backend only
    output_threads: int | None = None
    output_chunk: int = 32
    fuse: bool = True
    attention_threads: int = 0  # 0なら層内のOpenMPチーム人数を維持する
    output_block: int = 1  # parallel: 1行または4行ずつ計算する
    fuse_qkv: bool | None = None  # Noneならfuseを継承
    fuse_ffn: bool | None = None
    output_dynamic: bool | None = None  # 実験backendでは通常部分の分配を継承
