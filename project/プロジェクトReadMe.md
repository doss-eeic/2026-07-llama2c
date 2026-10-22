# CPU-Profiler

全1〜14論理CPUの最新実測と5探索法の比較は
[14CPU探索の記録](docs/AI向け_20261022_14CPU探索.md)、
[実行仕様](../src/cpuprofiler/TOPOLOGY14.md)を参照。

2026-10-22の実測・構成探索の成果は
[480候補の構成探索と追加ベンチマーク](docs/AI向け_20261022_構成探索.md)、
[実装と再現方法](../src/cpuprofiler/TOPOLOGY_SEARCH.md)を参照。

> CPUの構成(コアがなんこ，どのスレッドがL2キャッシュを共有しているか等)を認識
> ->色々な組み合わせでベンチマークを走らせて集計
> ->最適な構成を決定
> ->以後runしたらその構成でrunするようになる。

というのが簡単な目標

## やること

### PoC
- CPUの構成を取得，適切な構造(これ決める)で取得
- まず最初はコア１つ(複数組み合わせない)のベンチマークを取る。
- 一番早いのを選ぶ。(もしくは一番早い２つを組み合わせるのを基本構成に。)
- runをするときに自動でこの構成が動くようにする。(とりあえずはmakeファイルで専用の作るのが良いかも)

# cpu-profiler.py
## 仕組み
<details>
<summary>ディレクトリの構成はこのようになっていることが多い。</summary>

```shell
/sys/devices/system/cpu/
│
├── online
├── present
├── possible
│
├── cpu0/
│   ├── topology/
│   │   ├── physical_package_id
│   │   ├── core_id
│   │   ├── core_cpus_list
│   │   ├── thread_siblings_list
│   │   ├── package_cpus_list
│   │   └── ...
│   │
│   ├── cache/
│   │   ├── index0/
│   │   │   ├── level
│   │   │   ├── type
│   │   │   ├── size
│   │   │   └── shared_cpu_list
│   │   ├── index1/
│   │   ├── index2/
│   │   └── ...
│   │
│   └── cpufreq/
│       ├── cpuinfo_max_freq
│       ├── scaling_max_freq
│       └── scaling_cur_freq
│
├── cpu1/
├── cpu2/
├── ...
├── cpu13/
│
└── cpufreq/
    ├── policy0/
    ├── policy2/
    └── ...
```
</details>

### 論理CPU
```shell
cat /sys/devices/system/cpu/online
```
```shell
0-13
```
<details>
<summary> なお，表示には以下のようなOptionがある</summary>

| ファイル | 内容 |
|---|---|
| `possible` | カーネルが扱えるCPU番号の範囲 |
| `present` | 現在システムに存在すると認識されるCPU |
| `online` | 現在稼働しているCPU |

```shell
possible: 0-31
present:  0-13
online:   0-13
```

なお，オンラインであれば全てが利用できるわけではない。利用が許可されているかは以下で取得できる。

```python
import os

print(os.sched_getaffinity(0))
```


</details>

### 物理コア
`/topology`フォルダの中には，

```shell
ls /sys/devices/system/cpu/cpu0/topology/
```
```shell
cluster_cpus       cluster_id  core_cpus_list  core_siblings       die_cpus       die_id        package_cpus_list    thread_siblings
cluster_cpus_list  core_cpus   core_id         core_siblings_list  die_cpus_list  package_cpus  physical_package_id  thread_siblings_list
```
このようなファイルが有りそれぞれは以下のような役割を持つ。

| ファイル | 内容 | 例 |
|---|---|---|
| `physical_package_id` | CPUソケットの識別ID | `0` |
| `core_id` | 物理コアの識別ID | `0` |
| `core_cpus_list` | 同じ物理コアに属する論理CPU | `0-1` |
| `thread_siblings_list` | 同上（旧名称） | `0-1` |
| `package_cpus_list` | 同じCPUパッケージに属する論理CPU | `0-13` |
| `die_id` | CPUダイの識別ID | 環境依存 |
| `cluster_id` | カーネルが認識するクラスタID | 環境依存 |


```shell
cat /sys/devices/system/cpu/cpu0/topology/core_id
cat /sys/devices/system/cpu/cpu0/topology/core_cpus_list
# 結果が以下
0
0-1
```
の場合はCPU 0は物理コアID 0に属する。CPU 0と1は同じ物理コアのSMTスレッド


## キャッシュ構造：cache/
各CPUフォルダの/cacheの中身は
```shell
ls /sys/devices/system/cpu/cpu4/cache/
```
```
index0
index1
index2
index3
...
```

それぞれがキャッシュの一つの情報エントリ。

```shell
cat /sys/devices/system/cpu/cpu4/cache/index2/level
cat /sys/devices/system/cpu/cpu4/cache/index2/type
cat /sys/devices/system/cpu/cpu4/cache/index2/size
cat /sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_list
#結果が以下
2
Unified
2048K
4-7
```
例えば、このようになっているなら、
CPU 4〜7は2MBのL2キャッシュを共有していることが分かる。
| ファイル | 内容 |
|---|---|
| `level` | L1 / L2 / L3 |
| `type` | Data / Instruction / Unified |
| `size` | キャッシュ容量 |
| `shared_cpu_list` | キャッシュを共有する論理CPU集合 |
| `id` | キャッシュインスタンスの識別ID |
| `coherency_line_size` | キャッシュラインサイズ |


注意：index2が必ずL2とは限らない

# benchmarkrunner.py

CPUクラスを用いて，実際にpythonコードからshellを投げてベンチマークを取ってみる。そのまま分析もできるからPythonで一貫したい。

## 基本コマンド
例えば、CPU 4だけでllama2.cを実行する場合。
```shell
OMP_NUM_THREADS=1 \
OMP_DYNAMIC=FALSE \
taskset -c 4 \
./build/omp-run ./build/stories15M.bin \
    -n 256 \
    -t 0 \
    -s 42 \
    -i "Once upon a time"
```

| 設定 | 意味 |
|---|---|
| `OMP_NUM_THREADS=1` | OpenMPのスレッド数を1に固定 |
| `OMP_DYNAMIC=FALSE` | OpenMPによるスレッド数の自動調整を無効化 |
| `taskset -c 4` | CPU 4のみでプログラムを実行 |
| `./run` | llama2.cの実行ファイル |
| `./stories15M.bin` | モデル |
| `-n 256` | 最大256トークンまで実行 |
| `-t 0` | Greedy decoding |
| `-s 42` | 乱数シードを固定 |
| `-i` | 入力プロンプト |

> taskset はプロセスのCPU affinityを設定するLinuxのコマンド。

注意：今のビルドが`make runomp CC=gcc`でできているか確認。

## ベンチマークの取得

今回は各工程の最適化を取らないため，ttsのみをベンチマークとする。
```shell
OMP_NUM_THREADS=1 \
OMP_DYNAMIC=FALSE \
taskset -c 4 \
./build/omp-run ./build/stories15M.bin \
    -n 256 \
    -t 0 \
    -s 42 \
    -i "Once upon a time" \
    > benchmark/test-output.txt \
    2> benchmark/test-benchmark.log
```

## pythonから走らせる
```python
import re
import subprocess

result = subprocess.run(
    [
        "taskset", "-c", "4",
        "./build/omp-run", "./build/stories15M.bin",
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
```

## BenchMarkを管理する。
このようにBenchmarkとTopologyの両方を統括する形でAutoTunerを作った。現状の仕様では全くAutoではなく，決められたConfigのBenchmarkを全て実行するだけである。

<details>
<summary>Pythonコードはこちら</summary>

```python
from .benchmark import BenchmarkConfig, BenchmarkRunner
from .topology import CPUTopology

class AutoTuner:
    def __init__(self, topology: CPUTopology, runner: BenchmarkRunner):
        self.topology = topology
        self.runner = runner
        self.history = []

    def evaluate(self, config):
        result = self.runner.run(config)
        self.history.append(result)
        return result

    def search(self, args="Sample"):
        candidates = self.generate_fixed_candidates(args)

        best = None

        for config in candidates:
            result = self.evaluate(config)

            if (
                best is None
                or result.tokens_per_second > best.tokens_per_second
            ):
                best = result

        return best

    def generate_fixed_candidates(self, args="All"):
        if args == "All":
            return [BenchmarkConfig((cpu_id,), 1) for cpu_id in self.topology.cpus]
        elif args == "Sample":
            return [
                BenchmarkConfig((0,), 1),
                BenchmarkConfig((0, 2), 2),
                BenchmarkConfig((0, 1, 2, 3), 4),
                BenchmarkConfig((4, 5, 8, 9), 4),
                BenchmarkConfig((0, 2, 4, 8), 4),
            ]
        else:
            raise ValueError("Invalid argument for generate_fixed_candidates")
```

</details>

## BenchMarkを動的に探索する。
今回一番シンプルな探索を行う。まず前提としてthread数は論理CPUの数と一致させる。
### 仕組み(ビームサーチ)
Epoch`k`の結果からEpoch`k+1`のCandidatesへの推移という形で探索を進める。

Epoch`k`のResultで早かった上位３つのConfigに対して，全種類のCPUを追加したものをCandidatesとして，Benchmarkを実行する。

### Epoch1
まず最初に以下のように，論理CPU１つで行う。
```
BenchmarkConfig(cpus=(0,),threads=1)
BenchmarkConfig(cpus=(1,),threads=1)
...
BenchmarkConfig(cpus=(13,),threads=1)
```
結果として`BenchmarkConfig(cpus=(2,),threads=1)`，`BenchmarkConfig(cpus=(3,),threads=1)`，`BenchmarkConfig(cpus=(1,),threads=1)`という順位になった。
### Epoch2
以下のようなCnadidatesを作り実行する。
```
# CPU2を軸としたビーム
BenchmarkConfig(cpus=(2,0),threads=2)
BenchmarkConfig(cpus=(2,1),threads=2)
...
BenchmarkConfig(cpus=(2,13),threads=2)

# CPU3を軸としたビーム
BenchmarkConfig(cpus=(3,0),threads=2)
BenchmarkConfig(cpus=(3,1),threads=2)
...
BenchmarkConfig(cpus=(3,13),threads=2)

# CPU1を軸としたビーム
BenchmarkConfig(cpus=(1,0),threads=2)
BenchmarkConfig(cpus=(1,1),threads=2)
...
BenchmarkConfig(cpus=(1,13),threads=2)
```

### 結果
```txt
Stage 1: 14 candidates
│
├── (2,) 190.52 ───┐
├── (3,) 189.21 ───┤ Top 3
└── (1,) 187.93 ───┘
          │
          ▼
Stage 2: 36 candidates
│
├── (1,2) 251.42 ──┐
├── (1,3) 250.28 ──┤ Top 3
└── (0,2) 249.44 ──┘
          │
          ▼
Stage 3: 34 candidates
│
├── (0,1,3) 256.08 ──┐
├── (0,1,2) 255.49 ──┤ Top 3
└── (1,2,3) 246.10 ──┘
          │
          ▼
Stage 4: 31 candidates
│
├── (0,1,2,3) 287.01  ★ BEST
├── (0,1,3,10) 244.47
├── (0,1,3,8)  243.39
└── ...
```

<details>
<summary>探索のログはこちら</summary>
```txt
Running recursive benchmark...
Configuration: BenchmarkConfig(cpus=(0,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 187.765506 tokens/sec
Configuration: BenchmarkConfig(cpus=(1,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 187.92517 tokens/sec
Configuration: BenchmarkConfig(cpus=(2,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 190.517241 tokens/sec
Configuration: BenchmarkConfig(cpus=(3,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 189.212329 tokens/sec
Configuration: BenchmarkConfig(cpus=(4,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 84.902036 tokens/sec
Configuration: BenchmarkConfig(cpus=(5,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 84.609495 tokens/sec
Configuration: BenchmarkConfig(cpus=(6,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 84.869432 tokens/sec
Configuration: BenchmarkConfig(cpus=(7,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 84.67433 tokens/sec
Configuration: BenchmarkConfig(cpus=(8,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 85.09819 tokens/sec
Configuration: BenchmarkConfig(cpus=(9,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 85.065435 tokens/sec
Configuration: BenchmarkConfig(cpus=(10,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 85.032705 tokens/sec
Configuration: BenchmarkConfig(cpus=(11,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 84.739264 tokens/sec
Configuration: BenchmarkConfig(cpus=(12,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 64.866451 tokens/sec
Configuration: BenchmarkConfig(cpus=(13,), threads=1, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 64.638783 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 7), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 141.939627 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 7), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 135.582822 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 9), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 137.011779 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 8), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 142.122186 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 3), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 200.181159 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 250.283126 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 12), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:110.721443 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 4), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 140.050697 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 7), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 141.757537 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 5), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 140.228426 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 13), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:114.152893 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 13), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:113.333333 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 3), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 246.927374 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 8), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 140.495868 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 9), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 142.122186 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 6), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 136.167591 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 11), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:137.011779 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 9), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 141.757537 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 8), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 136.419753 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 11), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:140.943878 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 13), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:111.055276 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 4), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 140.495868 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 6), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 137.780549 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 251.422071 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 10), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:142.122186 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 249.435666 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 12), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:111.616162 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 11), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:142.213642 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 5), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 138.644918 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 10), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:136.92689 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 5), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 136.757426 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 4), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 135.749386 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 198.384201 tokens/sec
Configuration: BenchmarkConfig(cpus=(3, 10), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:139.168766 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 12), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance:113.043478 tokens/sec
Configuration: BenchmarkConfig(cpus=(2, 6), threads=2, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 141.394754 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 6), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 182.94702 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 3), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 245.828699 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 13), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 148.521505 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 13), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 150.956284 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 5), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.554817 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 246.10245 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 4), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.320267 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 11), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 185.247276 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 4), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.474124 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 12), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 149.324324 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 7), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.013322 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 256.08343 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 11), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 185.714286 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 6), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.474124 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 5), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 182.795699 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 13), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 149.425287 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 6), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.474124 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 9), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.937238 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 11), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 182.94702 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 10), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.098592 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 8), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 180.408163 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 8), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.250415 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 9), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 182.644628 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 4), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 184.013322 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 5), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 182.644628 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 10), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 185.714286 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 8), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.098592 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 255.491329 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 12), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 149.223498 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 2, 10), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.098592 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 12), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 148.322148 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 9), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 185.402685 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 3, 7), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 185.092127 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 7), threads=3, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 183.554817 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 4), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.478781 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 10), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 244.469027 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 11), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.478781 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 13), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 193.859649 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 8), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 243.39207 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 6), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 237.379162 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 13), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 192.508711 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 8), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 243.39207 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 7), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 243.39207 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 9), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 238.146552 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 4), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 237.124464 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 12), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 193.0131 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 10), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 230.92999 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 3), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 287.012987 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 11), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 242.857143 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 5), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 242.857143 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 5), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 235.607676 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 7), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 235.106383 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 7), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.478781 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 13), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 190.189329 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 12), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 188.245315 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 9), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 242.59056 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 8), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 235.356763 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 9), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.740741 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 10), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.740741 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 6), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 236.111111 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 12), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 189.536878 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 11), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 237.634409 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 3, 6), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 242.324561 tokens/sec
Configuration: BenchmarkConfig(cpus=(0, 1, 2, 4), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 240.740741 tokens/sec
Configuration: BenchmarkConfig(cpus=(1, 2, 3, 5), threads=4, steps=256, temperature=0.0, seed=42, prompt='Once upon a time', omp_dynamic=False, omp_proc_bind='TRUE', omp_places='threads'), Performance: 235.607676 tokens/sec
```txt
</details>

