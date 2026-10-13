# CPU-Profiler

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
