# 常駐ワーカー（-x pool）

`origin/feature/pool-workers`の`456e9c84`にある常駐ワーカーを移植した。
QKV・FFN融合、重みのhugepage配置、二段階argmaxはそのまま使える。

| `-x` | parallel領域を動かすスレッド |
|---|---|
| omp（既定） | 従来どおりOpenMP |
| pool | 起動時のCPU（tasksetで指定したもの）ごとに1本ずつ作って固定した常駐スレッド |

`forward()`には1トークンあたり数十個の小さなparallel領域がある。
libgompは領域ごとにチームの全員へ仕事を渡し、集中型のバリアで全員を待つので、この同期が重い。
`-x pool`では起動時に各CPUへ1本ずつスレッドを作って固定し、以後は作り直しも移動もしない。
メインスレッドは先頭のCPUに固定し、自分の分も計算する（rank 0）。
領域ごとに各ワーカーの受け取り口（別々のキャッシュライン）へ仕事を書き、全員が終わるまでスピンで待つ。

- 置き換えた領域は`matmul`、`matmul_qkv`、`ffn_up_gate`、attention、二段階argmaxの1段目。
  1行（1ヘッド）ごとの計算はOpenMPと同じ関数なので、出力はビット単位で一致する。
- 行は起動時のCPUの数で均等に分ける。`OMP_NUM_THREADS`と`OMP_SCHEDULE`は`forward()`に効かない。
- 二段階argmaxの1段目の`reduction(max)`は、ワーカーごとの最大をメインスレッドでまとめる。
- `OMP_PROC_BIND`が有効だとlibgompが`main()`より前にメインスレッドを1つのCPUに固定するので、
  `OMP_PROC_BIND=false`（または未指定）で使う。そうでなければエラーで終了する。
- ワーカーは出力層の初期化（OpenMP）が終わってから作る。プロセスが終わるまでスピンし続ける。
- Linux専用。他のOSで`-x pool`を指定するとエラーで終了する。

```sh
gcc -Ofast -march=native -fopenmp run.c -lm -o build/run-fused
env OMP_PROC_BIND=false taskset -c 0,2,4,5,6,7,8,9,10,11 \
    ./build/run-fused build/stories15M.bin -x pool -w 1 -a 1 -n 256 -t 0 -s 42 -i 'Once upon a time'
```

Pythonでは共通の`BenchmarkConfig(workers="pool", ...)`で指定する。
`workers="omp"`では`-x`を渡さないため、未改造upstreamも実行できる。
CLIでは`-x pool`を付ける（例：`uv run cpuprofiler kasikoi2 -x pool`）。`beam`・`sample`・`kasikoi`も同じ。
`kasikoi2`の補助設定の調整では、`-x pool`のときOpenMPの項目を飛ばし、`group_size`だけを試す。
