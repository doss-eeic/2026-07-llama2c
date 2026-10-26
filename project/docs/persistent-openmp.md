# OpenMP常駐チームで通常とoutputの人数を分ける

実測に基づく推奨設定と比較結果は [docs/output-tuning.md](../../docs/output-tuning.md) を参照。
以下の8→14は方式を説明する例で、最速設定ではない。

`dev/opt`の通常OpenMP版に、生成・chatループ全体を1つの`parallel`領域で囲む方式を追加した。
pthreadは使用しない。以前の`dev/wataru-2`系のようにoutputごとに別チームを開く方式ではない。

```python
config = BenchmarkConfig(
    # 先頭8人が通常演算、全14人がoutput。CPUの順番が重要。
    cpus=(0, 2, 1, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13),
    threads=8,
    persistent_openmp=True,
    output_threads=14,
    omp_dynamic=False,
    omp_proc_bind="close",
    omp_schedule="guided,16",
    omp_wait_policy="ACTIVE",
    optimize_weights=True,
    two_stage_argmax=True,
    group_size=144,
)
```

- 常駐人数は`max(threads, output_threads)`。指定どおりのチームを作れなければ終了する。
- `output_threads=None`なら通常と同人数。既定の`persistent_openmp=False`は従来方式。
- 通常matmul・QKV・FFN・attentionは先頭`threads`人だけで処理する。
- FP32出力matmul、または二段階argmaxの全語彙int8スコア計算は先頭`output_threads`人で処理する。
- RMSNorm・RoPE・残差加算・候補のFP32再計算・samplingは従来どおりcoordinatorが処理する。

`src/omp_pool.h`が仕事を渡し、全員が共通の開始・終了バリアを通る。
通常8人のとき、残り6人は行計算をせずバリアで待つ。
全員が同じバリア構文を同じ順序で通り、参加者だけの条件分岐内で`omp for`は使わない。
行分配は共通の`OMP_SCHEDULE`を読み、static（連続／チャンク巡回）、dynamic、guidedに対応する。
後二者は参加者だけが共有キューから行を取る。分配の細部はlibgompの通常ループと同一とは限らない。

`OMP_PLACES`はconfigのCPU順から作られ、`close`でrankとCPUを対応させる。
`OMP_WAIT_POLICY`と`GOMP_SPINCOUNT`は待機中のスレッドにも効く。
ACTIVEでは非参加者もCPUを使うため、人数切替が必ず速くなるわけではない。
chatでは入力待ちの間もチームを維持する。
モデル読み込み・重み量子化は常駐領域へ入る前に行う。

## shellからの実行

```sh
gcc -Ofast -march=native -fopenmp run.c -lm -o build/run-fused
env PERSISTENT_OPENMP=1 OMP_NUM_THREADS=8 OUTPUT_THREADS=14 \
    OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_WAIT_POLICY=ACTIVE \
    OMP_PLACES='{0},{2},{1},{3},{4},{5},{8},{9},{6},{7},{10},{11},{12},{13}' \
    OMP_SCHEDULE='guided,16' \
    taskset -c 0-13 ./build/run-fused build/stories15M.bin \
    -n 256 -t 0 -s 42 -i 'Once upon a time' -w 1 -a 1 -g 144
```

stderrの`persistent OpenMP: team=14 layer=8 output=14`は実際のチーム作成後に表示する。
OpenMPなしのビルドでこのモードを要求した場合や、`OMP_DYNAMIC=TRUE`の場合はエラーにする。

## 比較と検証

```sh
PYTHONPATH=src python3 -c 'from cpuprofiler.benchmark import main5; main5()'
PYTHONPATH=src python3 -c 'from cpuprofiler.benchmark import main5; main5(False)'
```

`main5`は通常8／14人、常駐8→8・8→14・14→14を各10回、順番を回して比較する。
`False`では二段階argmaxを無効にしてFP32出力を比較する。値は平均ms/tokenの逆数。
AC接続の変化も確認し、途中で変わった場合は速度比較を参考値として表示する。
既存の`check_top()`や`__main__`の選択は変更しない。

`tests/test_persistent.py`は同じ常駐チーム内のOSスレッドID、行の重複／欠落、
非参加rankの除外、空ジョブでの終了、複数回の起動・終了を検証する。
GQA/MHA、FP32/int8、端数のある行数、static/dynamic/guided、1→1・2→4・4→2・8→14・14→8で
既存方式のlogits・gate・KV cacheと比較する。
