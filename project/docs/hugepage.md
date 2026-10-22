# 重みのhugepage配置

`origin/dev/sakurada`の`e56eeaac`にある重み配置機能を移植した。
QKV・FFN融合と全体共通の`schedule(runtime)`は維持する。

| `-w` | 配置 |
|---|---|
| 0（既定） | 従来どおりチェックポイントを直接mmap |
| 1 | 匿名メモリの2MiB境界に重みをコピー。コピー前に`MADV_HUGEPAGE`を指定 |

`-w 1`は匿名mmapを確保し、2MiB境界を選んでから重みを読み込む。
ファイルのConfigヘッダはコピー対象から除き、重みの内容・順序は変更しない。
要求はTHPへのヒントなので、2MiBページの取得は保証されない。
従来の読み込み処理を再利用し、重みを匿名メモリへコピーしてポインタを付け直す。
元のファイルmmapを解放し、匿名領域の解放は既存の`free_transformer`で行う。

システムのTHP・電源設定は変更しない。コピーするモードはLinux用。
Sakurada版の比較専用モード・層単位の再配置は取り込んでいない。
出力層の量子化と二段階argmaxは[別機能](two-stage-argmax.md)として切り替える。
このブランチの`-w`は0/1の有効・無効だけを指定する。

```sh
gcc -Ofast -march=native -fopenmp run/run.c -lm -o build/run-fused
env OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
    OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
    OMP_SCHEDULE='dynamic,16' \
    taskset -c 0,2,4,5,6,7,8,9,10,11 \
    ./build/run-fused build/stories15M.bin -w 1 -n 256 -t 0 -s 42 -i 'Once upon a time'
```

Pythonでは共通の`BenchmarkConfig(optimize_weights=True, ...)`で指定する。
`optimize_weights=False`では`-w`を渡さないため、未改造upstreamも実行できる。
未改造upstreamには`optimize_weights=True`を指定できない。
`-w 0`対`-w 1`の差にはコピー・位置揃えの効果も含まれる。
