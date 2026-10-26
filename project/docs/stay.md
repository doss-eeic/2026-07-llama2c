# 本家の計算処理を共有するstay版

Linux用。`run.c`の内積・Attention・Norm等の計算式は共有し、
`-DLLAMA_STAY`で固定ワーカーを使う。通常ビルドは本家と同じ直列/OpenMP経路。
ワーカーの管理は`src/stay.h`にまとめ、推論処理全体のコピーは作らない。

```sh
make staybuild CC=gcc
STAY_CPUS=0,1,2,3,4,5,8,9 STAY_LAYER_THREADS=4 STAY_P_WEIGHT=2 \
  ./build/stay-run build/stories15M.bin -n 256 -t 0 -s 42 -i 'Once upon a time'
```

CPU番号は実機のトポロジー・利用可能CPUに合わせる。上記は以前の測定機用。

- `STAY_CPUS`（必須）：重複のないCPU番号を参加順に指定。
- `STAY_LAYER_THREADS`：先頭から何人を層内で使うか。省略時は全員。
- `STAY_P_WEIGHT`：outputで層内参加ワーカーに与える重み。既定2、1〜16。
  残りのワーカーは重み1。CPU種別を自動判別する設定ではない。

例では層内は先頭4人、outputは8人。outputの32000行を、先頭4人に
各5333/5334行、残りに各2666/2667行の連続範囲として割り当てる。
ワーカーは初回forwardで作成し、終了まで保持する。層内では非参加者を休眠させ、
参加者だけで同期する。コーディネーターは完了待ち中に休眠する。
forwardの同時呼び出しには対応しない。

既存のbenchmark入口もstay用に変更した。

```sh
PYTHONPATH=src python3 -m cpuprofiler.benchmark_distribute
```

`Upstream_P4_HT`、`Upstream_P4E4`、`P4_fixed`、`P4E4_fixed`、`Switch_equal`、`Switch_weight2`を
順序をランダムにして各10回測定する。
CPU番号は`main()`の`cpus`で変更する。結果は`results/stay_日時/`へ保存する。
`Upstream_P4_HT`は本家コミット`350e04fe35433e6d2941dce5a1f53308f87058eb`の
未改造`run.c`をgitから取り出し、`-Ofast -march=native -fopenmp`で別バイナリ
`build/upstream-run`へコンパイルする。本家ソースとrevision・SHA256は、そのバイナリを
ビルドした試行フォルダの`run.c`・`source.json`へ保存する。git履歴が必要。
P4_HTはこの測定機のCPU 0,1,2,3（2物理Pコア・SMT込み）をOpenMPの4スレッドで使う。
`Upstream_P4E4`は同じ未改造バイナリをCPU 0,1,2,3,4,5,8,9の8スレッドで実行する。
CPU番号を変更した場合は実機のSMT対応を確認する。
stayと本家でモデル・入力・生成設定を揃え、ランタイムの待機設定は既定値を使う。
以前の実験の`OMP_*`・`GOMP_*`・`STAY_*`環境変数は引き継がない。
指標は従来と同じ生成全体のtokens/sで、sampling・decode・文字出力も含む。
固定トークンのforward単体測定とは直接比較しない。

検証：`python3 -m unittest discover -s tests -p 'test_stay.py' -v`。
合成モデルの全logitsを通常版とビット単位で比較する（GQA/MHA、不均等な行数、
head数より多いワーカー、配分比1/2、通常最適化/Ofast）。
macOSでのテストはCPU固定を省くため、Linuxの配置・速度の検証にはならない。
