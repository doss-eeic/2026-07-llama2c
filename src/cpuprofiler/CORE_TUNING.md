# 異種コア向けOpenMP版

`python3 src/cpuprofiler/benchmark_cores.py`、またはIDEから同ファイルを実行する。
`main()`のBenchmarkConfigでCPU、層内人数、output人数、チャンク数を指定できる。
ビルド、CPU固定、環境変数、シャッフル、結果保存はPythonが行う。

現在の機械は14論理CPU・12物理コア。

| CPU | 構成 |
|---|---|
| 0,1 / 2,3 | Pコア2個のSMTペア |
| 4〜11 | Eコア8個 |
| 12,13 | 低電力コア2個（周波数上限2.4GHz） |

## 今回の有力構成

- P4E4: `(0,2,1,3,4,5,8,9)`、通常8人・output8人、`output_block=1`。
- P2E8: `(0,2,4,5,6,7,8,9,10,11)`、通常10人・output10人、`output_block=4`。
- 通常部分は`dynamic,16`、outputは`dynamic,32`、QKV/FFN統合を有効にする。
  output_block=4では32チャンクが128行に相当する。
- 10回平均でそれぞれ794.90 / 797.87 tok/s。同条件の本家P4E4は714.50 tok/s。
  上位2構成の差は小さく、10人の方が常に速いという結果ではない。

単独の測定もPythonから呼べる。

```python
from cpuprofiler.benchmarkconfig import ROOT, BenchmarkConfig
from cpuprofiler.benchmark_distribute import run

result = run(BenchmarkConfig(
    cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
    threads=10,
    output=ROOT / "results/my_parallel_benchmark",
    binary=ROOT / "build/parallel-run",
    backend="parallel",
    output_block=4,
))
```

`profile=True`なら層・演算別JSONLとCSVも保存する。ファイル名は使い分け、
通常版と詳細版を同じbinaryに混在させない。

## 実装と制約

`backend="parallel"`は現在のrun.cを`LLAMA_PARALLEL`付きでコンパイルする。
通常版、stay版、未改造upstreamの経路を残して比較できる。

- Q/K/Vの行を1つの並列領域で分配し、3回の領域起動を1回に削減。
- FFN up/gateの2つの内積とSwiGLUを行ごとに同時処理。SwiGLUも並列化。
- outputを動的に分配。4行同時計算も選択でき、入力ベクトルを再利用する。
- 層内チーム人数を維持する。Attentionだけ人数を減らす試作は遅かった。
- `fuse=False`、`output_block=1/4`で要素ごとに比較できる。

演算順序とモデル構造は維持しているが、-OfastではOpenMPへのアウトライン化や
ベクトル化による丸め差がある。小型モデルの全logitをO2では完全一致、Ofastでは
絶対誤差1e-7以下で検証する。GQA、SMTを含む14スレッド、不均等な行数と末尾行も対象。
実モデルの全比較で生成文一致を確認した。

詳しい結果とCPU利用の判断は`results/core_sweep_20261020/README.md`を参照。
