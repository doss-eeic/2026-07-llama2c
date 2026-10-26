# Candle の最小移植と CPU 探索

Candle fork: `/home/denjo/work/candle`。基点は upstream `12bb5d1`（2026-10-26）。

| branch | upstream との差 | 用途 |
|---|---|---|
| `perf/cpu-thread-cache` | `46cf81e`、`candle-core/src/utils.rs` のみ | 既定の CPU スレッド数を `OnceLock` で一度だけ決める |
| `perf/llama2c-argmax` | `966b804`、example/model のみ | 二段階 argmax を `--two-stage-argmax` で有効化 |
| `perf/llama2c-minimal` | 上記二つ（argmax は cherry-pick `92fb2be`） | 推奨構成 |

argmax は出力層の重みをグループ単位で int8 化し、各行の近似スコアと誤差幅を計算する。上界が最大下界に届く候補だけ f32 で再計算する。変更は `classifier.rs` と、隠れ状態・出力層重みの取得口、推論ループの切替に限る。`--group-size` の既定値は 144。非 GGUF・greedy（`--temperature 0`）・`--repeat-penalty 1` 専用。これは CPU 推論向けの探索である。

## 測定済みの基準

`Desktop/candle-llama2c-port/README.md` の 20 巡ベンチマーク（stories15M、Core Ultra 5 225U、`Once upon a time`）より、未修正 upstream 112.3 tok/s、バグ修正だけ 351.7 tok/s、バグ修正と argmax 478.5 tok/s。QKV/FFN 融合は含めない。生成文ハッシュは全構成で `84a86329adc4`。今回の branch は融合部分を除いた移植で、既存の測定値と同じ性能が出るかは環境を揃えて再測定する。

## Candle 用 Python 探索

`src/cpuprofiler/candle.py` は既存の `AutoTuner` の CPU トポロジー候補生成と探索・独立再測定を使う。Candle にない OpenMP 設定の探索を省き、CPU 配置／Rayon スレッド数と int8 グループサイズを探索する。各実行では `taskset`、`CANDLE_CPU_MASK`、`RAYON_NUM_THREADS` を同じ CPU 集合に合わせる。候補間で生成文ハッシュが違えば失敗する。探索結果の JSONL は `--results` で指定する。

```sh
cd /home/denjo/work/candle
cargo +1.99.1 build --release --example llama2-c --features candle-datasets
cd /home/denjo/2026-07-llama2c
PYTHONPATH=src python3 -m cpuprofiler.candle \
  --binary /home/denjo/work/candle/target/release/examples/llama2-c \
  --model build/stories15M.bin \
  --tokenizer /home/denjo/Desktop/candle-llama2c-port/pinned/tokenizer.json \
  --budget 56 --top-k 3 --repeats 3 --results results/candle_search.jsonl
```

`--cpus` は探索の基準構成を指定する。指定しなければ現在のプロセスに許された全 CPU を使う。探索対象は現在のプロセスに許された全 CPU で、各スレッド数と CPU 配置を試す。候補を限定したい場合は `taskset` で Python プロセス自体の affinity を設定する。Candle の推論 example は生成トークン数や seed を外部指定できないため、既存の C ハーネスの `steps` と `seed` は移していない。結果は Candle が表示する token/s で比較する。
