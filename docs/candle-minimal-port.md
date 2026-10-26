# Candle の最小移植と CPU 探索

スレッド数取得の原因、今回のLinux追試コマンド、20巡の生データは[追試記事](candle-thread-cache-rerun.md)を参照。

Candle fork: `/home/denjo/work/candle`。基点は upstream `12bb5d1`（2026-10-26）。

| branch | upstream との差 | 用途 |
|---|---|---|
| `perf/cpu-thread-cache` | `46cf81e`、`candle-core/src/utils.rs` のみ | 既定の CPU スレッド数を `OnceLock` で一度だけ決める |
| `perf/llama2c-argmax` | `966b804`、example/model のみ | 二段階 argmax を `--two-stage-argmax` で有効化 |
| `perf/llama2c-minimal` | 上記二つ（argmax は cherry-pick `92fb2be`） | 推奨構成 |

全差分は [candle-minimal-upstream.patch](candle-minimal-upstream.patch) に保存した。`git diff 12bb5d1..perf/llama2c-minimal` と同じ 4 ファイル、+214/−20 行（パッチ形式では309行）。バグ修正単体は +9/−5 行、argmax 単体は +205/−15 行。Python 探索は元の llama2.c リポジトリの `perf/candle-harness` にあり、Candle upstream との差分には入らない。2コミットを一つの PR として出せる構成になっている。

| ファイル | 追加 | 削除 | 内容 |
|---|---:|---:|---|
| `candle-core/src/utils.rs` | 9 | 5 | 既定スレッド数のキャッシュ |
| `candle-examples/examples/llama2-c/classifier.rs` | 144 | 0 | int8 の候補選別と f32 の再計算、単体テスト |
| `candle-examples/examples/llama2-c/main.rs` | 43 | 12 | CLI 引数と推論ループの切替 |
| `candle-transformers/src/models/llama2_c.rs` | 18 | 3 | 最終層直前の隠れ状態と出力重みの取得口 |

argmax は出力層の重みをグループ単位で int8 化し、各行の近似スコアと誤差幅を計算する。上界が最大下界に届く候補だけ f32 で再計算する。変更は `classifier.rs` と、隠れ状態・出力層重みの取得口、推論ループの切替に限る。`--group-size` の既定値は 144。非 GGUF・greedy（`--temperature 0`）・`--repeat-penalty 1` 専用。これは CPU 推論向けの探索である。

## 改善量と検証

`/home/denjo/Desktop/candle-llama2c-port/README.md` に記録した元の20巡ベンチマーク の条件は Core Ultra 5 225U、stories15M、greedy、`Once upon a time`、repeat penalty 1。数値は各構成の平均 tok/s で、倍率は平均値どうしから計算した概数。

| 構成 | tok/s | 直前から | 未修正 upstream から |
|---|---:|---:|---:|
| 未修正 upstream、既定スレッド数 | 112.3 | — | 1.00倍 |
| CPU スレッド数取得の修正だけ | 351.7 | **3.13倍（+213%）** | 3.13倍 |
| 修正 + 二段階 argmax、融合なし | 478.5 | **1.36倍（+36%）** | **4.26倍（+326%）** |

20巡の記録では、修正 + argmax の fix との差は巡ごとの比で **+36.08%**、95% bootstrap 区間は **+34.52〜+37.53%**。生成文ハッシュは全構成で `84a86329adc4`。CPU 既定値の問題は `RAYON_NUM_THREADS` を明示しても回避できるので、3.13倍は主に「未設定の既定動作」を直した効果であり、既に環境変数を設定している実行にはそのまま当てはまらない。

今回の最小ブランチを9コア固定（`0,2,4–10`）、group size 144 で3回実行した結果は **481.81 / 472.64 / 479.77 tok/s**、平均 **478.1 tok/s**、生成文ハッシュはいずれも同じ。Rust の `cargo test --release --locked --example llama2-c --features candle-datasets` は2件通過。Python 探索の短い実行は17測定を完了し、全候補で同じ生成文だった。3回の測定は20巡比較と同じ精度の性能推定ではない。

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
