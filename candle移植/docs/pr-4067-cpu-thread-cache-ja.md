# PR #4067: CPU スレッド数の解決結果をキャッシュする
 [huggingface/candle #4067](https://github.com/huggingface/candle/pull/4067) 

## 背景と変更

CPU matmul は演算のたびに `utils::get_num_threads()` を呼びます。`RAYON_NUM_THREADS` が設定されていない場合、Linux の参照機ではデフォルトのスレッド数を解決するために `num_cpus::get_physical()` が呼ばれ、そのたびに `/proc/cpuinfo` が開かれ、解析されていました。

`rayon_num_threads()` の解決結果を `OnceLock<usize>` に保存しました。環境変数の読み取り、正の整数かどうかの確認、CPU 数の検出は、プロセス内の最初の呼び出しで一度だけ行われます。変更対象は `candle-core/src/utils.rs` の 1 ファイルです。

## PR に記載した測定結果

参照 CPU は Intel Core Ultra 5 225U です。まず [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.bin) を CPU 版 `llama2-c` で生成しました。各条件 20 回の平均値です。

| 条件 | 平均 token/s |
| --- | ---: |
| 変更前、デフォルトの 12 スレッド | 112.7 |
| 変更前、`RAYON_NUM_THREADS=12` | 351.4 |
| 変更前、`RAYON_NUM_THREADS=14` | 350.5 |
| キャッシュ後、デフォルトの 12 スレッド | 351.1 |

変更前に 12 スレッドを明示すると、デフォルトに対して 3.12 倍でした。キャッシュ後のデフォルトは 3.11 倍でした。80 回すべてで生成テキストは一致しました。Linux で `/proc/cpuinfo` を調べると、変更前のデフォルト実行では 1 回の推論中に 13,806 回開いており、12 スレッドを明示した場合は 0 回、キャッシュ後は 1 回でした。

[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) の非量子化 Safetensors でも確認しました。CPU 版 `qwen` サンプルは重みを FP32 で読み込みます。同じ CPU で、同一プロンプト、temperature 0、repeat penalty 1、16 トークン生成を各条件 10 回実行した平均値です。

| 条件 | 平均 token/s |
| --- | ---: |
| 変更前、デフォルトの 12 スレッド | 3.67 |
| 変更前、`RAYON_NUM_THREADS=12` | 4.13 |
| キャッシュ後、デフォルトの 12 スレッド | 4.15 |

変更前のデフォルトに対して、キャッシュ後は 13.1%、12 スレッドの明示指定は 12.5% 高速でした。30 回すべてで生成テキストは一致しました。これらの値は 16 トークン生成という条件での結果です。

## PR で示した再現方法と検証

変更前と変更後の `llama2-c` をそれぞれビルドして、対応する実行ファイルを `UPSTREAM` と `FIXED` に指定します。PR の例はモデルと tokenizer を Hugging Face Hub から取得します。

```bash
UPSTREAM=/path/to/upstream/llama2-c
FIXED=/path/to/fixed/llama2-c

run() {
  env -u RAYON_NUM_THREADS -u CANDLE_NUM_THREADS \
      -u CANDLE_CPU_MASK -u CANDLE_CPU_AFFINITY \
      "$@" --cpu --repeat-penalty 1 \
      inference --temperature 0 --prompt 'Once upon a time'
}
run "$UPSTREAM"
run env RAYON_NUM_THREADS=12 "$UPSTREAM"
run env RAYON_NUM_THREADS=14 "$UPSTREAM"
run "$FIXED"
```

既存の CPU matmul テストも実行し、12 件すべて通過しました。

```bash
cargo test --release --locked -p candle-core --test matmul_tests
```

## 挙動の範囲と代替案

現在の変更では、最初の参照時に `RAYON_NUM_THREADS` を含むスレッド数全体が固定されます。その後に環境変数を変更しても結果には反映されません。そのため、`RAYON_NUM_THREADS` は最初の `get_num_threads()` より前に設定します。

PR では代替案として、`default_num_threads()` の検出結果だけをキャッシュする形も提示しました。この場合、`RAYON_NUM_THREADS` は呼び出しごとに読み直すため、途中で変更した値が反映されます。一方、デフォルト CPU 数の検出結果は最初の一度で固定されます。upstream に対して追加 5 行で実装できます。提出したパッチは、この代替案ではなく、スレッド数の解決結果全体をキャッシュする方式です。
