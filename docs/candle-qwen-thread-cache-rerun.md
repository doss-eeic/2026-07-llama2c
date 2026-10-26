# Qwen2.5-1.5B-InstructでCandleのスレッド数取得を再測定

[stories15Mでの追試](candle-thread-cache-rerun.md)では、未修正Candleの既定動作が112.7 token/s、既定と同じ12スレッドを環境変数で指定すると351.4 token/sだった。大きなモデルでも同じ設定取得コストが残るかを、通常のSafetensors版Qwen2.5-1.5B-Instructで調べる。

## 条件

- CPU: Intel Core Ultra 5 225U（14論理CPU、物理12コア）、メモリ30 GiB。`platform_profile=performance`、CPU governorは`powersave`。`taskset`や`CANDLE_CPU_MASK`による固定は行わない。
- OS: Ubuntu 24.04.5 LTS、kernel 7.0.0-38-generic。
- Candle: upstream `12bb5d1f6350f4ea9827d9086f213feec1f0af6a`、スレッド数取得だけをキャッシュした`46cf81e9486bf448ebbbabbbd9b0f291d5bf54db`。
- モデル: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)、revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。量子化GGUFではなく`model.safetensors`を使う。
- Candleの`qwen` exampleで`--cpu --model 2-1.5b --model-id Qwen/Qwen2.5-1.5B-Instruct`を指定する。モデルはBF16で保存されているが、このexampleはCPUの場合、`DType::F32`のVarBuilderで重みを読み込む。
- このexampleはローカルの単一`model.safetensors`をカレントディレクトリから探すため、測定スクリプトは各プロセスの作業ディレクトリをモデルディレクトリにする。
- プロンプト`Once upon a time`、temperature 0、repeat penalty 1、seed 42、最大16生成トークン。時間はexampleの生成ループが出力するtoken/sで比較し、モデル取得・読み込みは含まない。
- 各条件を別プロセスで起動し、条件ごとに1回ウォームアップした後、巡ごとに順序を入れ替えて10巡測定する。スレッド数・CPU配置関連の環境変数を実行前に除去し、明示条件だけ`RAYON_NUM_THREADS=12`を付ける。

| 入力・実行ファイル | SHA-256 |
|---|---|
| `config.json` | `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `model.safetensors` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 未修正qwenバイナリ | `65d19c92d0261ef8d61eb2f5df1e3a053dcef652204a1a65c3347be5af0f5d98` |
| 修正後qwenバイナリ | `c89f81b7486c8be792432dd6820a518212e17c868353c15226ba66996f7c2b6b` |

## 結果

| 条件 | 平均 token/s | 中央値 token/s | 最小〜最大 | 平均プロセス時間 |
|---|---:|---:|---:|---:|
| 未修正・既定（12スレッド） | 3.67 | 3.67 | 3.62〜3.74 | 6.72秒 |
| 未修正・12スレッド明示 | 4.13 | 4.12 | 3.99〜4.20 | 6.23秒 |
| キャッシュ修正・既定（12スレッド） | 4.15 | 4.14 | 4.09〜4.21 | 6.19秒 |

平均値の比では、未修正・既定に対して12スレッド明示が**1.125倍（+12.5%）**、キャッシュ修正が**1.131倍（+13.1%）**だった。キャッシュ修正とスレッド数明示の差は約0.6%で、この10巡だけで差があるとは言えない。巡ごとの比を単位にしたbootstrap（10,000回、乱数seed 0）による+13.1%の95%区間は**+11.9〜+14.2%**。全30回で16トークンを生成し、プロンプトと生成文のSHA-256先頭16桁は`26785b76e0357c82`で一致した。1回の試走では未修正・既定の最大RSSは約8.8 GiBだった。

同じ16トークンの生成を`strace -f -e trace=openat`で追うと、次の回数になった。各条件は正常終了した。`strace`付きの速度は表の平均に含めない。

| 条件 | `/proc/cpuinfo`のopenat回数 | 全openat回数 |
|---|---:|---:|
| 未修正・既定 | 4,050 | 4,945 |
| 未修正・12スレッド明示 | 0 | 871 |
| キャッシュ修正・既定 | 1 | 895 |

Qwenでも、未修正・未設定の実行は物理コア数を繰り返し調べていた。一方、計算量が増えたため、この16トークンの比較での速度差は小型モデルの約3倍から約13%に縮んだ。これはこのCPU、モデル、短いプロンプトと生成長での結果であり、長い入力や別のモデルで同じ割合になるとは限らない。[30回の生データ](../results/candle-qwen-thread-cache-rerun-20261026.jsonl)には各回のtoken/s、プロセス時間、生成文ハッシュを保存した。

## 実行手順

```sh
hf download Qwen/Qwen2.5-1.5B-Instruct \
  config.json tokenizer.json model.safetensors \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir /home/denjo/work/models/Qwen2.5-1.5B-Instruct

cp /home/denjo/Desktop/candle-llama2c-port/pinned/Cargo.lock \
  /home/denjo/work/candle-qwen-upstream/Cargo.lock
cd /home/denjo/work/candle-qwen-upstream  # upstream 12bb5d1
CARGO_TARGET_DIR=/home/denjo/work/candle-upstream-target \
  cargo +1.99.1 build --release --locked --example qwen --features candle-datasets

cp /home/denjo/Desktop/candle-llama2c-port/pinned/Cargo.lock \
  /home/denjo/work/candle-cache-fix/Cargo.lock
cd /home/denjo/work/candle-cache-fix  # cache fix 46cf81e
CARGO_TARGET_DIR=/home/denjo/work/candle-qwen-fix-target \
  cargo +1.99.1 build --release --locked --example qwen --features candle-datasets

cd /home/denjo/2026-07-llama2c
python3 scripts/candle_qwen_thread_cache_rerun.py \
  /home/denjo/work/candle-upstream-target/release/examples/qwen \
  /home/denjo/work/candle-qwen-fix-target/release/examples/qwen \
  /home/denjo/work/models/Qwen2.5-1.5B-Instruct \
  10 results/candle-qwen-thread-cache-rerun-20261026.jsonl
```

`strace`確認では、モデルディレクトリに移動して次の引数を使った。明示条件では`RAYON_NUM_THREADS=12`を追加し、修正条件では実行ファイルを修正版に替えた。

```sh
cd /home/denjo/work/models/Qwen2.5-1.5B-Instruct
env -u RAYON_NUM_THREADS -u CANDLE_NUM_THREADS \
  -u CANDLE_CPU_MASK -u CANDLE_CPU_AFFINITY \
  strace -f -qq -e trace=openat -o /tmp/candle-qwen-upstream_default-openat.log \
  /home/denjo/work/candle-upstream-target/release/examples/qwen \
  --cpu --model 2-1.5b --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --weight-path /home/denjo/work/models/Qwen2.5-1.5B-Instruct \
  --prompt 'Once upon a time' --temperature 0 --repeat-penalty 1 \
  --seed 42 --sample-len 16 >/dev/null
```

[測定スクリプト](../scripts/candle_qwen_thread_cache_rerun.py)は、生成トークン数、token/s、全プロセスの経過時間、生成テキストのハッシュを各回保存する。モデルファイルは大きいためリポジトリには含めず、上記revisionとSHA-256で固定した。
