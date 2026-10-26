# llama2.cの最適化をCandleに移植しようとしたら、スレッド数を指定するだけで約3倍速くなった

もともとやりたかったのは、Candleのバグ探しではなかった。llama2.c側で進めていたCPU推論の最適化をRust製フレームワークCandleにも移す実験だった。移植前にスレッド数を調整していたら、未修正のCandleが`RAYON_NUM_THREADS`の指定だけで約3倍速くなった。今回は、その理由とLinuxでの追試を記録する。

## 発見までの順序

llama2.cのmasterやdev/optで、QKVの計算をまとめる処理、FFNのup/gateをまとめる処理、二段階argmaxなどを試していた。コアの性能やキャッシュ共有を踏まえ、使う論理CPUとスレッド数も探索していた。そこでCandleをforkし、まず移植しやすい変更から速度と生成文を比べることにした。Candleの`llama2-c` exampleは`stories15M.bin`をそのまま読める。

変更前のCandleをLinuxのCore Ultra 5 225Uで動かすと約112 token/sだった。移植の前提として`RAYON_NUM_THREADS`でスレッド数を、`taskset`や`CANDLE_CPU_MASK`でCPU配置を調整したところ、スレッド数を明示するだけで約3倍速くなった。計算処理の移植前に出た差だったので、既定動作を調べた。

## 原因：行列積のたびに既定スレッド数を取得

CandleのCPU行列積は`crate::utils::get_num_threads()`を呼ぶ。基点コミット`12bb5d1`では、その先の`rayon_num_threads()`が毎回次を実行していた。

```rust
fn rayon_num_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
        .filter(|nt| nt > &0)
        .unwrap_or_else(default_num_threads)
}
```

環境変数が未設定なら`default_num_threads()`に進む。このLinux環境ではCPU選択用の情報が得られず、`num_cpus::get_physical()`へフォールバックする。`num_cpus 1.17.0`のLinux実装は、物理コア数を得るたびに`/proc/cpuinfo`を開いて解析する。小型モデルでは小さな行列積が多く、この取得コストが積み重なる。環境変数を設定するとフォールバックを通らないため、同じスレッド数でも速くなり得る。

LinuxのCPU選択の一部はすでにキャッシュされている。どのフォールバックを通るかは環境に依存するため、全Linux環境で同じ`/proc`アクセスや同じ倍率が出るとは限らない。

## 修正：最初に決めた値を保存

`46cf81e`では`OnceLock`を使い、初回の値を共有する。

```rust
fn rayon_num_threads() -> usize {
    static NUM_THREADS: OnceLock<usize> = OnceLock::new();
    *NUM_THREADS.get_or_init(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| usize::from_str(&s).ok())
            .filter(|nt| nt > &0)
            .unwrap_or_else(default_num_threads)
    })
}
```

初回取得後に`RAYON_NUM_THREADS`を変更しても反映されない、という仕様上の変化はある。

## 元の比較記録

元資料のLinux・20巡の比較では、未修正・既定が112.3 token/s、取得値のキャッシュだけで351.7 token/s、さらに二段階argmaxを有効にすると478.5 token/sだった。平均値の比で順に3.13倍、さらに1.36倍である。生成文のハッシュは全条件で一致したと記録されている。QKV・FFN融合はこの実験では大きな改善にならず、最小構成には残していない。元の20巡の[生データ](../results/candle-original-20261026.jsonl)もリポジトリ内に保存した。比較条件の詳細は元の`/home/denjo/Desktop/candle-llama2c-port/README.md`と[最小移植の記録](candle-minimal-port.md)にある。

## 今回のLinux追試

2026-10-26に、元実験と同じCore Ultra 5 225UのUbuntu 24.04.5 LTS（kernel 7.0.0-38-generic）で再実行した。14論理CPU、メモリ30 GiB、`platform_profile=performance`、CPU governorは`powersave`だった。モデルは`stories15M.bin`、プロンプトは`Once upon a time`、temperature 0、repeat penalty 1。既定の物理コア数は12（`/proc/cpuinfo`の`cpu cores`）。`taskset`によるCPU固定は行わず、既定と同じ12スレッドを明示する条件と、元の比較に合わせた14スレッド明示の条件を設けた。各条件を別プロセスで起動し、条件ごとに1回ウォームアップした後、5条件を巡ごとに順序を入れ替えて20巡測った。

| 条件 | 平均 token/s | 中央値 token/s | 最小〜最大 |
|---|---:|---:|---:|
| 未修正・既定（12スレッド） | 112.7 | 113.0 | 110.7〜113.9 |
| 未修正・12スレッド明示 | 351.4 | 352.2 | 344.6〜356.6 |
| 未修正・14スレッド明示 | 350.5 | 349.9 | 346.5〜357.1 |
| キャッシュ修正・既定（12スレッド） | 351.1 | 352.6 | 328.0〜356.1 |
| キャッシュ修正＋二段階argmax・既定 | 478.2 | 485.2 | 422.8〜496.4 |

平均値の比では、キャッシュ修正は未修正・既定の**3.11倍**、未修正で同じ12スレッドを明示した場合は**3.12倍**だった。キャッシュ修正と未修正・12スレッド明示の平均値は約0.1%の差で、人数をそろえても大きな速度差は残る。二段階argmaxはキャッシュ修正からさらに**1.36倍**だった。全100回で生成文のMD5先頭12桁は`84a86329adc4`で一致した。速度の最小・最大にはばらつきがあるため、小さな差はこの測定だけで評価しない。

原因も実行時に確認した。1回のモデル実行を`strace -f -e trace=openat`で追った結果は次の通り。3条件とも正常終了した。

| 条件 | `/proc/cpuinfo`のopenat回数 | 全openat回数 |
|---|---:|---:|
| 未修正・既定 | 13,806 | 14,455 |
| 未修正・12スレッド明示 | 0 | 626 |
| キャッシュ修正・既定 | 1 | 650 |

この環境で、設定取得が繰り返し物理コア数の解析に進んでいたことを直接示す。`strace`を付けた実行の速度はベンチマークには使っていない。

### 実行したシェルと固定した入力

```sh
rustc +1.99.1 --version
sha256sum build/stories15M.bin \
  /home/denjo/Desktop/candle-llama2c-port/pinned/tokenizer.json \
  /home/denjo/Desktop/candle-llama2c-port/pinned/Cargo.lock
git -C /home/denjo/work/candle rev-parse main perf/cpu-thread-cache perf/llama2c-minimal
git -C /home/denjo/work/candle worktree add --detach \
  /home/denjo/work/candle-cache-fix 46cf81e9486bf448ebbbabbbd9b0f291d5bf54db
cp /home/denjo/Desktop/candle-llama2c-port/pinned/Cargo.lock \
  /home/denjo/work/candle-cache-fix/Cargo.lock
cd /home/denjo/work/candle-cache-fix
CARGO_TARGET_DIR=/home/denjo/work/candle/target \
  cargo +1.99.1 build --release --locked --example llama2-c --features candle-datasets
cp /home/denjo/work/candle/target/release/examples/llama2-c ./llama2-c-cache-fix
cd /home/denjo/work/candle  # perf/llama2c-minimal (92fb2be)
cp /home/denjo/Desktop/candle-llama2c-port/pinned/Cargo.lock ./Cargo.lock
touch candle-transformers/src/models/llama2_c.rs \
  candle-examples/examples/llama2-c/main.rs
cargo +1.99.1 build --release --locked --example llama2-c --features candle-datasets
cd /home/denjo/2026-07-llama2c
python3 scripts/candle_thread_cache_rerun.py \
  /home/denjo/work/candle-upstream-target/release/examples/llama2-c \
  /home/denjo/work/candle-cache-fix/llama2-c-cache-fix \
  /home/denjo/work/candle/target/release/examples/llama2-c \
  build/stories15M.bin \
  /home/denjo/Desktop/candle-llama2c-port/pinned/tokenizer.json \
  20 results/candle-thread-cache-rerun-20261026.jsonl
```

実際の二段階argmax版のビルドでは、共用のCargo targetを別worktreeのビルドで上書きしたため、依存先の`llama2_c.rs`とexampleの`main.rs`の更新時刻を進めて再コンパイルした。次に再実行する場合はworktreeごとに別の`CARGO_TARGET_DIR`を使えば、この操作は不要になる。比較元のバイナリは以前の固定ビルドのもので、今回そのSHA-256を再確認した。

`strace`確認に使ったコマンドは、バイナリと環境変数だけを替え、共通の引数を指定したものだ。以下は未修正・未設定の例である。

```sh
strace -f -qq -e trace=openat -o /tmp/candle-upstream-openat.log \
  env -u RAYON_NUM_THREADS -u CANDLE_CPU_MASK \
  /home/denjo/work/candle-upstream-target/release/examples/llama2-c \
  --cpu --repeat-penalty 1 \
  --tokenizer /home/denjo/Desktop/candle-llama2c-port/pinned/tokenizer.json \
  inference --config /home/denjo/2026-07-llama2c/build/stories15M.bin \
  --temperature 0 --prompt 'Once upon a time' >/dev/null
rg -c '/proc/cpuinfo' /tmp/candle-upstream-openat.log
# 13806
```

比較元の実行ファイルは、固定した基点`12bb5d1f6350f4ea9827d9086f213feec1f0af6a`とCargo.lockから以前ビルドして保存したものを使用した。今回のビルドでは、修正だけのコミット`46cf81e9486bf448ebbbabbbd9b0f291d5bf54db`と、二段階argmaxを含む`92fb2bea58c5e1acae580eb405182bd7a136e822`を使用した。実行時に`RAYON_NUM_THREADS`、`CANDLE_NUM_THREADS`、`CANDLE_CPU_MASK`、`CANDLE_CPU_AFFINITY`は設定されていなかった。測定スクリプトもこれらを一旦除去して条件ごとに設定する。

| 固定した入力 | SHA-256 |
|---|---|
| `stories15M.bin` | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| `tokenizer.json` | `8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1` |
| `Cargo.lock` | `62882b4f9cc876831c73c0812a8ea8e2d9aab0dffc16dfda8cbe014258a95f5e` |

| 実行ファイル | SHA-256 |
|---|---|
| 未修正 `12bb5d1` | `bf649c457cb8f0e59a0e333210823242f8dfe23f246a6bf168f7466e6a025845` |
| キャッシュ修正 `46cf81e` | `e9e4d7152dc9f8aff4e5835c82a4f4b41060898c07c4f30b85a40c8aeff81036` |
| 修正＋二段階argmax `92fb2be` | `9f4a3ddcedfcb1cc046e44dd565bb166b9d0b3be252fca3b5c95b0e1324bf606` |

全100回の各回の値と生成文ハッシュは[生データ](../results/candle-thread-cache-rerun-20261026.jsonl)に保存した。12スレッド明示を追加する前の4条件・80回の[初回データ](../results/candle-thread-cache-rerun-initial-20261026.jsonl)も残した。測定スクリプトは[こちら](../scripts/candle_thread_cache_rerun.py)。

## Macでの既存の追試

別環境のApple M3で行った元原稿の追試では、小さな行列積を既定4スレッドで7回測った中央値が未修正約1.20µs、修正後約0.47µs、未修正で同じ4スレッドを明示すると約0.56µsだった。macOSでは既定値の取得時に`sysctlbyname()`で高性能コア数を問い合わせる。モデル全体の20巡では、未修正・未設定562.8、未修正・4スレッド明示583.5、修正後・未設定581.4、修正後・4スレッド明示578.6 token/s（各平均）だった。全80回で生成文が一致した。Macのモデル全体では約3倍の差は出ておらず、ばらつきも大きい。このMacの測定は今回再実行していないため、元原稿の記録として扱う。

## 移植の準備で見つかったもの

今回の差の大部分は、計算処理の工夫ではなく、未設定時に繰り返したスレッド数取得の削減によるものだ。すでに環境変数を指定している実行が、修正だけで同じだけ速くなるという意味ではない。二段階argmaxの効果は、その既定動作を直した土台の上で別に評価する必要がある。変更前の条件をそろえる作業自体が、移植候補とは別の性能問題を見つけるきっかけになった。

### 根拠と限界

- 発見の順序は提供されたClaude Codeの会話ログに基づく。貼付ログには一部文字欠けがある。
- 元のLinuxの20巡は[保存した生データ](../results/candle-original-20261026.jsonl)と`/home/denjo/Desktop/candle-llama2c-port/README.md`に基づく。今回のLinux追試は別の生データに保存した。
- 最初に環境変数を指定した瞬間の正確なスレッド数とCPU配置は、貼付ログからは確定できない。今回の追試は既定と同じ12スレッドを明示し、設定取得の影響を切り分けた。
- 元のC版とCandle版では停止条件と速度表示の定義が異なるため、両者のtoken/sを直接比較しない。
- [Rust標準ライブラリのOnceLock](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)、[Candleの基点](https://github.com/huggingface/candle/tree/12bb5d1f6350f4ea9827d9086f213feec1f0af6a)、[Mac側の元原稿で参照した修正](https://github.com/WatNeru/candle/commit/1af45e4f)。今回Linuxで測った修正コミットは上記の`46cf81e`。
