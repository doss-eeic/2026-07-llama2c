# 層・演算別 CPU 構成の探索と stay 実行

`autotune_detail.py` は既存の `benchmark_detail.run()` を呼び、
各 Transformer 層 × 演算の最速 CPU 集合を調べる。
`run_stay.c` は、そのポリシーを読み込む新しい llama2.c 実行コード。
元の `run.c` と `run_detail.c` は変更しない。

## Linux で実行する

```sh
make detailbuild staybuild stayprofile

# CPU ID は対象機の利用可能な CPU に置き換える。
# 各集合を3回測定する。0,1,2,3 の非空部分集合15通りを探索。
PYTHONPATH=src python3 -m cpuprofiler.autotune_detail \
  --cpus 0,1,2,3 --repeats 3 --output benchmark/detail-search

# 推奨された構成で通常の生成（詳細計測なし）
./build/stay-run build/stories15M.bin \
  -c benchmark/detail-search/recommended-policy.txt \
  -n 256 -t 0 -s 42 -i 'Once upon a time'
```

`uv run cpuprofiler detail ...` でも同じ CLI を使用できる。
`--max-workers 4` で CPU 集合のサイズを制限できる。
4095 集合を超える列挙は停止する。P/E や SMT の区別は CPU ID だけから推測せず、
対象機でトポロジーを確認して探索範囲を決める。

既知の有力構成だけを探索する場合：

```sh
PYTHONPATH=src python3 -m cpuprofiler.autotune_detail \
  --candidate 0,1,2,3 --candidate 0,1,2,3,4,5 \
  --repeats 10 --refine-rounds 1 --output benchmark/detail-refined
```

`--model`, `--tokenizer`, `--binary`, `--stay-binary`, `--steps`,
`--skip-tokens`, `--prompt`, `--seed`, `--timeout` も指定できる。
出力ディレクトリは新規でなければならない。途中で失敗した試行の生ログも残る。

## 探索方法

1. 候補の CPU 集合ごとに OpenMP 詳細版を測定する。
   `threads = CPU 数`、`OMP_DYNAMIC=FALSE`、明示した `OMP_PLACES` を使用。
   各反復ブロック内で順序を seed 付きでシャッフルする。
2. `profile.jsonl` を読み、`(layer, stage)` ごとのトークン平均時間を算出する。
   試行間中央値で CPU 集合を選ぶ。ウォームアップ除外は全条件で同じ
   `--skip-tokens`（既定1）を使用する。
3. `q_proj / k_proj / v_proj / attention / attn_out / ffn_up / ffn_gate /
   ffn_down / output` にポリシーを割り当てる。
   `layer_total` や `forward_total` を演算時間に重ねて合算しない。
   Norm、RoPE、SwiGLU、残差加算は調停スレッドで逐次実行する。
4. 混合ポリシーと**全固定候補**を、同じ stay バイナリで改めて測る。
   詳細版の最初の試行から得た入力トークン列を全条件で再生する。
   EOS やサンプリングに測定回数を左右させない。
   各位置の greedy token ID が一致しなければ停止する。
5. 演算タイマーを無効にして `forward_total` だけを計測する。
   ワーカー起床、仕事の分配、完了待ちを含む実測時間の中央値で推奨構成を選ぶ。
   初期推定が遅ければ固定構成が推奨される。

`--refine-rounds N` を付けると、さらに各 `(layer, stage)` の CPU 集合を一つずつ
全候補に置き換え、実際の forward 時間が改善する組み合わせへ座標降下する。
各候補を同じ回数測り、最後に探索前の最速ポリシーとの比較を新しい試行で行う。
試行数は概ね `N × 演算数 × 候補数 × repeats` と大きくなるため、既定は0。
これは局所探索であり、全演算への割り当ての直積を総当たりするものではない。
独立した演算時間の最小値も、大域的な最速構成の保証にはならない。

`--no-validate` は詳細測定と推定だけを行う。
既存試行の再集計だけなら、Linux 以外でも実行できる：

```sh
PYTHONPATH=src python3 -m cpuprofiler.autotune_detail \
  --analyze benchmark/detail-search --skip-tokens 1
```

試行ディレクトリに `config.json`（最低限 `cpus` と `threads`）と
`benchmark_detail` の `profile.jsonl` があれば、既存ログも同じ形式で集計可能。
試行間で入力列、モデルの形状、演算・層の網羅性が異なるログは混ぜない。

## ワーカーを stay させる構造

```text
起動時: ポリシー内の CPU の和集合に1ワーカーずつ作成・固定

調停スレッド: Norm等 → Q → K → V → Attention → ... → Output
                        ↓                           ↓
CPU 0 worker:       担当なら計算                 担当なら計算
CPU 1 worker:       担当なら計算                 担当なら計算
CPU 4 worker:       cond_waitで待機              担当なら計算
CPU 5 worker:       cond_waitで待機              担当なら計算
```

pthread の永続プールを使用し、演算ごとにチームを作り直したり CPU affinity を
付け替えたりしない。参加するワーカーだけを個別の condition variable で起こし、
不参加ワーカーは待機を継続する。調停スレッドは default の最初の CPU に固定し、
ワーカーの完了を condition variable で待つ。各ワーカーについて、
許可 CPU 集合が指定した1個であることと実行 CPU を起動時に検証して stderr に保存する。

行列積は既存と同じ内積順序を保ち、出力行を参加ワーカーに静的均等分配する。
Attention は head 単位で分配する。現在の実装は llama2.c 用であり、GGML の
動的チャンク割り当てや llama.cpp のスレッドプールを利用する実装ではない。
起床・同期方式の異なる llama.cpp へ移植した際は再測定が必要。

## ポリシー形式

```text
# layer stage cpu,cpu,...
-2 default 0,1,2,3
-2 q_proj 0,1,2,3
0 q_proj 0,2
1 attention 0,1
-1 output 0,1,2,3,4,5
```

`-2` は全層に共通、`-1` は最終 Output。層番号は0始まり。
優先順位は層を指定した演算 > 全層共通の演算 > default。
`-2 default ...` は必須。重複ルール、重複 CPU、未知の演算、使用不可 CPU はエラー。
CPU 一覧は範囲記法ではなくカンマ区切りの整数で指定する。
`-c` の代わりに `LLAMA_STAY_POLICY` でもファイルを渡せる。

## 主な成果物

| ファイル | 内容 |
|---|---|
| `search.json` | 探索設定、候補、利用可能 CPU、実行ファイル・モデル等の SHA256 |
| `trial-*/config.json` | 詳細ベンチマークの実行設定 |
| `trial-*/profile.jsonl` | 層・演算ごとの生計測 |
| `selection.json` | 演算ごとの選択、推定時間、固定構成の基準 |
| `policy.txt` | 独立した演算計測から推定した混合ポリシー |
| `replay.txt` | 全体評価に用いる固定入力トークン列 |
| `validation-*/` | 全体評価の生ログ、ワーカー配置 |
| `validation.json` | 固定候補と混合構成の全体実測と中央値 |
| `refinement.json` | オプションの座標降下探索の経過 |
| `recommended-policy.txt` | 全体評価で選んだ推奨構成 |

`forward_tokens_per_second` は測定対象の forward だけの処理率。
Tokenizer、Sampler、文字列出力を含む生成全体の tokens/s とは区別する。
主評価では `stay-run`（詳細計測なし）でも生成動作・性能を確認する。

## 検証

```sh
python3 -m unittest discover -s tests -v
```

合成した2層 GQA モデルを用い、層・演算ごとに異なる参加集合、
仕事数より参加人数が多い場合、不正ポリシー、全位置の logits のビット一致を検証する。
Mac の数値・同期テストに限り明示したテスト用ビルドで CPU 固定を省略する。
通常ビルドは Linux 以外で CPU 固定を要求すると停止する。
Mac 上のテスト結果は Linux 対象機の配置検証・高速化の根拠にはしない。
