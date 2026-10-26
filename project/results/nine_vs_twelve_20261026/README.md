# 9コアと12スレッドの再検証（2026-10-26）

班の測定では「9コア（0,2,4–10）が12スレッド（0–11）より約9%速い」、櫻田の測定では「12スレッドが少し速い」と、結果が逆になっていた。
その原因を調べた4回の測定の生データ、集計スクリプト、集計結果をこのディレクトリにまとめた。

## 結論

- 他に何も動いていなければ、0–11 の12スレッドが9コアより1〜2%速い。
- 裏で CPU 1個の1割程度の処理が動くだけで、12スレッドだけが約8%遅くなり、9コア ÷ 12 の比が 1.101 になる。班の測定の 1.10 とほぼ同じ。
- CPU 0〜11 は同じ最終キャッシュを共有するグループで、12スレッドはこのグループの CPU を全部埋める。ほかの処理が目を覚ますと OpenMP のスレッドと CPU を取り合い、1トークンの間に何度もある同期で全員が待たされる。9コアは 1・3・11 が空いているので影響を受けない。
- schedule（guided,4）、-g、常駐チーム、熱では、班との差は説明できなかった。
- LP-E を含む12（0,2,4–13）が約10%遅いのは班と同じ結果で、遅い LP-E コアのためだった。

## 測定環境

| 項目 | 値 |
|---|---|
| PC | 学科PC、hostname EE00025、Intel Core Ultra 5 225U |
| OS | Ubuntu 24.04.3 LTS（ネイティブ） |
| カーネル | rigorous の2回は 6.17.0-35-generic、gap と noise は 7.0.0-38-generic |
| コンパイラ | gcc 13.3.0、`-Ofast -fopenmp -march=native` |
| 電源 | AC 接続、測定中は power profile performance（終了後に balanced へ戻す） |
| モデル | stories15M.bin sha256 cd590644d963867a…、tokenizer.bin sha256 50a52ef822ee9e83… |
| 生成条件 | `-n 256 -t 0 -s 42 -i "Once upon a time"`、全測定で生成文は同一 |

カーネルは同じ日の中で変わっている。理由は記録していない。

## 使ったソース

各スクリプトが GitHub からコミット固定でソースを取得し、自分でビルドする。

| 名前 | コミット | ファイル |
|---|---|---|
| upstream | master 256b0b24 | run/run_upstream.c |
| team | master 256b0b24 | run/run.c、src/fused.h、src/classifier.h |
| ours | dev/sakurada a49a36f7 | run.c |
| omptmp | dev/omp-tmp 3bd572ed | run.c |
| pt（常駐チーム） | dev/wataru2 f0e4ffbb | run.c、src/omp_pool.h、src/classifier.h、src/fused.h、src/attention.h |

取得したソースの sha256 は各ディレクトリの `environment.txt` に記録されている。

## ディレクトリ

| ディレクトリ | 測定 | 時刻（JST） | スクリプト |
|---|---|---|---|
| `rigorous_session1/` | 13構成 × 30ブロック、冷えた状態から開始 | 13:57–14:00 | rigorous_check.sh（a040a215） |
| `rigorous_session2/` | 同じ内容を、1回目の直後に休まず実行 | 14:02–14:05 | 同上 |
| `gap/` | 19構成、cool 20 / hot 20 / recover 10 ブロック | 15:04–15:22 | gap_check.sh（e05a8428） |
| `noise/` | 4構成 × 5条件 × 25ブロック | 15:40–15:48 | noise_check.sh（bf2ef146） |

各ディレクトリの中身:

- `raw.tsv`：1回の実行ごとの記録。構成、tok/s、生成文の sha256、実行前後の温度と周波数、熱制限カウンタの増分、時刻。
- `environment.txt`：OS、カーネル、電源、CPU 構成、ソースの sha256、候補一覧。
- `progress.log`：実行中のログ。
- `summary.txt`：集計結果。
- `analyze_*.py`：集計スクリプト。`summary.txt` はこれの出力。
- 実行したスクリプト本体のコピー。上のコミットのファイルと一致することを確認済み。
- `gap/blocks.tsv`：ブロックごとの待ち時間、開始時と終了時の温度、電源の状態。
- `gap/trace.tsv`：2秒ごとの温度と周波数。
- `noise/pyrun.py`、`noise/bg.py`：Python 経由の起動と、裏で動かした処理。

ディレクトリ直下のファイル:

- `rigorous_combined_summary.txt`：rigorous の2回をまとめた集計。
- `check_numbers.py` と `check_numbers.txt`：`summary.txt` に出ない数字を計算するスクリプトとその出力。gap の周波数と温度の推移、キャッシュ共有、班の記録から引いた数字。
- `SHA256SUMS`：このディレクトリの全ファイルのハッシュ。

## 再現方法

### 集計だけをやり直す

リポジトリのルートで次を実行する。Python 3.12.3 で、各 `summary.txt` と完全に一致することを確認済み。ブートストラップの乱数は seed 42 で固定している。

```sh
R=project/results/nine_vs_twelve_20261026
python3 $R/gap/analyze_gap.py $R/gap
python3 $R/noise/analyze_noise.py $R/noise
python3 $R/rigorous_session1/analyze_rigorous.py $R/rigorous_session1
python3 $R/rigorous_session1/analyze_rigorous.py $R/rigorous_session1 $R/rigorous_session2
git fetch origin && python3 $R/check_numbers.py
```

### 測定をやり直す

学科PC のネイティブ Ubuntu で、AC 接続のまま実行する。sudo は不要。
実行中は PC を操作せず、ブラウザやエージェントなど他のプログラムも止めておく。
下の URL はスクリプトのコミットを固定しているので、後でブランチ上のスクリプトが変わっても同じものが動く。

```sh
curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/e05a84285d24c3a5ef4d24bb8fde0da8e87e6f87/gap_check.sh -o gap_check.sh && bash gap_check.sh
curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/bf2ef1463ef976b8bced73463108d9742c9c1d43/noise_check.sh -o noise_check.sh && bash noise_check.sh
curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/a040a215b9f0e8a2f76d7f7551389222e5dd15c5/rigorous_check.sh -o rigorous_check.sh && bash rigorous_check.sh
```

所要時間の目安は、gap_check.sh が20〜45分、noise_check.sh が10〜15分、rigorous_check.sh が1回3分ほど。
結果は `~/llama_gap/`、`~/llama_noise/`、`~/llama_rigorous/` に `results_<hostname>_<種類>_<日時>` として保存される。
複数回測った場合は、集計スクリプトに複数のディレクトリを渡すとまとめて集計できる。

## 主な結果

### gap：条件をそろえた比較（cool フェーズ、20ブロックの中央値）

| 構成 | tok/s |
|---|---:|
| 12スレッド 0–11、guided,16 | 1601 |
| 12スレッド 0–11、static | 1579 |
| 9コア 0,2,4–10、guided,4（班の設定） | 1567 |
| 常駐チーム 層9→出力10、guided,4（班の推奨） | 1567 |
| 12スレッド 0–11、guided,4 | 1556 |
| LP-E を含む12、guided,4 | 1421 |
| 元のコード、P コア4スレッド | 637 |

- 同じブロック内で比べると、9 guided,4 ÷ 12 guided,4 = 1.007（班は 1.10）、12 static ÷ 9 guided,4 = 1.014。
- 9コアは static だと guided,4 より 8.5% 遅い。12は guided,4 でも static より 1.4% 遅いだけだった。
- hot フェーズでは、PC が温度を約70℃に保つために周波数を約4.0GHz から3.0GHz に下げ、全構成がそろって約10%遅くなった。順位はほぼ変わらなかった。

### noise：裏の処理を加えた比較（25ブロック）

中央値 tok/s:

| 構成 | file | sha | py | bg10 | bg50 |
|---|---:|---:|---:|---:|---:|
| 9コア guided,4 | 1567 | 1567 | 1556 | 1590 | 1590 |
| 12スレッド static | 1601 | 1579 | 1556 | 1473 | 1188 |
| 12スレッド guided,4 | 1579 | 1556 | 1524 | 1444 | 1169 |
| LP-E を含む12 | 1426 | 1426 | 1426 | 1426 | 1408 |

9コア ÷ 相手（同じブロック内の比の中央値）:

| 相手 | file | sha | py | bg10 | bg50 |
|---|---:|---:|---:|---:|---:|
| 12 guided,4（班は 1.10） | 1.000 | 1.007 | 1.021 | 1.101 | 1.350 |
| 12 static | 0.979 | 0.986 | 1.000 | 1.079 | 1.331 |
| LP-E を含む12（班は 1.12） | 1.099 | 1.092 | 1.092 | 1.116 | 1.129 |

各条件の意味:

- file：出力をファイルに書くだけ。
- sha：出力を sha256sum に渡す。
- py：Python の `subprocess.run(capture_output=True)` で起動する。班の測定スクリプトと同じ方法。
- bg10：sha に加えて、10ms ごとに1ms CPU を使う処理を裏で1つ動かす。
- bg50：sha に加えて、10ms ごとに5ms 使う処理を裏で1つ動かす。

### 班の記録から読めること

`check_numbers.txt` の part 2 を参照。

- 9 vs 0–11（dev/wataru2 1e1fa037 の `bench9v0_11.jsonl`）を測った順に3つに分けると、9コアの中央値は 1579 / 1562 / 1556 と一定だった。0–11 は 1335 / 1449 / 1483 と、だんだん回復している。
- LP-E を含む12（`bench9v12.jsonl`）は 1399 / 1412 / 1408 で一定。
- 同じ日の出力分離の再測定では、全構成をまとめた速度が5巡目から約1450から約800に落ちていた。
- 同じ日に `/home/denjo/2026-07-llama2c` と `/home/denjo/2026-07-llama2c-omp-tmp` の2つのフォルダで測定が行われていた。dev/omp-tmp の測定は 13:26–13:31 JST。
- USB-C 給電ポートの状態は、再測定では 001=1、dev/omp-tmp の測定と櫻田の gap では 002=1 だった。

班の測定のときに他に何が動いていたかは記録がなく、特定できていない。

## 注意点

- gap と noise では、熱制限カウンタの増分がすべて0だった。gap の温め中は最高102℃に達しているので、カーネル 7.0 ではカウンタを読めていなかった可能性がある。0を「熱制限なし」と解釈しないこと。rigorous の2回（カーネル 6.17）では増分が記録されており、1回目は 39683、2回目は 113374。
- RAPL の消費電力は権限がなく読めなかった（`rapl_readable=no`）。
- gap の温度記録用プロセスは CPU 13 で2秒ごとに動いている。CPU 13 を使う X_LP12_* の2構成だけは、ごくわずかに影響を受けた可能性がある。
- noise の bg10 と bg50 は、実際に動いていた処理を人工的にまねたもの。班の測定時の状況そのものではない。
- 測定はすべて1台の PC で、同じ日に行った。別の日や別の PC では、絶対値が数%変わることがある。
