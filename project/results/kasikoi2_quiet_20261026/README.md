# kasikoi2 を静かな PC で実行した結果（2026-10-26）

dev/opt の自動探索 `uv run cpuprofiler kasikoi2` を、他のプログラムをすべて止めた状態で2回実行した。
そのあと、探索が選んだ構成を既知の構成と同じブロックの中で測り直して比べた。

## 結論

- kasikoi2 は2回とも9コア（0,2,4–10）を選んだ。報告された速さは 1575 tok/s と 1564 tok/s。
- 確認の測定では、12スレッド 0–11（static、ACTIVE）が中央値 1601 tok/s で最速だった。kasikoi2 の2つの結果はどちらも 1567 tok/s で、12スレッドより2〜3%遅い（比 0.979 と 0.972、95%区間はどちらも1未満）。
- kasikoi2 の結果は、9コア guided,4 ACTIVE（1556 tok/s）とほぼ同じ速さだった。
- 12を見落とした原因は探索の順番にある。kasikoi2 は CPU の組み合わせを OpenMP の既定設定（dynamic,16、ACTIVE なし）のままで決め、上位3つの組み合わせだけ設定を調整する。既定設定では12（1139 tok/s）は9（1195 tok/s）より遅いので、設定調整の対象から外れた。12は ACTIVE と static で一番伸びる構成だが、探索中にその組み合わせで測られたことは一度もなかった。

## 測定環境

| 項目 | 値 |
|---|---|
| PC | 学科PC、hostname EE00025、Intel Core Ultra 5 225U |
| OS | Ubuntu 24.04.3 LTS（ネイティブ）、カーネル 7.0.0-38-generic、Wayland のデスクトップ上のターミナルから実行 |
| コンパイラ | gcc 13.3.0、`-Ofast -fopenmp -march=native` |
| Python | uv 0.13.5、CPython 3.14.9（uv が自動でダウンロード） |
| 電源 | AC 接続、測定中は power profile performance（終了後に balanced へ戻す） |
| 対象 | dev/opt 1d7c2136（2026-10-26 15:42）。モデル stories15M.bin sha256 cd590644d963867a… |
| 生成条件 | `-n 256 -t 0 -s 42 -i "Once upon a time"`、全508回で生成文は同一 |

この PC で `run.c` からビルドし直した `run-fused` は、dev/opt にコミットされているビルド済みのファイルと sha256 が一致した（6d5ea690ec1749bd…）。

## 測定の静かさ

`session/quiet_check.txt` に、測定の前後で5秒ずつ調べた他のプロセスの CPU 使用量を記録した。どの時点でも CPU 1個の1%以下で、ほとんどが gnome-shell だった。

記録用ラッパーで、1回の実行ごとに次の値も記録した。

| 段階 | 実行回数 | 他の処理に CPU を取られた回数（中央値 / 90%点） | 他のプロセスが使った CPU 秒（中央値） | 最高温度 |
|---|---:|---:|---:|---:|
| search1 | 215 | 3 / 10 | 約0 | 91℃ |
| search2 | 212 | 4 / 12 | 約0 | 92℃ |
| 確認の測定 | 75 | 5 / 20 | 約0 | 65℃ |

## 確認の測定（15ブロック、ブロックごとに順番をランダムに入れ替え）

| 構成 | 中央値 tok/s | 最速との比（同じブロック内） |
|---|---:|---:|
| 12スレッド 0–11、static、ACTIVE、g288 | 1601 | — |
| kasikoi2 2回目：9コア、guided,8、wait policy 指定なし、GOMP_SPINCOUNT=10000000、CPU の並べ順を変更、g144 | 1567 | 0.979 [0.957, 0.993]、15ブロック中3回勝ち |
| kasikoi2 1回目：9コア、guided,16、ACTIVE、GOMP_SPINCOUNT=10000、g144 | 1567 | 0.972 [0.957, 0.993]、1回勝ち |
| 9コア、guided,4、ACTIVE、g288 | 1556 | 0.972 [0.944, 1.000]、1回勝ち |
| kasikoi2 の出発点：全14CPU、既定設定 | 965 | 0.604 |

2回の探索は同じ CPU の組み合わせを選んだが、OpenMP の細かい設定は違った。確認の測定では2つに差がなかったので、ほぼ同じ速さの候補のうちどれが選ばれるかは、測定のばらつきで決まっていると考えられる。

## kasikoi2 の探索の中身

`layout_search.txt` を参照。CPU 探索の段階で、CPU 数ごとに一番速かった組み合わせ（1回目）は次のとおり。

| CPU 数 | tok/s | 測定直前の温度 |
|---:|---:|---:|
| 8 | 1145 | 69℃ |
| 9 | **1195** | 70℃ |
| 10 | 1169 | 72℃ |
| 11 | 1163 | 76℃ |
| 12（0–11） | 1139 | 77℃ |
| 14 | 965 | 91℃ |

- 設定の調整に進んだのは、9〜10 CPU の3つの組み合わせだけだった。
- CPU 0–11 で実行されたのは、2回の探索ともに1回ずつで、どちらも既定設定だった。
- 探索は約215回を休みなしで続けるので、温度は探索の最初の5分の1で最高65〜70℃、最後の5分の1で91〜92℃まで上がった。CPU 数の多い組み合わせほど後で測られるので、熱い状態で測られる分だけ不利になる。

## ファイル

`session/` が、学科PC で出力された結果ディレクトリそのもの（tar.gz とダウンロードしたソースは除く）。

- `runs.jsonl`：記録用ラッパーが残した、llama2c の全実行の記録（508行）。各行に、段階、使った CPU、OpenMP の環境変数、引数、tok/s、生成文の sha256、他の処理に CPU を取られた回数（nivcsw）、他のプロセスが使った CPU 秒、実行時間、前後の温度、終了コード。
- `search1.log`、`search2.log`：`uv run cpuprofiler kasikoi2` の出力。Domain の分類結果と、選ばれた構成。
- `confirm.jsonl`、`confirm_candidates.txt`：確認の測定の記録と、比べた構成の一覧。
- `quiet_check.txt`：他のプロセスの CPU 使用量の記録。
- `environment.txt`：OS、カーネル、電源、ソースとバイナリの sha256、CPU 構成。
- `topology.txt`：`uv run cpuprofiler topology` の出力（uv の準備のログを含む）。
- `progress.log`：実行中のログ（uv のインストールのログを含む）。
- `summary.txt`、`analyze_opt.py`：集計結果と集計スクリプト。
- `confirm.py`：確認の測定のスクリプト。
- `opt_quiet_check.sh`：実行したスクリプトのコピー。dev/sakurada d5ff767e のファイルと一致する。

ディレクトリ直下:

- `layout_search.py`、`layout_search.txt`：`runs.jsonl` から、kasikoi2 の CPU 探索の中身を取り出すスクリプトとその出力。
- `SHA256SUMS`：このディレクトリの全ファイルのハッシュ。

## 再現方法

### 集計だけをやり直す

リポジトリのルートで次を実行する。`summary.txt` と完全に一致することを確認済み。

```sh
R=project/results/kasikoi2_quiet_20261026
python3 $R/session/analyze_opt.py $R/session
python3 $R/layout_search.py $R/session
```

### 測定をやり直す

学科PC のネイティブ Ubuntu で、AC 接続のまま実行する。sudo は不要。
他のプログラムをすべて閉じてから始め、終わるまで PC を操作しない。
`Ctrl+Alt+F3` の文字だけのコンソールから実行すると、デスクトップの描画も止まるので、さらに静かになる。
下の URL はスクリプトのコミットを固定している。所要時間は約10分。

```sh
curl -fsSL https://raw.githubusercontent.com/doss-eeic/2026-07-llama2c/d5ff767e/opt_quiet_check.sh -o opt_quiet_check.sh && bash opt_quiet_check.sh
```

スクリプトは dev/opt 1d7c2136 を GitHub から取得してビルドし、`build/run-fused` を記録用ラッパーに置き換える（本物は `build/run-fused.real`）。
そのうえで `uv run cpuprofiler kasikoi2` を手を加えずに2回実行し、最後に確認の測定を行う。
uv がなければ `~/.local/bin` に入れる。結果は `~/llama_opt/results_<hostname>_opt_<日時>` に保存される。

## 注意点

- 記録用ラッパーは、llama2c が負の tok/s を出したとき（時計が飛んだとき）だけ、その実行を最大2回測り直す。今回は一度も起きなかったので、kasikoi2 は元のコードとまったく同じ動きをした。
- 熱による速度制限のカウンタは、このカーネルでは読めていない可能性があるので記録していない。
- 測定は1台の PC で1回だけ。kasikoi2 は乱数の seed を固定しているが、測定値のばらつきで選ぶ構成が変わることは、2回の探索の結果からも分かる。
