# bench_memopt — 出力層メモリ最適化の計測

`プロジェクト/output-layer-memory-optimization.md` の方針に沿って、stories15M の forward をステージ別に計測する。
現在は **Step 1（ログを出さない計測モードでベースラインを取る）** まで。

## ファイル

| パス | 内容 |
|---|---|
| `run_baseline.sh` | ビルド → 正解データ確認 → 全構成を計測 → 集計、までを一括で行う |
| `analyze_baseline.py` | 計測結果の集計（単体で再実行可能） |
| `golden.sha256` | 正解出力の sha256（master の `run.c`、P4_HT）。無ければ `run_baseline.sh` が自動生成 |
| `env/` | Step 0 で保存したマシン情報（`lscpu`、コア種別、L2/L3 共有、THP 設定など） |
| `baseline/<timestamp>/` | 計測結果 |

## 前提

- Linux（Intel Core Ultra 5 225U を想定。CPU 0-3 が P コア、4-7 / 8-11 が E コアのクラスタ）
- リポジトリ直下に `stories15M.bin` と `tokenizer.bin`
  ```bash
  git show origin/dev/wataru:stories15M.bin > stories15M.bin
  ```
- `run.c` に `STAGE_TIMING` 版の計測コードが入っていること（下記「run.c の計測モード」）
- `perf` と `turbostat`（`linux-tools-$(uname -r)`）

## 計測前の準備

`run_baseline.sh` は以下を最初にチェックし、満たさなければ止まる（EPP のみ警告）。

| 項目 | 必要な状態 | 設定方法 |
|---|---|---|
| AC 電源 | 接続（`/sys/class/power_supply/AC/online` = 1） | 電源アダプタを接続 |
| `perf_event_paranoid` | 2 以下（1 推奨） | `sudo sysctl kernel.perf_event_paranoid=1` |
| sudo | パスワード無しで通る状態（turbostat 用） | `sudo -v` |
| EPP | `performance`（警告のみ） | `powerprofilesctl set performance` |

`perf_event_paranoid` は `perf stat` で cycles を取り、平均周波数（cycles ÷ task-clock）を出すために下げる。
2 だとユーザー空間のサイクルしか数えられず周波数がやや低く出るので 1 を推奨。再起動で既定値（Ubuntu は 4）に戻る。

```bash
sudo sysctl kernel.perf_event_paranoid=1
powerprofilesctl set performance
sudo -v
```

## 実行

```bash
bash bench_memopt/run_baseline.sh
```

所要時間は 3〜4 分ほど（7 構成 × 5 周、各実行の間に 3 秒待機）。

集計だけやり直す場合:

```bash
python3 bench_memopt/analyze_baseline.py bench_memopt/baseline/<timestamp> --warmup 16
```

## 計測条件

- 推論: `-n 256 -t 0 -s 42 -i "Once upon a time"`（BOS で止まるため実際は 222 トークン）
- ビルド: `gcc -Ofast -fopenmp -march=native -falign-functions=64 -falign-loops=64`。計測ありは `-DSTAGE_TIMING` を追加
  - `make runomp` のフラグに、関数とループの先頭を 64 バイト境界に揃えるオプションを足している
  - 揃えないと、`run.c` を少し変えただけで `matmul` のホットループの配置がずれ、P1 で約 10% 速度が変わる
    （20261013_162539 の計測で、機械語が同じなのに計測あり版が 10% 遅く見えた）。
    最適化案の効果を比べるときも、このフラグでビルドすること
- OpenMP: `OMP_PROC_BIND=TRUE OMP_PLACES=threads OMP_DYNAMIC=FALSE`、CPU の割り当ては `taskset`

| 構成 | スレッド | CPU | バイナリ | 用途 |
|---|---|---|---|---|
| P1 | 1 | 0 | timing | P コア 1 スレッド |
| P2 | 2 | 0,2 | timing | P コア 2 つ（HT なし） |
| P4_HT | 4 | 0-3 | timing | P コア 2 つ + HT（現時点の最速） |
| E4 | 4 | 4-7 | timing | E コア 1 クラスタ |
| X4_2_2 | 4 | 4,5,8,9 | timing | E コア 2 クラスタにまたがる（A3 の比較用） |
| P1_plain | 1 | 0 | plain | 計測コード自体の影響の確認 |
| P4_HT_plain | 4 | 0-3 | plain | 同上 |

- 7 構成を 1 周として 5 周する。周ごとに開始する構成をずらし、熱の影響が特定の構成に偏らないようにしている
- 構成を追加・変更するときは `run_baseline.sh` の `CASES` を編集する（書式 `name|threads|cpus|bin`）。
  `X` と `X_plain` の組があれば、集計で計測コードの影響が自動で計算される

## run.c の計測モード（`-DSTAGE_TIMING`）

- `forward()` の各ステージの経過時間を、配列 `[steps][n_layers][11 ステージ]` に記録するだけで、forward の中では何も出力しない
- 前のステージの終了時刻を次のステージの開始時刻として使うので、ステージの合計と forward の経過時間が一致する
- 生成が終わってから、`dev/wataru` と同じ形式で stderr にまとめて出力する
  ```
  [token:20] 001:embedding-lookup: 0.000000353 seconds
  [token:20, layer:0] 003:qkv-projection: 0.000046... seconds
  [token:20, layer:0] 102:layer-compute-total: ... seconds
  [token:20] 011:final-projection: 0.000798134 seconds
  [token:20] 103:forward-compute-total: 0.001586397 seconds
  ```
- `-DSTAGE_TIMING` なしでビルドすると計測コードはすべて消え、通常の `run` と同じになる
- 計測の有無で stdout は変わらない（sha256 は master と一致）

## 出力（`baseline/<timestamp>/`）

| ファイル | 内容 |
|---|---|
| `environment.txt` | 計測条件、git HEAD、コンパイラ、EPP / AC / 電源プロファイル、警告、`lscpu` |
| `env/` | `bench_memopt/env/` のコピー |
| `git.diff` | 未コミットの変更があった場合のみ |
| `bin/` | 計測に使ったバイナリ（`run`, `run_timing`、golden 作成時は `run_master`） |
| `runs.csv` | 各実行の周・構成・開始/終了時刻・終了コード・sha256 一致 |
| `<case>_r<N>.out` / `.log` / `.perf` | stdout / stderr（ステージ計測と tok/s）/ `perf stat` の出力 |
| `turbostat.txt` | 計測全体を通した turbostat のログ（0.1 秒間隔） |
| `summary.tsv` | 1 行 1 構成の集計結果 |
| `per_token.tsv` | 全トークン・全層・全ステージの生データ（ns）。グラフ用 |

## 集計の方法（`analyze_baseline.py`）

- 終了コードが 0 で sha256 が一致した実行だけを使う
- 1 回の実行の代表値: 先頭 `--warmup`（既定 16）トークンを除いた、ステージごとの**中央値**（層ごとのステージは 6 層の合計）
- 構成の値: 5 回分の代表値の中央値
- 従来（`bench_affinity.sh`）と同じ「全トークンの平均」も `*_mean_all` 列に残す
- forward は `103` の中央値を使う（ステージの中央値の合計ではない）

`summary.tsv` の主な列:

| 列 | 意味 |
|---|---|
| `embedding` 〜 `final`, `layers`, `forward` | ステージごとの時間（ms / トークン） |
| `final_gbps` | 出力層の実効帯域（`vocab × dim × 4 B ÷ final`） |
| `forward_gbps` | forward 全体の実効帯域（1 トークンで読む重みのバイト数 ÷ forward） |
| `tok_s` | `achieved tok/s` の中央値 |
| `overhead_pct` | `X_plain` に対する計測ありの tok/s 低下率 |
| `ghz`, `ipc` | perf の cycles ÷ task-clock、instructions ÷ cycles（P/E の cycles は合計） |
| `bzy_mhz` | turbostat の、使った CPU の Bzy_MHz（Busy% で重み付け平均） |
| `pkg_watt`, `pkg_tmp_max` | パッケージ電力の平均、パッケージ温度の最大 |
| `spread_pct` | 5 回のばらつき（(最大−最小)÷中央値）。計測ありは forward、plain は tok/s |

## Step 1 の完了条件

集計の最後に表示される。1 つでも NG なら終了コード 1。

| 確認 | 目安 |
|---|---|
| 正しさ | 全実行で sha256 が `golden.sha256` と一致 |
| 計測コードの影響 | `overhead_pct` が 1% 未満 |
| 再現性 | `spread_pct` が 5% 未満 |
| トークン数 | 全実行で同じ（BOS で止まる位置が揃っている） |
| 周波数 | perf の `ghz` が記録されている |

NG のときに見るところ:

- ばらつきが大きい → `pkg_tmp_max` と `bzy_mhz` が回によって下がっていないか。AC 電源、EPP、バックグラウンドの負荷を確認する
- 計測コードの影響が大きい → `P4_HT` と `P4_HT_plain` の `ghz` が揃っているか（周波数の違いが原因ではないか）
- sha256 が一致しない → `run.c` のステージの順序や引数が変わっていないか

## 後片付け

```bash
sudo sysctl kernel.perf_event_paranoid=4   # 再起動でも戻る
powerprofilesctl set balanced
```
