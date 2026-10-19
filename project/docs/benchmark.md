# 詳細benchmark

通常用の `run.c`・`benchmark.py`・既存ビルドは変更しない。
詳細用は `run_detail.c`、`benchmark_detail.py`、設定用の `benchmarkconfig.py`。

## ビルド（Linux）

```sh
make detailbuild CC=gcc
```

`build/omp-run-detail` に出力する。通常用の `build/omp-run` は上書きしない。
モデルの既定値は `build/stories15M.bin`、tokenizerは `tokenizer.bin`。

## autotunerからimportして実行

`PYTHONPATH=src`、またはパッケージをインストールした環境で使う。
`run(config)` は指定された構成で1回だけ実行し、全体とstage別の集計を返す。

```python
from pathlib import Path
from cpuprofiler.benchmarkconfig import BenchmarkConfig
from cpuprofiler.benchmark_detail import run

config = BenchmarkConfig(
    cpus=(0, 1, 2, 3),
    threads=4,
    output=Path("benchmark/detail/task-001"),
)
summary, stages = run(config)
print(summary["forward_ms_per_token"])
```

CPU番号は対象Linux機に合わせる。`output` は実行ごとに異なる新規フォルダを指定する。

候補の作成、CPUの利用可否確認、反復、ウォームアップ、順序、試行間の集計はautotunerが担当する。
詳細benchmarkは実行回数やCPU構成を自分で決めず、CSV出力やコンソールへの進捗表示もしない。
importだけでは実行しない。

設定は `benchmarkconfig.py` の `BenchmarkConfig` にまとめる。
必須は `cpus`、`threads`、`output`。
その他は実行ファイル・モデル・tokenizer、steps、temperature、seed、prompt、
OpenMP設定、timeout、skip_tokens、集計対象のstagesを指定できる。

既存ログだけの解析は、実行と別に呼べる。

```python
from cpuprofiler.benchmark_detail import parse_profile, summarize

profile = parse_profile(Path("benchmark/detail/task-001/profile.jsonl").read_text())
summary, stages = summarize(profile["timings"], skip_tokens=1)
```

## 役割とログ

C側は処理時間、token位置、モデル構成を記録する。
Python側はtasksetとOpenMP環境変数を指定し、実行・解析・集計を行う。
CPU集合やOpenMP設定は呼び出し元のautotunerが管理する。実際のワーカー配置の観測は行わない。

生ログはフラットなJSONL。各レコードは次の4種類。

| event | 内容 |
|---|---|
| `meta` | schema、時計、モデルの各次元 |
| `timing` | forward_id、pos、input_token、layer、stage、整数duration_ns |
| `token` | forward_id、pos、input_token、next_token |
| `end` | forward回数、timing件数、token件数 |

計測stage：

- 層外：`embedding`、`final_norm`、`output`。
- 各層：`attention_norm`、`q_proj`、`k_proj`、`v_proj`、`rope`、`attention`、
  `attn_out`、`ffn_norm`、`ffn_up`、`ffn_gate`、`swiglu`、`ffn_down`。
- 全区間：`layer_total`、`forward_total`。
- forward外：`sampling`。

Attention本体はscore / softmax / weighted sumをまとめる。
`attn_out` と `ffn_down` はresidual加算を含む。
`layer=-1` は層外。`forward_id` は通し番号、`pos` はコンテキスト位置。

`CLOCK_MONOTONIC` の整数ナノ秒で測り、終了後にまとめて出力する。
全区間時間は内部stageの計測費用を含み、内部stageとの重複があるので全行を足さない。
forward時間にsampling、tokenizer、文字出力、ログ保存は含めない。

1回の実行で保存するファイル：

- `stdout.txt`：生成テキスト。
- `stderr.log`：標準エラー。
- `profile.jsonl`：C側の全計測レコード。

返す `summary` はforward回数・集計対象回数・平均ms/token。
`stages` はstageごとの呼出回数・平均ms/call・全層を合計したms/token。

実行失敗、ログ完了と件数、集計対象の有無を確認する。
`steps` は上限なので、早期終了した場合は実際のforward回数で集計する。

解析だけ使う場合は `parse_profile(text)` と
`summarize(profile['timings'], skip_tokens, selected_stages)` をimportできる。

テスト：`python3 -m unittest discover -s tests -v`。

## stay用benchmark

Linuxで`make stayprofile CC=gcc`を実行し、`build/stay-run-detail`を用意する。
`benchmark_stay.run(config)`は外部から渡したポリシーファイルで1回だけ実行する。
CPU集合・演算別の参加人数はポリシーで指定し、stay自身がワーカーを固定する。

```python
from pathlib import Path
from cpuprofiler.benchmarkconfig import StayBenchmarkConfig
from cpuprofiler.benchmark_stay import run

# policy.txt:
# -2 default 0,1,2,3
# -1 output 0,1,2,3,4,5,8,9
summary, stages = run(StayBenchmarkConfig(
    policy=Path("policy.txt"),
    output=Path("benchmark/stay/trial-001"),
))
print(summary["forward_ms_per_token"])
```

戻り値とログ形式は`benchmark_detail`と共通。使用したポリシーも保存先の
`policy.txt`にコピーして、そのコピーを実行に使用する。
`timing_only=True`なら内部stageの時計を止めて`forward_total`のみを計測する。
既定値は`False`で、`layer_total`や`output`も比較できる。
OpenMP環境変数・threads・tasksetの指定は不要。

比較用の`main()`は次のコマンドで実行する。

```sh
PYTHONPATH=src python3 -m cpuprofiler.benchmark_stay
```

P2、P4_HT、P4にCPU 4,5 / 6,7 / 8,9 / 10,11を加えた構成、
P4_4567、P4_4589、P4_891011を固定構成として各1回実行する。
さらに、defaultをP4_HT、outputだけをP4X4（P4_4589）またはP4_89にした
2構成を実行する。CPU番号は既存のLinux測定機に合わせた例なので、必要に応じて
`main()`の構成を変更する。
`results/stay_<日時>/`に各試行のログ・ポリシーと`summary.csv`を保存する。
自動探索や反復・統計評価は行わない。importだけでは実行しない。
