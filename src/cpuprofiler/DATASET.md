# Autotune 用の測定データ

リポジトリのルートで実行する。

```bash
uv run cpuprofiler collect --output benchmark/autotune.sqlite3 \
  --steps 32 64 128 256 --repeats 3 --beam-width 12 --max-depth 4
```

120 token も比較するなら `--steps 32 64 120 128 256` とする。
`-n` に渡す値を steps として保存する。実際の生成が EOS で早期終了する場合、
指定 steps と実生成 token 数は一致しない。生の stderr も各測定に保存する。

全単一 CPU から開始し、各 token 数について各回の順位と３回の中央値の
上位 beam-width 個を親に選ぶ。親に CPU を１つ追加した候補の和集合を
次の深さで測る。max-depth は構成の最大 CPU 数（threads は CPU 数と同じ）。
全組合せの列挙ではないが、小さい CPU 数では全組合せになる場合がある。
同じ token 数・同じ順位規則を使う、収集幅以下の beam の経路を包含する。
別の集約方法や段階的な絞り込みの経路までは保証しない。

各結果は SQLite に即時 commit する。
`samples.elapsed_seconds` に各 runner.run の実時間（秒）も保存する。
プロセス起動・モデル読み込み・推論・結果解析を含み、DB 保存時間は含まない。
既存 DB には列を追加し、過去に未計測だった行は NULL のまま保持する。Ctrl-C 後に同じコマンドで再実行すると
保存済みの測定をスキップする。設定、ホスト、利用 CPU、実行ファイル、モデルの
同一性を確認するため、収集条件を変える場合は別の出力ファイルを使う。
同じデータベースへの収集は１プロセスで行う。

推論を回さず評価する例（`PYTHONPATH=src` またはインストール済み環境）：

```python
from cpuprofiler.dataset import BenchmarkDataset, ReplayRunner
from cpuprofiler.benchmark import BenchmarkConfig

dataset = BenchmarkDataset('benchmark/autotune.sqlite3')
runner = ReplayRunner(dataset, mode='cycle')  # 保存３回を順番に返す
result = runner.run(BenchmarkConfig((0,), 1, steps=128))
print(result.tokens_per_second)
dataset.close()
```

`mode='median'` は保存値の中央値を返す。未収集構成は KeyError とし、推論へ
フォールバックしない。ReplayRunner は AutoTuner の runner としても渡せる。
120→256 などの段階的な絞り込みは `steps` を変えた config で評価できる。
比較時は tok/s の順位精度に加え、測定回数と各測定の所要時間を考慮する。
データが測られた時の熱・負荷・周波数の影響は再生時にも含まれる。

## Mock で探索を実行する

```bash
uv run cpuprofiler mock --data benchmark/benchdata.sqlite3 --steps 120 --mode median
uv run cpuprofiler mock --steps 256 --mode cycle --beam-width 3 --max-depth 4
```

```python
from cpuprofiler.mock import MockBenchmarkRunner
from cpuprofiler.benchmark import BenchmarkConfig

runner = MockBenchmarkRunner('benchmark/benchdata.sqlite3', mode='cycle')
result = runner.run(BenchmarkConfig((0,), 1, steps=120))
print(result.tokens_per_second, result.elapsed_seconds)
print(runner.calls, runner.elapsed_seconds)  # 呼び出し数と仮想的な総測定時間
runner.reset()  # 呼び出し順、乱数状態、集計値を初期化
```

DB は読み取り専用で読み込み、推論も sleep も行わない。
`cycle` は各構成の測定値を順番に返し、末尾から先頭に戻る。
`random` は seed 指定可能な復元抽出。
`median` は速度が中央値に最も近い実測１回の速度・生ログ・時間を返す
（各３回なら中央値そのもの）。中央値を得るための３回分の費用ではなく、
選んだ１回の費用を加算する。３回測定する戦略は run を３回呼んで集約する。
時間が未保存の行は result.elapsed_seconds=None とし、missing_elapsed に数える。
未収集構成は KeyError。CPU ID は DB から取得するため別ホストでも再生できる。
AutoTuner に渡す場合は `base_config=BenchmarkConfig((), 1, steps=120)` で
探索全体の token 数を指定できる。
