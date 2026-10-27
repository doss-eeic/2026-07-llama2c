# kasikoi2 Autotuner

`dev/opt` のkasikoi2をmasterに取り込み、Autotunerはこの探索だけに統一した。
旧 `beam`・`sample`・`kasikoi` コマンドと旧探索APIは削除した。
Python APIはスペルを修正した `AutoTuner.kasikoi2_search(...)` を使う。

## 実行

Linuxのsysfs、`taskset`、OpenMP対応の実行ファイルとPython 3.14以上が必要。
macOS/WindowsではCLIのヘルプ表示とモックによるテストを実行できる。

```bash
make runomp
uv run cpuprofiler topology
uv run cpuprofiler
make run
make run ARGS='-n 128 -i "Hello"'
uv run cpuprofiler --help
```

モデルは事前に用意する。既定の入力は `build/stories15M.bin`、`tokenizer.bin`、
256 steps、temperature=0、seed=42、prompt="Once upon a time"。
CLIではhugepage重み配置とtwo-stage argmaxを有効にする。
`--group-size`（既定144）はモデルのdimを割り切る値を指定する。
`--binary`・`--model`・`--tokenizer`・`--steps`・`--timeout`で実行条件を変更できる。
`--top-k`・`--repeats`・`--calibration-repeats`・`--tolerance`・`--seed`で探索条件を変更できる。
引数なしの`uv run cpuprofiler`でkasikoi2を実行する。
`uv run cpuprofiler --model ...`のようにオプションだけを指定できる。
明示的な`kasikoi2`サブコマンドも利用できる。`topology`は情報表示専用。
ビルドは探索前に`make runomp`で明示的に行う。実験ログや旧benchmark_distributeは取り込んでいない。

## 最良設定の保存と実行

CLIは探索成功後、最良設定を展開した`build/run-tuned.sh`を出力する。
CPU affinity・OpenMP環境変数・モデル/生成引数を含むBashスクリプトで、そのまま読んで編集できる。
実行時のPython/uvは不要。Linuxの`taskset`と探索に使った推論バイナリ、モデル、tokenizerを使う。
パスは絶対パスで保存するため、別ディレクトリからも実行できる。
`--output /path/to/script.sh`で保存先を変更できる。

`make run`は`build/run-tuned.sh`を実行する。
出力がない場合は探索を先に実行するよう案内して終了する。
別のスクリプトを使う場合は`make run TUNED_SCRIPT=/path/to/script.sh`を指定する。
`make run ARGS='-n 128 -i "Hello"'`またはスクリプトへの引数で生成設定を上書きできる。
モデルは探索時のものを使い、計測用timeoutは通常推論には適用しない。

元の`make run`のビルド処理は`make build-run`に移した。
引数なしの`make`は引き続き通常版をビルドする。
スクリプトは保存したパスのバイナリを実行するので、再ビルドするとその新しいバイナリを使う。
設定を更新するには`uv run cpuprofiler`を再実行する。
生成スクリプトはマシン固有なのでGit管理しない。

Python APIで保存する場合は次を使う。

```python
from pathlib import Path
from cpuprofiler import export_tuned

export_tuned(best, Path("build/run-tuned.sh"))
```

## 探索の流れ

1. 入力構成でウォームアップする。OpenMP設定はCPU探索用の既定値に戻す。
2. 各CPUの単独性能と、同じ構造を持つDomainの全体性能を測る。
   DomainはL2共有または同一物理コアで接続されたCPU群。
   同じ構造で性能差とばらつきが許容値以内のDomainだけを同型にまとめる。
3. 1 CPUから始め、その段階で最速だった配置に1 CPUずつ追加する。
   同型クラス内の等価な配置は代表1つだけを測る。各スレッド数の結果を残す。
4. 最速配置でschedule、wait policy、spincount、proc_bind、dynamic、CPU配置順、
   argmaxのgroup sizeを1項目ずつ試し、項目ごとに上位3値に絞る。
   CPU探索の上位top_k配置で、それらの組み合わせを座標降下で最大2周調整する。
   group sizeはtwo-stage argmax使用時だけ、dimの約数（16以上）と入力値を試す。
5. 調整後の構成と入力の基準構成を、順番を入れ替えながらrepeats回独立に測り直す。
   ms/tokenの平均で比較し、最良のBenchmarkResultを返す。

greedy探索なので全CPU部分集合や全設定の総当たりによる最適性は保証しない。
測定中の他プロセスや温度による影響は残るため、負荷の少ない環境で実行する。
キャッシュは探索ごとに初期化し、同一探索中の同じBenchmarkConfigだけを再利用する。
ウォームアップと最終確認はキャッシュを使わない。

## Python APIと設定

```python
from cpuprofiler import AutoTuner, BenchmarkConfig, BenchmarkRunner, CPUTopology

topology = CPUTopology()
config = BenchmarkConfig(
    cpus=tuple(sorted(topology.cpus)), threads=topology.n_logical_cpus,
    optimize_weights=True, two_stage_argmax=True,
)
tuner = AutoTuner(topology, BenchmarkRunner())
best = tuner.kasikoi2_search(config, top_k=3, repeats=3, calibration_repeats=3)
print(best.config)
print(best.tokens_per_second)
```

`BenchmarkConfig`に実行条件を集約する。`BenchmarkRunner().run(config)`はその
binary/model/tokenizer/timeoutを含む設定をそのまま実行する。
Runner側の重複するパス設定や、実験スクリプト専用のoutput/build設定は持たない。
既定のパスはリポジトリ基準。独自のパスは `pathlib.Path` で渡す。
親プロセスのOMP_/GOMP_/OUTPUT_環境変数を除去してから、構成のOpenMP設定を渡す。
単発benchmarkではhugepage/two-stage argmaxは既定で無効なので、upstreamとの比較にも使える。

探索中は`threads=len(cpus)`とし、入力のcpusは基準構成に使う。
探索候補はプロセスが利用可能な全CPUから作るため、探索対象を制限する場合は
`taskset`でPythonプロセス自体のaffinityを制限する。
入力の`omp_places`は探索対象のCPUから作り直す。基準構成もCPU番号順のplacesを使う。

`tuner.last_search`には所要時間、実測回数、基準構成、最終確認結果、
キャリブレーション結果、スレッド数別の探索結果、調整後の構成、絞った設定値を残す。
`calibrated=`で以前のキャリブレーションを渡す場合は、同じCPU構造と実行条件で採取したものを使う。

## テスト

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

実モデルを使わず、CPU配置・同型判定・測定再利用・独立確認・設定反映・CLIを検証する。
