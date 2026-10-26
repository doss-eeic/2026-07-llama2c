# omp-tmp: 旧FP32版の重み付きP/E分配ベンチマーク

この文書は`3bd572ed`の旧FP32版を測った記録。
**融合・hugepage・二段階argmaxを両側で揃えた比較は
[optimized-output-benchmark.md](optimized-output-benchmark.md)を参照。**
現在の`run.c`には最適化を統合済みであり、以下は当時の実装・測定結果である。

測定日: 2026-10-26 / Intel Core Ultra 5 225U。

**FP32の範囲では改善した。最良は層内10人・output10人・P側2倍配分で988.81 tok/s。
探索した同ブランチの非分離最良に対して+3.65%、同一配置・待機設定での重み付けの効果は+7.13%。
一方、masterの既存最良構成は平均1575.20・最高1625.00 tok/sで、1600級の記録を再現した。
このブランチ単独でmasterの速度には届いていない。**

## 対象実装

`origin/dev/omp-tmp` をfetch/pullし、`3bd572ed59901fda549601953b4d8ab7361e25e0` を測定した。
当時は推論ソース `run.c` を変更せず、ベンチマーク、検証テスト、結果、docsを追加した。

|ソース|処理|
|---|---|
|親upstream `350e04fe`|未改造のFP32推論|
|omp-tmp `3bd572ed`|同じFP32推論。最終outputだけ手動の重み付き行分配|
|master `256b0b24` / FP32|QKV・FFN融合。必要に応じてhugepage重み配置|
|master `256b0b24` / int8|融合＋hugepage＋二段階argmax、`-w 1 -a 1 -g 144`|

この測定で使った`3bd572ed`にはQKV/FFN融合、hugepage指定、二段階argmaxは入っていない。
masterとの最適化済み比較にはこれらの差を含むため、人数配分だけの効果とは扱わない。
masterのバイナリは保存済みの実行ファイルを使わず、上記コミットのソースから独立してビルドした。

### 分配方法

`LLAMA_OUT_THREADS=N`、`LLAMA_OUT_P=P`、`LLAMA_OUT_WP=W` のとき、
outputチームの先頭P rankへ各W単位、残りN−P rankへ各1単位の連続した行を割り当てる。
人数・重みはトークン生成中に変えず、outputの並列領域だけ `num_threads(N)` を指定する。
常駐チーム方式ではなく、通常のOpenMP並列領域を使う。

このCPUのPコアは2物理コアで、SMTを含めた論理CPUが0・1・2・3。
「P2」はCPU0・2の2 rank、「P4」はSMTを含む4 rankを表し、4物理Pコアという意味ではない。
rankの順序を `OMP_PLACES` で指定することが重要。
このブランチの通常matmulには `schedule(runtime)` がなく、重み付きoutputも手動分配なので、
`OMP_SCHEDULE` を変更してもこれらの行分配は変わらない。

## 測定条件

- stories15M、同じモデル・tokenizer、greedy、seed 42、prompt `Once upon a time`、`-n 256`。
- GCC 13.3.0、`-Ofast -march=native -fopenmp`、`OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=close`。
- 条件を並列実行せず、各巡で固定seed 42によるシャッフル。各実行は別プロセス。
- 粗探索116構成×2回、近傍・待機設定40構成×3回、独立確認15構成×20回。計652回。
- 各段階で別途2回warmup。ロード・量子化準備は元の `achieved tok/s` の計測区間に含まない。
- `-n 256` は上限で、元の生成ループどおり終了トークンで早期終了する。

表の代表値は `1000 / 平均ms/token`。masterの既存 `check_top()` と同じ
tok/s算術平均も `ranking.json` と `summary.csv` に併記した。

### 探索範囲

P2／SMT込みP4と、E・低速側CPUを組み合わせた6・8・10・12・14人の配置を評価した。
層内人数は候補ごとに2〜14人、P側重みは1〜4、近傍では人数と重みをさらに調整した。
待機設定はACTIVEのspin未指定・0・100・1000・10000と、PASSIVEを試した。
列挙した候補内の最良であり、全CPU部分集合の大域最適を保証する探索ではない。

## 確認結果: FP32

各20回の独立確認。CPU集合A=`(0,2,4,5,6,7,8,9,10,11)`、
集合B=`(0,2,1,3,4,5,8,9)`。spinは `GOMP_SPINCOUNT`、待機は全てACTIVE。

|case ID|ソース・構成|CPU集合|spin|tok/s|
|---|---|---|---:|---:|
|`7c8bebd99380`|omp-tmp 10→10、P2、重み2|A|未指定|**988.81**|
|`052b98625f6d`|omp-tmp 10→10、P2、重み2|A|10000|982.22|
|`52a3ae1a4561`|omp-tmp 10→10、P2、重み2|A|1000|974.21|
|`9c2d9f1b6f0f`|omp-tmp 9→10、P2、重み2|A|未指定|942.63|
|`fd480809fd21`|omp-tmp 10→10、P2、重み1（均等）|A|10000|917.77|
|`ecfc8a9dd03f`|omp-tmp 非分離10人、重み付き経路OFF|A|10000|916.82|
|`876a1d0c733a`|omp-tmp 非分離8人、重み付き経路OFF|B|10000|954.03|
|`d1df64e59ba2`|omp-tmp 非分離8人、重み付き経路OFF|B|未指定|950.33|
|`f473ab42901e`|親upstream 非分離8人|B|10000|956.71|
|`d513ca3f8aab`|master FP32、融合あり、hugepageなし、8人|B|10000|978.96|
|`f661fc474b5c`|master FP32、融合＋hugepage、8人|B|10000|1000.91|

同ブランチの非分離最良954.03に対する988.81の改善は **+3.65%**。
ただしCPU配置・人数・spinも異なるため、これを全て重み付け単体の効果とはしない。
同一配置・人数・spin10000で、重み付きOFFの916.82に対する重み2の982.22は **+7.13%**。
同じチームによる均等配分917.77から重み2への変更は **+7.02%**。

一方、9→10は10→10より **4.67%遅い**。
今回の最良構成は「人数を切り替えず、outputだけP/Eの仕事量を変える」構成だった。

### 探索中の傾向

集合Aの10→10、ACTIVE、spin未指定での粗探索結果（各2回）:

|P側の重み|tok/s|
|---|---:|
|1|903.89|
|2|997.74|
|3|938.43|
|4|907.60|

集合Aの10→10、重み2での待機設定探索（各3回）:

|待機|spin|tok/s|
|---|---:|---:|
|ACTIVE|未指定|980.77|
|ACTIVE|10000|1000.00|
|ACTIVE|1000|982.22|
|ACTIVE|100|782.76|
|ACTIVE|0|739.13|
|PASSIVE|未指定|730.18|

探索ではspin10000が上位だったが、独立確認では未指定がわずかに上回った。
差の95%区間は−0.47〜+1.94%で、spin10000に対する未指定の優越は確認できない。
最速の一回だけを採用せず、再確認を優先して推奨設定を選んだ。

## masterの1600級との直接比較

masterの `check_top()` に存在する2構成をそのまま測り、さらに配置を変えた2構成を比較した。
以下は全て `-w 1 -a 1 -g 144`、9人。P側重み付き分配は使用しない。

|CPU順|schedule / 待機|代表tok/s|算術平均tok/s|最高1回tok/s|
|---|---|---:|---:|---:|
|`(0,1,2,3,6,7,9,10,11)`|static / 未指定（既存構成）|1166.84|1168.99|1227.78|
|`(1,2,3,5,6,7,8,10,11)`|guided,16 / ACTIVE（既存構成）|**1575.20**|1575.58|1625.00|
|`(0,1,2,3,6,7,9,10,11)`|guided,16 / ACTIVE（追加対照）|1546.00|1547.39|1589.93|
|`(0,2,3,5,6,7,8,10,11)`|guided,16 / ACTIVE（追加対照）|1560.18|1561.13|1637.04|

既存の1600級の記録は今回も再現した。omp-tmp最良988.81はmaster最良平均1575.20に対して **37.23%遅い**。
FP32限定の改善と、二段階argmax等を含めた全体の最高性能は区別する必要がある。
master FP32については表の2設定を対照測定しただけで、全設定の最適化探索はしていない。

## 推奨コマンド

旧FP32版を再現する場合の設定。CPU番号はこの225Uのもの。

```sh
git show 3bd572ed:run.c > /tmp/run-omp-tmp-fp32.c
gcc -Ofast -march=native -fopenmp /tmp/run-omp-tmp-fp32.c -lm -o /tmp/run-omp-tmp
env -u GOMP_SPINCOUNT -u OMP_THREAD_LIMIT \
  OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=ACTIVE \
  OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
  LLAMA_OUT_THREADS=10 LLAMA_OUT_P=2 LLAMA_OUT_WP=2 \
  taskset -c 0,2,4,5,6,7,8,9,10,11 \
  /tmp/run-omp-tmp /path/to/stories15M.bin -z tokenizer.bin \
  -n 256 -t 0 -s 42 -i 'Once upon a time'
```

全CPU rankに層内計算を割り当て、outputではCPU0・2が各E rankの約2倍の行を担当する。
当時は二段階argmaxなどと組み合わせた性能を評価していなかった。

## 再現手順

リポジトリルートから実行。Python標準ライブラリ、GCC、tasksetを使う。
比較ソースの取得にgitを使うので、対象コミットがローカルにない場合は先に`git fetch origin`する。
モデルは共有fixtureを使用し、ブランチへモデルや実行バイナリを追加していない。

```sh
python3 -m unittest discover -s tests -p test_weighted_output.py -v

python3 benchmarks/weighted_output.py --model /path/to/stories15M.bin \
  --phase coarse --repeats 2 --output results/replay_coarse \
  --main-ref 256b0b24 --upstream-ref 350e04fe

python3 benchmarks/weighted_output.py --model /path/to/stories15M.bin \
  --phase tune --repeats 3 --previous results/replay_coarse \
  --output results/replay_tune --main-ref 256b0b24 --upstream-ref 350e04fe

python3 benchmarks/weighted_output.py --model /path/to/stories15M.bin \
  --phase final --repeats 20 --previous results/replay_tune \
  --output results/replay_final --main-ref 256b0b24 --upstream-ref 350e04fe
```

各実行前後のAC状態、温度、周波数、loadavg、stdoutのSHA256を記録する。
親プロセスのOpenMP・LLAMA_OUT関連環境変数は除去してから設定する。
ビルド生成物は一時ディレクトリに置き、終了時に削除する。

## 正しさ・安定性・記録

652回すべてで生成文のSHA256が同じFP32 upstream参照と一致した。
追加テストではO2とOfast、3〜14人、P rank数1〜4、重み1〜6、奇数の行数を含むサイズで
行の欠落と数値一致を確認し、14 rankのCPU配置も検証した。
実際のチーム人数を固定する `OMP_DYNAMIC=FALSE` の条件で評価している。

全実行前後のAC状態は接続のまま。最終確認の各巡を全15構成で集計した速度は
1035.61〜1064.89 tok/sで、前回の再測定のような途中の大幅な速度低下はなかった。
異なる構成をまとめた巡速度は動作状態の変動を見る目安であり、特定構成の性能値ではない。

改善率の区間は、巡番号の対応を保った10000回のpaired bootstrap（seed 42）で計算した。

|比較|改善率|95%区間|
|---|---:|---:|
|最良 vs 同ブランチ非分離最良|+3.65%|+2.66〜+4.66%|
|同一配置・spinで重み付きON vs OFF|+7.13%|+5.41〜+8.95%|
|同一配置・spinで重み2 vs 重み1|+7.02%|+5.65〜+8.24%|
|9→10 vs 10→10|−4.67%|−5.77〜−3.60%|

記録:

- [粗探索](../results/omp_tmp_coarse_20261026/ranking.json): 116構成×2回。
- [近傍・待機設定探索](../results/omp_tmp_tune_20261026/ranking.json): 40構成×3回。
- [最終確認CSV](../results/omp_tmp_final_20261026/summary.csv): 15構成×20回。
- [最終確認の全実行](../results/omp_tmp_final_20261026/runs.jsonl): 設定、速度、環境、stdout SHA256。
- [比較・区間・巡ごとの速度](../results/omp_tmp_final_20261026/comparison.json)。
- [メタデータ](../results/omp_tmp_final_20261026/metadata.json): ソース、モデル、tokenizer、バイナリのハッシュ等。
- [検証ログ](../results/omp_tmp_final_20261026/tests.log)。

各段階の`cases.json`と`runs.jsonl`にCPU順を含む全設定を保存した。
