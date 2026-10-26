# output の人数分離のチューニング結果

測定・整理日: 2026-10-26。コマンドと`src/`・`results/`のパスはリポジトリのルートを基準とする。

Core Ultra 5 225U / stories15M / greedy / `Once upon a time` / `-n 256`。
**非分離にほぼ匹敵する構成は得られた。安定して上回ることまでは確認できなかった。**

## 結果

各構成20回、測定順を固定seedでシャッフル。数値は `1000 / 平均ms/token` の tok/s。
第1確認と追加確認は別の測定で、絶対速度には時間帯による変動がある。

|構成|第1確認|追加確認|
|---|---:|---:|
|修正前・元の 8→14 / guided,16 / g144|1056.91|—|
|修正前・探索最良 9→10 / static / g144|1443.97|—|
|修正後 9→10 / guided,4 / g288|**1539.53**|1429.96|
|修正後 9→10 / guided,4 / g144|1525.72|1442.09|
|非分離・通常OpenMP 9人 / guided,4 / g288|1531.53|1454.43|
|非分離・通常OpenMP 9人 / guided,4 / g144|1519.42|1466.00|
|非分離・通常OpenMP 10人 / guided,4 / g288|—|1441.15|
|非分離・常駐10人 / guided,4 / g288|1537.93|1455.14※|

※追加確認では同一の常駐10人設定が2候補に含まれたため、このセルは計40回をまとめた値。
探索スクリプトには重複候補の除去を追加済み。

第1確認では g288 の分離構成が非分離の最速構成に対して +0.10%。
反復番号を単位にしたpaired bootstrapの95%区間は −0.84〜+1.25%で、優越は判定できない。
追加確認では −2.46%（同区間 −4.30〜−0.83%）。
g144 の分離構成は、それぞれの測定の非分離最速に対して −0.79% / −1.63%。
従って「実用上かなり近い速度」は達成したが、人数分離自体が常に有利とは言えない。

元の8→14から最良構成への改善は、第1確認の同時比較で **+45.7%**。
修正前の探索最良からは **+6.6%**。これらは配置・スケジュール・gとコード修正を合わせた効果である。

## 推奨設定

```python
config = BenchmarkConfig(
    cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11),
    threads=9,
    persistent_openmp=True,
    output_threads=10,
    omp_dynamic=False,
    omp_proc_bind="close",
    omp_schedule="guided,4",
    omp_wait_policy="ACTIVE",
    gomp_spincount=None,
    optimize_weights=True,
    two_stage_argmax=True,
    group_size=288,  # 144も僅差。追加確認では144の方が速かった
)
```

層内はCPU `(0,2,4,5,6,7,8,9,10)`、outputではCPU11も参加する。
PコアのSMT側CPU1・3と、低速側CPU12・13を使わない。
同じチームに属するだけの待機スレッドもバリアに参加するため、outputを14人に広げると層内にも費用がかかる。
`GOMP_SPINCOUNT` の小さい値や、outputを層内より少人数にする設定も探索したが、この構成を上回らなかった。

## 変更

- `src/omp_pool.h`: guidedのチャンク確保をcritical領域からatomic CASに変更。行の配分方式は維持。
- `src/classifier.h`: rankごとの最大値を64バイト間隔に配置し、チャンクごとの更新でfalse sharingが起きる配置を避けた。
- `src/cpuprofiler/tune_output.py`: ビルド、逐次実行、順序シャッフル、粗探索、近傍探索、schedule/spin/g探索、独立確認を追加。

ランナーは各実行の設定・stdout・stderr・電源状態を保存し、生成文のハッシュがwarmupと異なれば停止する。
コンパイラ、CPU情報、ソースのSHA256も保存する。既存の`check_top()`や既定設定は変更していない。

## 再現

推奨構成だけを実行する場合:

```sh
gcc -Ofast -march=native -fopenmp run.c -lm -o build/run-fused
env -u GOMP_SPINCOUNT PERSISTENT_OPENMP=1 OMP_NUM_THREADS=9 OUTPUT_THREADS=10 \
  OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_WAIT_POLICY=ACTIVE \
  OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
  OMP_SCHEDULE='guided,4' \
  taskset -c 0,2,4,5,6,7,8,9,10,11 \
  ./build/run-fused build/stories15M.bin \
  -n 256 -t 0 -s 42 -i 'Once upon a time' -w 1 -a 1 -g 288
```

探索・確認ランナーは親プロセスのOpenMP関連環境変数を除いてから設定を適用する。

```sh
# 最初から探索
PYTHONPATH=src python3 -m cpuprofiler.tune_output

# 保存済み候補で、現在のソースをビルドして20回ずつ確認
PYTHONPATH=src python3 -m cpuprofiler.tune_output \
  --refine results/output_tune_groups --final --repeats 20
```

探索はCPU番号がこの225Uに対応する前提。FP32出力のみの速度や、別モデルでの最適性は今回の測定対象外。
`-n 256`は上限であり、元の生成ループどおり終了トークンで早期終了する。
速度は元の`achieved tok/s`を使い、モデルロード・量子化準備・最初のtokenを除く生成区間を比較した。

## 記録と検証

計1270回の本測定（ほか各段階のwarmup）。段階ごとの記録:

- `results/output_tune_coarse`: 人数・CPU順・待機設定。修正前のヘッダも`original_src/`に保存。
- `results/output_tune_schedule`: 修正前のスケジュール探索。
- `results/output_tune_neighbors`: 修正前の近傍探索。outputを減らす方向も含む。
- `results/output_tune_cas`: 修正後のスケジュール探索。
- `results/output_tune_groups`: g探索。
- `results/output_tune_final`: 各20回の第1確認。修正前バイナリとも比較。
- `results/output_tune_confirm`: 各20回の追加確認。非分離の通常OpenMP10人も含む。

全本測定で生成文のSHA256は一致し、記録したAC状態は接続のままだった。
確認測定後に、3プロンプト×128/256 steps×FP32参照・分離g144・分離g288の18生成も比較し、全てFP32参照の生成文と一致した。
その記録は `results/output_tune_validation/validation.json`。

既存の `test_persistent.py`（3テスト、O2/Ofast、数値一致・配分・チーム寿命など）と
`TEST_OPENMP=1 test_classifier.py`（1テスト）も通過。ログは `results/output_tune_validation/tests.log`。

常駐チームの仕組みは [persistent-openmp.md](../project/docs/persistent-openmp.md) を参照。
