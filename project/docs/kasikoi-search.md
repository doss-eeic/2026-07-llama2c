# kasikoi_serch

`AutoTuner.kasikoi_serch(config, budget=56, top_k=3, repeats=3, seed=42)`。
戻り値は独立確認で平均ms/tokenが最小だった`BenchmarkResult`。
`tokens_per_second`は確認試行の平均ms/tokenを逆数にした値で、
`raw_stderr`は最後の確認試行のログ。個別の実測値は`tuner.history`に残る。

1. SMT占有数とL2共有関係で配置をまとめる。この14CPU機では269代表になる。
2. 入力configのCPU配置を残しつつ、1〜14スレッドを各1回測定する。
3. 上位配置の追加・削除・入れ替え近傍と大域候補を探索する。
   予測は種類別の人数が近い実測3点の重み付き平均。5回に1回はランダム探索。
4. 探索予算の約1/4を上位配置の共通scheduleとgの調整に使う。
5. 上位候補と入力configの基準配置を、別のランダム順ブロックで測り直す。

これは旧ブランチのridge版`model_counts`の移植ではなく、
同じ「トポロジーで候補を整理し、人数中心に選ぶ」方針の簡略版。
全組合せの最適性は保証しない。対照値による時間の補正は行わない。
同型の物理CPU間の性能差は近似しており、入力した基準配置は代表として残す。

`budget`は探索測定の上限で、最初の人数全域の測定を含む。
別途ウォームアップ1回と、最大`(top_k + 1) * repeats`回の確認がある。
全探索時間・実行回数・確認結果は`tuner.last_search`から取得できる。

`config`の重み最適化・二段階argmax・モデル入力条件は固定する。
層内と出力は同人数で、各候補のCPU集合からOMP_PLACESを作り直す。
scheduleは入力値・static・dynamic,16・dynamic,64。
二段階argmax有効時はgの入力値に加えて96/144/288のうちdimを割り切る値を候補にする。
予算内で試すため、調整候補の全組合せを測るとは限らない。
異なるw/aの比較は別々の呼び出しで行う。

```python
from cpuprofiler import AutoTuner, BenchmarkConfig, BenchmarkRunner, CPUTopology

runner = BenchmarkRunner(binary_path="./build/run-fused")  # 先にビルドする
tuner = AutoTuner(CPUTopology(), runner)
config = BenchmarkConfig(
    cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11), threads=10,
    optimize_weights=True, two_stage_argmax=True,
)
best = tuner.kasikoi_serch(config)
print(best.config, best.tokens_per_second)
print(tuner.last_search["seconds"], tuner.last_search["measurements"])
```
