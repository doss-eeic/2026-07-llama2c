# 並列化単体・複合とトポロジー構成探索（2026-10-22）

480採用候補を実測し、QKV/FFN単独と併用の追加48セルも比較した。独立8ブロックの確認で最速平均だった候補は **pthread P4E4(E=2/2) 8→12人, S/D, 1行**、**824.03 tok/s**。この機械・stories15M・256steps・今回の有限空間に対する結果である。

トポロジー予測探索は48候補で、別ブロック最良値への平均latency差が **1.73%**、ランダム探索は **4.58%**。ただし3ブロックを再利用するreplayであり、60件の独立実測ではない。

## 計画と実施範囲

機械はCore Ultra 5 225U、P物理2個（SMT計4）、E8個（L2ごと4個）、LP-E2個。PのSMT有無、E0/2/4/6/8個、compact/spread、outputだけCPU追加を含む30配置に、管理方式・層分配・output分配・1/4行を交差した480候補。融合は採用探索では有効にし、2配置でQKV/FFNの4組合せを別に検証した。

採用空間と事前固定した探索法は [TOPOLOGY_SEARCH.md](../../src/cpuprofiler/TOPOLOGY_SEARCH.md)。`dev/wataru` のL2占有分割と `dev/cpu-profiler` のトポロジー取得を活かし、CPUの追加だけでなくSMTを外す入替・output人数変更も扱う。任意のCPU部分集合、重み、任意chunk、バリア実装は今回の空間外。

全実験でstories15M、temperature=0、seed42、同じprompt、steps256。256は上限で、詳細計測では区切りtokenで222 forward呼出しで終了し、先頭を除く221回を集計した。速度測定はprofileなし。各完全ブロック内で順番をシャッフルし、温度・周波数snapshot・負荷・電源状態と実行前後の時刻を保存した。未改造upstreamとqkv-ffnブランチを同じコンパイル条件で再ビルドして含めた。

|実験|候補/比較測定|control|warmup|合計|
|---|---|---|---|---|
|topology_adoption_20261022|1590|105|2|1697|
|topology_online_raw_20261022|48|49|2|99|
|topology_online_normalized_20261022|48|49|2|99|
|topology_confirm_20261022|120|16|2|138|
|topology_detail_20261022|21|6|2|29|

## QKV・FFN単独と併用

以下は各配置・方式でstatic/dynamicと1/4行を均等平均した、両方なしからの短縮ms/token。周辺効果とは違い、QKV単独とFFN単独は両方なしを基準にする。FFN処置はup/gate融合＋SwiGLU並列化を含む。両者を独立には分離していない。

|配置|方式|QKV単独|FFN単独|併用|相乗短縮|
|---|---|---|---|---|---|
|P4E4 spread|openmp|+0.0683|+0.0275|+0.1497|+0.0539|
|P4E4 spread|pthread|+0.0826|+0.0411|+0.1399|+0.0162|
|P2E8|openmp|+0.1135|+0.0547|+0.1802|+0.0121|
|P2E8|pthread|+0.1320|+0.0343|+0.1629|-0.0034|

相乗短縮=T(Qのみ)+T(Fのみ)−T(併用)−T(なし)。正なら単独短縮の合計より速い。今回も併用の平均改善は確認できるが、相乗の大きさは条件依存である。[全64対比較](fusion_contrasts.csv)には条件別の値、3ブロックの差、探索的bootstrap区間を保存した。3ブロックの区間を確証的な有意差として扱わない。

## 独立した上位確認

探索の補正順位上位4、生値順位上位4、方式/SMT別代表、従来control、実際のオンライン探索の推薦を測定前に固定し、別の8ブロックで比較した。下表は生の平均速度。S=static、D=dynamic（層16物理行、output32物理行）、Pの数字はSMT込み論理CPU数。

|候補|tok/s|ms/token|
|---|---|---|
|pthread P4E4(E=2/2) 8→12人, S/D, 1行|824.03|1.2138|
|qkvffn_branch|815.56|1.2262|
|openmp P2E6(E=4/2) 8→8人, D/D, 4行|814.23|1.2285|
|openmp P4E4(E=2/2) 8→8人, D/D, 1行|810.75|1.2336|
|pthread P4E4(E=2/2) 8→12人, S/D, 4行|810.80|1.2342|
|pthread P2E4(E=2/2) 6→12人, S/D, 1行|806.35|1.2404|
|pthread P2E4(E=2/2) 6→10人, D/D, 1行|804.51|1.2432|
|openmp P4E6(E=4/2) 10→10人, D/D, 1行|802.75|1.2472|
|pthread P2E4(E=2/2) 6→10人, S/D, 4行|799.62|1.2511|
|openmp P2E8(E=4/4) 10→10人, D/D, 4行|797.10|1.2568|
|openmp P4E4(E=2/2) 8→8人, S/S, 4行|795.77|1.2568|
|pthread P4E4(E=2/2) 8→8人, D/D, 1行|793.92|1.2596|
|pthread P2E4(E=4/0) 6→10人, D/D, 1行|782.77|1.2777|
|pthread P2E8(E=4/4) 10→10人, D/D, 1行|743.55|1.3484|
|upstream|726.72|1.3773|

最速平均候補の実行設定は [採用候補JSON](best_config.json)。「真の最適」の保証はなく、この確認自体でも複数候補から最良を選ぶバイアスが残る。

upstreamに対する速度改善はブロック内のlatency比の幾何平均で **13.43%**、探索的95%区間 [11.85, 15.28]%。各比較対象の未改造ソースを同じコンパイル条件でビルドした実装全体の比較であり、差を単一の要因に帰属しない。

qkvffn_branchに対する速度改善はブロック内のlatency比の幾何平均で **1.03%**、探索的95%区間 [-0.08, 1.83]%。各比較対象の未改造ソースを同じコンパイル条件でビルドした実装全体の比較であり、差を単一の要因に帰属しない。区間がゼロをまたぐため、この比較の改善は確定しない。

## 探索アルゴリズムの比較

同じ480候補を対象に、random、最良点からの1因子近傍探索（beam幅1、5回に1回再始動）、カーネルのみのridge予測器、トポロジーを含むridge予測器を比較した。予測器は最初の12点を特徴空間で分散配置し、指定した相互作用・E数の二次項も使う。その後は予測平均−0.05×探索指標の点を選び、5回に1回は一様探索する。推薦は実際に照会した構成の最良観測値に限定する。

3ブロックのleave-one-block-outで、2ブロック平均を探索への観測、残り1ブロックを評価に使った。seed0〜19、各12/24/48/96照会。以下は生のms/tokenでの結果。regretは「推薦構成の評価ブロックlatency／その評価ブロック全480候補の最小latency−1」。単一ブロックの最小値も測定雑音で楽観的になり得る。

|探索法|候補照会|平均差%|p90差%|3%以内到達率|
|---|---|---|---|---|
|random|12|10.79|13.50|16.7%|
|random|24|5.76|11.68|28.3%|
|random|48|4.58|8.92|46.7%|
|random|96|4.36|7.43|48.3%|
|beam|12|11.66|15.99|11.7%|
|beam|24|6.62|11.68|30.0%|
|beam|48|4.11|7.81|55.0%|
|beam|96|3.03|5.58|65.0%|
|model_no_topology|12|5.22|10.04|38.3%|
|model_no_topology|24|4.21|8.93|51.7%|
|model_no_topology|48|3.72|6.69|51.7%|
|model_no_topology|96|2.39|5.36|76.7%|
|model|12|7.24|13.57|28.3%|
|model|24|3.67|7.43|66.7%|
|model|48|1.73|2.92|96.7%|
|model|96|1.70|2.92|100.0%|
|exhaustive|480|1.70|2.92|100.0%|

48/480候補（候補照会90%削減）でトポロジー予測器は3%以内へ **96.7%** 到達した。予測器からトポロジーを除くと性能が落ち、このデータでは構成の特徴を使う意義が見える。ただし12点ではトポロジーなし予測器より良くなく、初期設計の費用は小さくない。480候補の総当たりも訓練ブロックで選んで評価し、評価ブロック最良を直接推薦していない。

48照会時の予測器処理は平均0.071 CPU秒。候補実行時間は訓練2ブロックの合計で平均34.07秒だった。これらはcontrol・warmup・ビルド・独立確認を含まない。

[全照会trace](raw_search_trace.csv)、[各fold/seedの結果](raw_search_runs.csv)、[集計](raw_search_summary.csv)、[評価プロトコル](raw_replay_protocol.json)。60件は同じ3ブロックから作った探索順の反復であり、60回の独立ベンチマークとは解釈しない。

## 粗い間隔の補正の失敗と実測オンライン探索

当初は16セルごとのcontrolを時刻で線形補間して割る方針だった。しかし第1ブロックのcontrolは 1.226〜1.697ms/token と揺れ、一時的なcontrolの遅れが周辺候補を過度に有利にした。温度・負荷のsnapshotは残したが、原因は断定しない。

|探索法|照会|生値regret%|補正後regret%|
|---|---|---|---|
|random|48|4.58|14.31|
|beam|48|4.11|13.92|
|model_no_topology|48|3.72|14.56|
|model|48|1.73|13.62|
|exhaustive|480|1.70|13.49|

補正後のheld-out最良値自体も変わるため、上の2列のregretは同じ物差しではない。この16セル間隔の単純補正を主な採用基準にはしない。生値も評価する方針への変更は感度分析を見てからの判断であり、事前固定とは区別する。補正値・失敗結果・最初の計画も消さずに保存した。

|実測探索の目的|48候補の推薦|確認時tok/s|確認最速とのlatency差|候補実行秒|control等込み秒|
|---|---|---|---|---|---|
|raw|openmp P2E8(E=4/4) 10→10人, D/D, 4行|797.10|3.54%|15.99|32.12|
|normalized|pthread P4E4(E=2/2) 8→12人, S/D, 1行|824.03|0.00%|16.09|32.21|

各オンライン実測は候補48回＋前後control49回＋warmup2回＝99回。全480候補の既存速度表は探索器へ渡していない。上の秒数はsubprocess実行時間の合計で、コンパイル、Python処理、ファイル保存、独立確認の費用を含まない。replayでは1観測に訓練2ブロックを平均するため48照会=候補96測定で、オンラインの1候補1回と同じ測定条件ではない。

今回のオンライン探索では、粗い間隔の補正と違って候補ごとの前後にcontrolを置いた補正版が、独立確認で最速平均の構成を見つけた。replayとオンラインは対照の間隔・測定回数・時期が異なるため、補正全般が悪いとも、生値目的が常に良いとも結論しない。各オンライン目的はseed0の1回だけである。トポロジー予測探索48点→生値と局所対照比の両方で上位候補を残す→独立確認、を今回の推奨とする。一点の比だけで採用を確定しない。次に試すならcontrolの複数反復/robust平滑化と、最良点の再測定を予算内に組み込む。これらの改良案は今回まだ検証していない。

## 再現・検証と限界

標準ライブラリのみで測定・探索・集計を実行できる。生データの欠損/重複、全stdout hash、summaryとCSVの値、ソースhash、controlの前後挟み込みを検証した。Cの数値一致と探索の因果性・ridge更新・補正/相互作用の算術を含むテストの結果は [tests.log](tests.log)。

全実験を再検査した結果は [final_audit.json](final_audit.json)。`python3 results/topology_adoption_20261022/verify.py` で再検査できる。

## 演算別の追加計測

詳細計測は [topology_detail_20261022](../topology_detail_20261022/stage_summary.csv) に別保存した。同じCPU配置とカーネルでQKV/FFNの4組合せを測り、上位構成も比較した。値はms/token、各3回平均。FFNはSwiGLUとdownを含む。層別CSVと圧縮JSONLも保存した。profileありの時計取得・記録・バリア費用を含むため、速度ランキングに混ぜない。layer_totalと各stageは重複するため合計しない。

|構成|QKV|FFN|output|
|---|---|---|---|
|pthread P4E4(E=2/2) 8→12人, S/D, 1行 Q1F1|0.1711|0.3180|0.5184|
|openmp P4E4(E=2/2) 8→8人, D/D, 1行 Q0F0|0.2269|0.3654|0.5434|
|openmp P4E4(E=2/2) 8→8人, D/D, 1行 Q0F1|0.2334|0.3222|0.5477|
|openmp P4E4(E=2/2) 8→8人, D/D, 1行 Q1F0|0.1389|0.3734|0.5481|
|openmp P4E4(E=2/2) 8→8人, D/D, 1行 Q1F1|0.1504|0.3282|0.5459|
|openmp P2E8(E=4/4) 10→10人, D/D, 4行 Q1F1|0.1497|0.3447|0.4848|
|openmp P2E6(E=4/2) 8→8人, D/D, 4行 Q1F1|0.1402|0.3290|0.5238|

tok/sはC側の既存タイマーで先頭反復を除き、forwardに加えてsampling/decode/出力を含む。モデル読込・ビルド・プロセス起動を含まない。タイマーはミリ秒単位である。個々の構成の短い測定に雑音があり、温度・周波数・直前の構成の影響を完全には分離できない。今回の探索空間で有望でも、大型モデル、長い文脈、別CPU、複数リクエストでの最適性は未確認。bootstrap区間は少ないブロックでの探索的な値で、多重比較を補正していない。

```sh
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --output results/NEW_sweep --repeats 3
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_sweep --raw-replay
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --online --objective raw --budget 48 --output results/NEW_online_raw
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --online --objective normalized --budget 48 --output results/NEW_online_normalized
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_sweep --no-replay --include-online results/NEW_online_raw --include-online results/NEW_online_normalized
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --confirm results/NEW_sweep --output results/NEW_confirm --repeats 8
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_confirm --no-replay
PYTHONPATH=src python3 -m cpuprofiler.report_topology results/NEW_sweep results/NEW_confirm results/NEW_online_raw results/NEW_online_normalized
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --confirm results/NEW_sweep --output results/NEW_detail --repeats 3 --profile
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_detail --no-replay
PYTHONPATH=src python3 -m cpuprofiler.report_topology results/NEW_sweep results/NEW_confirm results/NEW_online_raw results/NEW_online_normalized --detail results/NEW_detail
```

結果フォルダにはplan、machine、environment、実行設定、状態snapshot、stdout/stderr、生CSV、ソースsnapshot、ビルドコマンド、モデル/tokenizer/バイナリhashを残した。`analysis_source/` と `analysis_provenance.json` は集計時のPythonソースと入力hash。`report_metrics.json` とこの報告は `report_topology.py` から再生成できる。比較ブランチの解決済みcommitは `builds/*/provenance.json` に記録。
