# 採用候補の探索と並列化ベンチマーク

## 実験の狙い

`dev/wataru` の `bench_l2_scaling.sh` はL2占有分割を明示的に列挙しており、
そのcompact／spreadの考え方を採用候補へ取り込む。`dev/cpu-profiler` と現行の
`autotuner.py` は固定CPUリストから開始し、上位にCPUを1個追加する。
CPU数は増やせても、SMTを外してEを増やす入替、L2配置、
層内とoutputのCPU切替、融合・分配の相互作用は探索しにくい。
`topology_search.py` は実際の `BenchmarkConfig` に落とせる有限集合を作り、その中を探索する。
従来APIは変更せず、新しいask/tell探索器を追加した。

## 探索空間（計測前に固定）

対象機 Intel Core Ultra 5 225U のsysfsと現在のaffinityから、SMT兄弟、L2共有、
最大周波数を収集する。この機械ではP物理2個、E8個（4個ずつL2共有）、LP-E2個。
CPU番号をクラス判定に埋め込まない。SMTが見えない、周波数が欠ける、複数socketなどの
曖昧な環境では止め、明示的な `Machine` を使う。周波数は分類の補助であり性能推定値ではない。

- P物理コアを全て使い、SMTは両方使用／両方不使用。
- Eは0/2/4/6/8個。L2へ詰めるcompact／L2をまたぐspreadを重複排除。
- LP-Eは原則output追加用。全14を層内でも使う境界構成も含む。
- Eを4/8個使う層内構成から、outputをE全体、P-SMT+E全体、全14へ拡張。
  層内CPUはoutput CPUの部分集合に制限する。実装がそのまま対応するprefix順を保存。
- 30種類の層内→output配置 × OpenMP/pthread × 層static/dynamic16 ×
  output static/dynamic32 × output1/4行 = **480候補**。
- dynamic32の単位は物理行。4行カーネルでも32行。Attentionは層内全員で静的分配。
- 本探索ではQKV/FFN融合を有効。既存の要因実験で両方の平均改善が確認されているため。
  その仮定を今回も確認するため、P4E4-spread／P2E8の2配置でQKV単独・FFN単独・
  併用・両方なしを管理方式／分配／1・4行と交差させる。追加48セルで計528セル。
  FFN要因はup/gate融合とSwiGLU並列化を含む処置で、純粋な融合だけの効果ではない。

任意の14CPU部分集合、全CPU順列、重み、バリア実装、任意chunk数まで広げない。
採用時に指定しそうな構成を比較するための範囲であり、全実装・全マスクの大域最適を主張しない。
LP-Eだけを層内で使う構成、P物理を1個に減らす構成は今回の採用空間に含めない。
4行128物理行は既存の比較結果を参照し、今回の空間では32行に固定する。

## 測定設計

`benchmark_topology.py` は全528セルと未改造upstream／未改造qkv-ffnブランチを
独立の3完全ブロックで測る。ブロック内シャッフルseed=20261022+block。
stories15M、256steps、greedy、seed42、prompt=Once upon a time。
比較ブランチはビルド時にcommitを解決し、ソースをgit showで保存して同じコンパイラ設定でビルドする。
測定用バイナリは結果ディレクトリ内に置き、既存buildバイナリを書き換えない。

バックエンド毎にwarmup後、16セルごとに固定control（P4E4 spread、OpenMP、融合、
dynamic16/output dynamic32、1行）を測る。各runの前後に温度、周波数snapshot、
loadavg、/proc/stat、電源状態を記録。control間の時刻線形補間でms/tokenを割った
相対latencyと生値の両方を保存する。これは緩やかな共通変動の補正であり、
セル固有の発熱・直前の構成・突発負荷の影響を消すものではない。

速度測定はprofileなし。詳細計測は別実験。生成文SHA256は全runで照合する。
ソース、ビルドコマンド、バイナリhash、モデル/tokenizer hash、実行設定、
標準出力、標準エラー、測定順、時刻と失敗を残す。テストや別ベンチマークとは並走しない。

## 試す探索アルゴリズム

1. **random**: 同じ480候補から重複なし一様抽出。
2. **beam**: 最良観測点の1因子近傍を探索。CPU配置の入替を1操作として許し、
   5回に1回はランダム再始動。従来の追加だけの探索より強い比較相手とする。
3. **model_no_topology**: カーネルとその相互作用だけの予測器。トポロジー特徴を除く比較。
4. **model**: トポロジーとカーネルを使う逐次探索。
   最初の12点は特徴空間のfarthest-firstで分散配置（速度値は見ない）。
   SMT/E/LP-E比率、L2の使用数と占有集中度、E数の二次項、output切替、方式、分配、
   1/4行と、その指定済み相互作用を特徴とする。
   log相対latencyをridge=1で逐次推定し、予測平均−0.05×予測標準偏差が最小の未観測点を選ぶ。
   5回に1回は一様探索。Sherman–Morrison更新で行列の再逆行列化を避ける。
   係数・初期点数・探索率は今回の結果を見る前に固定。

評価済みの点から観測値が最良の構成を推薦し、未観測点の予測値だけで採用しない。
モデルは真のBayesianな誤差保証を与えない。標準偏差はridge特徴空間の探索指標であり、
測定雑音に校正した信頼区間ではない。強い相互作用を全て表現する保証もない。

## 評価方法（先読み防止）

3ブロックのleave-one-block-outで、2ブロックを探索観測、残り1ブロックを評価専用にする。
探索器はaskしたセルの訓練値だけをtellで受け取り、未照会・評価ブロックの値を参照できない。
各手法をseed0〜19、budget12/24/48/96で比較する。seedは独立実測の反復ではなく、
有限データ上の探索順の反復。held-out最良候補とのlatency regret、1%/3%以内到達率、
照会数、探索CPU時間を全traceとともに保存する。全探索も比較基準として同じデータで評価する。
正規化に使うcontrolの測定費用は候補照会数とは別に必ず示す。

全480候補を測るのは今回の探索器評価用oracleを作るため。運用時の削減効果は、
候補照会＋control＋warmup＋確認の費用として評価する必要がある。
既存の2配置64セルに限定した結果を、今回の480候補全体に一般化しない。
上位を別の独立ブロックで確認し、探索時のwinner's curseと時系列変動を点検する。
単体短縮・併用短縮・相互作用は同じブロック内のms/token差で計算する。

## 実行

リポジトリルートから（追加Pythonパッケージ不要）:

```sh
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --output results/topology_adoption_20261022 --repeats 3
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/topology_adoption_20261022
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --online --budget 48 --output results/topology_online_20261022
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --confirm results/topology_adoption_20261022 --output results/topology_confirm_20261022 --repeats 8
```

`topology_search.SearchPolicy` の `ask()` → 実測 → `tell(index, log相対latency)` が
オンライン利用口。`benchmark_topology.config_for` で同じ候補を実行設定に変換できる。
オンライン実測モードでは各候補の前後にcontrolを置き、後のcontrolを次の前controlと兼用する。
48候補＋49control＋2warmup=99回。完全ブロックの評価用oracleは探索器に渡さない。
`--objective raw` で生のms/token、`--objective normalized` で前後control比を目的にする。
今回の結果では粗い16セル間隔の補正に問題が出たため、生値でもreplayを評価した。
候補ごとの局所controlによるオンライン探索とは区別する。詳細は結果報告を参照。
出力先は新規ディレクトリを指定する。既存結果の上書きは禁止。

今回の数値・判断・生成済みデータへの入口は `project/docs/AI向け_20261022_構成探索.md` を参照。
