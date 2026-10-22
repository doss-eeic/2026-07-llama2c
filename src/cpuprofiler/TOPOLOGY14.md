# 全14論理CPUのトポロジー探索：実験計画と実行仕様

前回の「P物理コア2個を必須、Eは偶数個」等の制約を外す。
層内1〜14人を全て実測する。Pなし、P片側のみ、片側SMT、奇数E、LP-Eの層内利用を含む。
対象機械はP物理2個×SMT2、Eは4個ずつL2共有する2群、LP-E2個。

## 同値化と空間

各P物理コアと各E L2群の「層内人数l、output人数o」を記録する（0≤l≤o≤群の容量）。
同じ種類のP物理コア間、E L2群間、各群内のCPUを入れ替えた配置を同値とみなす。
LP-EをPや通常Eと混同しない。PのSMT関係と、Eが同じL2か別L2かを保持する。
物理的な同種CPUが実測でも完全同速だという仮定ではないため、交換配置も一部測って点検する。

- 固定maskは6(Pの人数分割)×15(Eの人数分割)×3(LP人数)−空mask=269種類。
  元の2^14−1=16,383部分集合を全て構造上の代表へ写像できる。
- 層内/outputの入れ子maskは21(Pのl/o分割)×120(Eのl/o分割)×6(LP分割)
  −層内0人の270種類=14,850種類。元は3^14−2^14=4,766,585組。
- backend2×層static/dynamic16×output1/4行=8設定を交差し、最終空間118,800候補。
  QKV/FFN融合、output dynamic32物理行、p_weight1、Attentionは層内チーム全員に固定する。
  任意chunk・バリア方式・重み・CPU rank順列の探索は今回含めない。

## 測定と比較

1. **固定配置のoracle**: 全269代表×2 backend、層dynamic16/outputdynamic32、1行を3完全ブロック。
   1〜14人の全人数を含み、その中の有限空間最良と探索法を比較できるデータを作る。
   同値化を点検する交換配置、前回最良、未改造upstreamとqkv-ffnも測る。
   このoracleは118,800候補全体の総当たりではない。
2. **広い空間の実測探索**: 以下5手法×seed0/1×112候補。
   全手法が最初に1〜14人の固定配置を1つずつ測定する。
   方法とseedを各ラウンド内でシャッフルし、推論は1つずつ実行する。
   探索器へoracleの速度を渡さない。目的は生のlog(ms/token)。
3. **独立確認**: 各探索の上位3、oracle上位、前回最良と14人境界を測定前に固定し、8ブロックで再測定する。
   同値化の精査として、各探索上位1とoracle上位2は交換したCPU代表も確認する。
   全14人は全8kernel設定×標準/交換の2順序も独立確認に含める。
   真の118,800候補最良は未観測なので、確認した候補集合内の比較として報告する。

### 比較する探索法

- random: 同値類の一様探索。生のCPU部分集合一様とは分布が異なる。
- beam14: 最初の14点後、深さ1〜14へ各7照会。上位3層内配置からCPUを1個追加する。
  各層内配置に対応するoutput拡張とカーネルも候補にする。単純な追加だけの手法の限界も調べる。
  固定oracleのreplayで前線が尽きた際は同人数、さらに必要なら全体から再始動する。
- hierarchical: 最初の56点は人数で層別化した固定配置、その後は良かった層内配置を優先し、
  output拡張とカーネル設定を試す。層内配置の早期枝刈りで良い組合せを落とす可能性がある。
- model: P論理数/P物理数/SMT、E数/L2使用数/占有集中度、LP数、層内/output人数、
  曲率・人数切替・カーネルとの相互作用を40特徴で逐次ridge推定する。
  グローバル一様候補と最良4点のグラフ近傍から最大256候補を作り、平均−0.05×探索指標で選ぶ。
  5回に1回は一様探索する。係数ridge1。未測定の予測点だけでは採用しない。
- model_counts: 同じ予測器と提案方式だが、SMT物理構造とL2占有分割を除きP/E/LPの合計人数で推定する。
  人数だけの粗いヒューリスティックと、キャッシュを保持する表現の違いを比較する。

固定oracleのreplayではleave-one-block-out（2ブロックを観測、残り1ブロックを評価）、
seed0〜9、予算28/56/112/168を比較する。基準となる有限oracleを明記し、
同じ実測を再利用した探索順の反復を独立実験と数えない。

stories15M、最大256steps、greedy、seed42、同じprompt。前回は222 forwardで区切りtokenに到達した。
controlは16候補ごとに測り、温度・周波数・負荷・電源状態を保存する。
前回の粗いcontrol補正の失敗を踏まえ、今回は生値を目的にする。controlは変動の監視用。
ソース、ビルド、実行設定、stdout/stderr、全探索traceと失敗を保存する。

## 実行

```sh
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology14 oracle --output results/NEW_oracle --repeats 3
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology14 search --output results/NEW_search --budget 112 --seeds 2
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology14 --oracle results/NEW_oracle --search results/NEW_search
PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology14 confirm --source results/NEW_search --output results/NEW_confirm --repeats 8
PYTHONPATH=src python3 -m cpuprofiler.analyze_topology14 --oracle results/NEW_oracle --search results/NEW_search --confirm results/NEW_confirm --no-replay
PYTHONPATH=src python3 -m cpuprofiler.verify_topology14 results/NEW_oracle results/NEW_search results/NEW_confirm
```

出力先は新規フォルダ。前回のデータは上書きしない。
