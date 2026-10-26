# stay2 temporary investigation — 2026-10-20

Intel Core Ultra 5 225U、stories15M、元の実験の222入力トークンをreplay。
先頭1トークンを除外して各221トークンを平均、各条件10回、ブロック内で順序を乱数化。
ウォームアップは各条件1回を別ブロックに保存して集計から除外。
生成結果のトークン列・argmaxのハッシュは全full-forward条件で一致した。

## 分かったこと

- run_stay2.cのTIMING_ONLYがforward_startまで0にする不具合を修正した。
  全体時計は常に開始し、TIMING_ONLYでは計測用の追加バリアも除く。
  数値計算に必要なバリアは残す。通常版・詳細版・TIMING_ONLYで参照logits一致を確認。
  既存の詳細版CSVはTIMING_ONLY未使用なので、ゼロ開始時計の不具合が直接の原因ではない。
- 詳細計測版は各ステージに追加バリアがあり、計測コスト込みの結果だった。
  センサーあり10回ではP4_HTの全体中央値は詳細2.150ms、TIMING_ONLY 1.920ms。
  P4X4は詳細4.759ms、TIMING_ONLY 3.098ms。詳細版と通常版を同一視できない。
- センサー取得なしでも均等分割Switchingの劣化は再現した。
- 別のワーカー診断走行ではSwitchingのoutput 4000行/workerに対して、
  P側wall中央値0.486–0.512ms、E側0.832–0.841ms。
  起床待ち中央値は0.010–0.018msで、処理時間差の方が大きい。
  E側はスレッドCPU時間も約0.83–0.84msで、単にスケジューラから外されて待っているだけではない。
  固定P4X4のE側は同じ行数で約0.60msだった。
  休止からの周波数立ち上がり・キャッシュ状態・共有資源などは候補だが未分離。
- 診断介入としてoutputだけP worker重み2、E worker重み1にすると改善した。
  Pは5333/5334行、Eは2666/2667行。layersはP4_HTのまま。
  これは最適比率の確定ではなく、均等割り当てが制約になっている根拠。

## 最後の確認実験

結果: `results/probe_stay2_20261020_133855_042738/`
4条件を同一ブロックでランダムに交ぜて10回。センサー取得なし、TIMING_ONLY。

|条件|全体時間の中央値 (ms/token)|最小–最大|
|---|---:|---:|
|P4_HT固定|1.785|1.730–1.892|
|P4X4固定（均等）|3.310|2.073–3.640|
|Switching（均等）|2.149|1.988–4.976|
|Switching（output P:E=2:1）|1.704|1.665–1.772|

対応する反復どうしでは重み2がP4_HTに8/10回勝ち、時間比の変化の中央値は-4.94%。
均等Switchingには10/10回勝ち、時間比の変化の中央値は-21.18%。
独立した中央値同士の比と、反復内時間比の中央値は区別する。

全full-forward走行で同じ入力・同じargmaxを確認。合成モデルではlogitsのビット一致と
行範囲の欠落・重複なしを重み1/2で確認。

## 温度・周波数

センサーありの記録: `results/probe_stay2_20261020_133611_450528/`
`environment.json`にトポロジー、governor/EPP、バイナリ・ソースハッシュ、センサー一覧。
構成別・反復別の`sensors.csv`に単調時計、各CPUのscaling_cur_freq、温度、読めるthrottle count。
パッケージ温度の各走行の最大値は47–67℃、採取したcore/package throttle countは増加なし。
これは電力制限や短時間のクロック変動の不存在を保証しない。
`scaling_cur_freq`は50ms周期のスナップショットであり、output中の実効クロックではない。
RAPL energyはこの環境では読めず、電力は測定できていない。

初回のセンサーモニタはCPU6で走らせた（Eクラスタ4–7を共有するため交絡候補）。
取得処理にも負荷があったため、取得なしで追加実験・最終確認を行った。
現在のスクリプトは空いている最大CPU番号（この機械では13）をモニタに選ぶ。
`--monitor-cpu`で変更できる。センサーあり/なしの絶対時間差を温度だけに帰属しない。
システムのgovernor/EPP/電力設定は変更していない。

## 再実行

リポジトリルートから（通常版へ重み変更は入れていない）:

```bash
# センサーあり、詳細/全体のみ/outputだけ、各10回
python3 tmp/stay_probe/experiment.py --repeats 10 \
  --tokens results/stay2_output_20261020_133126_301350/P4_HT/profile.jsonl

# 最終確認と同じ4条件を交ぜる。重み2はSwitching_weight2だけに適用
python3 tmp/stay_probe/experiment.py --repeats 10 --modes clean \
  --no-sensors --seed 2718 --compare-weights \
  --tokens results/stay2_output_20261020_133126_301350/P4_HT/profile.jsonl

python3 tmp/stay_probe/trace_summary.py results/probe_stay2_日時
```

`--tokens`未指定ならP4_HTで一度生成して入力列を固定する。
CPU番号は`--p4 0,1,2,3 --p8 0,1,2,3,4,5,8,9`で変更可能。
`--p-weight 2`は全P4X4/Switchingのoutput配分を変更する。
`build_probe.py`は現在のrun_stay2.cから診断用Cを生成し、一時バイナリをコンパイルする。

`comparison.csv`: 条件ごとの中央値/MAD/最小/最大。
`trials.csv`: 実行順序、反復、全体時間、利用可能なステージ時間、トークンハッシュ、センサー集計。
`trace_*`: 時間比較とは別にワーカーの起床/開始/終了/スレッドCPU時間を採った走行。
`workers.csv`: trace_summary.pyで生成するワーカー別集計。traceは統計用10回に混ぜない。

`output-only`は入力embeddingから直接outputする切り分け用マイクロベンチ。
実際のlayers後の隠れ状態ではなく、キャッシュ状態やCPUの稼働履歴も変わる。
従って全体ベンチのoutput時間をそのまま予測するものではない。

未解決: 実効クロック/電力/キャッシュの寄与の分離、P4X4 layersの大きい変動、別日における再現性。
特に6 attention headsに8 workersを割り当てる構成では一部が仕事なしでバリアに参加する。
P4X4 layers全体の変動は、output配分変更では解決しない。
