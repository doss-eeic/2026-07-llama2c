# 融合・二段階argmaxを揃えたoutput分配の比較

2026-10-26 / Core Ultra 5 225U / stories15M。

**両側にQKV/FFN融合・hugepage・二段階argmaxを入れて測り直した。
重み付き10→10人は同条件の分配OFFより約1.7%遅く、9→10人では人数変更だけで約3.8%、
重み付きで約4.8%遅かった。今回の候補では分配方式による改善を確認できなかった。**

以前の「FP32版989対master1575」という比較は最適化の有無が違った。
その差を分配・人数切替のオーバーヘッドとして扱わず、以下の条件を揃えた比較で評価する。
旧FP32測定は[別の記録](weighted-output-benchmark.md)に残した。

## 揃えた条件と実装

基準はmaster `256b0b2445ecd2c8c88d085fdf616313d5df91f3`。
そこから推論本体と融合・classifierを取り込み、最終outputの分配だけを追加した。

|条件|master本体|分配機能付き本体|
|---|---|---|
|QKV融合・FFN up/gate/SwiGLU融合|同じ|同じ。`src/fused.h`は基準コミットと同一|
|重みのhugepage配置|`-w 1`|`-w 1`|
|二段階argmax|`-a 1`|`-a 1`|
|量子化グループサイズ|`-g 144`|`-g 144`|
|量子化方法・誤差境界・候補のFP32再計算|基準実装|同じ数式・加算順序|
|ビルド|GCC 13.3 / `-Ofast -march=native -fopenmp`|同じ|
|モデル・tokenizer・prompt・seed|stories15M / `Once upon a time` / 42|同じ|
|生成|`-n 256 -t 0`|同じ|

単にFP32 output用関数を追加しただけでは、`-a 1`の経路は変わらない。
今回は二段階argmaxの**全語彙int8スコア計算と最大下界のreduction**へ人数・行分配を適用した。
入力の誤差境界準備、候補のFP32再計算、samplingは基準どおりである。
`-a 0`のFP32 outputにも同じ人数・重み指定を適用できるが、今回の速度表は全て`-a 1`。

比較する2バイナリは毎回ソースからビルドした。分配付き本体のOFFと、
未変更master本体も同条件で比較し、共通処理の移植に伴う差を確認した。
測定時HEADと未コミットのソース変更はメタデータで区別し、実際にビルドした各C/ヘッダのSHA256を保存した。

## 4つの分配モード

|モード|環境変数|int8スコア計算の分配|
|---|---|---|
|OFF|`LLAMA_OUT_THREADS=0`, `LLAMA_OUT_P=0`|元の`parallel for schedule(runtime)`|
|人数指定のみ|`LLAMA_OUT_THREADS=N`, `LLAMA_OUT_P=0`|同じruntime schedule、outputだけ人数Nを明示|
|手動均等分配|`LLAMA_OUT_THREADS=N`, `LLAMA_OUT_P=P`, `LLAMA_OUT_WP=1`|rankごとに連続する均等な行範囲|
|重み付き分配|上記で`LLAMA_OUT_WP=2`など|先頭P rankが他rankのW倍の行を担当|

いずれも通常のOpenMP並列領域を使い、以前の常駐チーム実験とは異なる。
人数指定のみの9→9／10→10は、出力の人数を明示するが実人数は切り替わらない。
9→10／10→9は実際の人数変更を含む。
P数は**Pコア上の論理CPU rank数**であり、SMTを含む。OMP_PLACESの順序で指定する。

## 探索と確認

全て逐次実行し、各巡の実行順をseed 42でシャッフルした。
`OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=close`、原則`OMP_WAIT_POLICY=ACTIVE`。
CPU順・層内人数・output人数・schedule・spinが同じ組合せを、比較対象として明示的に残した。

|段階|構成数|反復|本測定数|
|---|---:|---:|---:|
|粗探索|128|2|256|
|schedule・spin等の調整|108|3|324|
|上位＋対応するOFF/master等の確認|30|20|600|
|主要条件の追加確認|13|20|260|
|合計|||1440|

粗探索ではP2+E8、SMT込みP4+E8、既存masterの9人構成、低速側CPU込みの構成を評価した。
層内6/8/9/10/12人、output8/9/10/12/14人などの候補を試し、重みは1/2/3。
調整段階では`guided,4`・`guided,16`・`static`と、spin未指定・1000・10000を評価した。
列挙した候補内の探索であり、全CPU部分集合を網羅した最適化ではない。

600回の確認では8巡目付近から全体が減速したため、主要13条件を別途確認した。
追加確認では最初の2回のwarmupに加え、masterを30秒以上連続実行してから計測を始めた。
warmupは本測定に含めない。以下はこの**追加確認の各20回**を主な評価とする。

代表tok/sは`1000 / 平均ms/token`。tok/s算術平均、最小・最大値も生データに保存した。
元の生成ループどおり最初のtokenとモデル準備はタイマー外で、`-n 256`は終了トークンによる早期終了を含む上限。

## 結果1: 人数を変えず、分配だけを変える

CPU順A=`(0,2,4,5,6,7,8,9,10,11)`、層内10人、`guided,16`、ACTIVE、spin未指定。
手動分配のP数は2、すなわちCPU0・2がP側。

|case ID|モード|output人数|tok/s|
|---|---|---:|---:|
|`6c9eb55de4ba`|未変更master本体|10|1544.37|
|`196c914323a8`|分配付き本体・OFF|10|1540.07|
|`51b2eaa4d025`|人数指定のみ|10|1553.06|
|`a9438ef21917`|手動均等分配、重み1|10|1512.66|
|`3a689d9848c5`|P側2倍の重み付き分配|10|1513.70|

重み付きは対応するOFFに対し **−1.71%**（paired bootstrap 95%区間 −3.07〜−0.28%）。
手動均等分配に対しては **+0.07%**（−1.22〜+1.28%）で、P側を2倍にする利点を確認できない。
人数指定のみはOFFに対し+0.84%だが区間は−0.56〜+2.25%で、明確な差とは判定しない。

## 結果2: outputの人数を増やす

CPU順は同じA。層内9人、output10人、`guided,16`、ACTIVE、spin10000。
P側はCPU0・2。OFFではoutputも先頭9人で処理する。

|case ID|モード|層内→output|tok/s|
|---|---|---|---:|
|`fb512a82d972`|未変更master本体|9→9|1525.72|
|`a67d31679f1b`|分配付き本体・OFF|9→9|1520.47|
|`40808bb7c502`|人数指定のみ|9→10|1462.12|
|`4fca78fae4be`|手動均等分配、重み1|9→10|1432.74|
|`ec26e01c5072`|P側2倍の重み付き分配|9→10|1446.81|

対応するOFFに対して、人数指定のみは **−3.84%**（−4.95〜−2.67%）、
重み付き分配は **−4.84%**（−6.30〜−3.40%）。
重み2は手動均等分配より+0.98%だが、区間は−0.72〜+2.80%で改善は確定できない。

## 結果3: masterの既存最良構成

CPU順B=`(1,2,3,5,6,7,8,10,11)`、層内9人、`guided,16`、ACTIVE、spin未指定。
これはmasterの`check_top()`に含まれる構成。

|case ID|モード|tok/s|
|---|---|---:|
|`525b83f91361`|未変更master本体|**1569.60**|
|`2ef7662cd09f`|分配付き本体・OFF|1567.38|
|`0aa853b54577`|人数指定のみ、9→9|1564.60|

OFFはmasterに対し−0.14%（−0.95〜+0.75%）で、移植後のOFFに明確な退行は確認できない。
9→9の明示指定はOFFに対し−0.18%（−1.06〜+0.74%）。人数指定の分岐自体が大きな損失を生む結果ではない。
最良の重み付き構成1513.70とmaster既存最良1569.60との差は−3.56%だが、
この比較はCPU集合・人数が異なるため、分配単体の効果は前の同条件表で判断する。

## 「オーバーヘッドが大きいか」の評価

- 最適化を揃えると、旧FP32版との約37%という大差はなくなる。その差を分配コストとは呼ばない。
- 同人数を明示するだけの経路はOFFとほぼ同じだった。
- 実際に人数を変えた9→10構成は、runtime scheduleを維持しても遅かった。
- 固定の重み付き行分配は、今回のint8スコア計算では既存のguided分配を上回らなかった。

従って、現状は既存masterの非分離構成を推奨する。
ただし人数変更の差には、チーム構成、計算の並列度、待機、仕事の偏りなどが含まれる。
重み付き分配とguidedの差にも負荷配分・キャッシュ等の違いが入る。
空の並列領域やバリア単体の時間を測ったものではなく、純粋な切替オーバーヘッドだけの数値ではない。
FP32 outputで有効だったP:E=2:1を、そのまま二段階argmaxへ移しても同じ利点は得られなかった。

## 実装変更

- `run.c`: masterの融合・hugepage・二段階argmaxを取り込み、FP32最終outputの分配関数も維持。
- `src/fused.h`: masterのQKV/FFN融合をそのまま使用。
- `src/classifier.h`: 既存のスコア計算を行関数へ分け、OFF／人数指定／手動分配のどれでも同じ処理を使う。
  最大下界はOpenMPの`reduction(max:L)`で集計し、候補のFP32再計算は変更しない。
- `src/output_split.h`: 設定の検証と行範囲計算。要求人数より実チームが小さい場合も、実人数で全行を配分する。
- `benchmarks/optimized_output.py`: 同じ最適化・入力条件で、master、OFF、人数指定、均等、重み付きを比較する。
- 旧`benchmarks/weighted_output.py`は`3bd572ed`のソースを明示的にビルドし、旧FP32測定を再現できるようにした。

`LLAMA_OUT_THREADS=0`が既定で、人数切替・手動分配は無効。
`OMP_SCHEDULE`は今回の融合版で有効である。旧FP32版の「指定しても分配が変わらない」という条件とは異なる。

## 再現

リポジトリルートから実行。CPU番号はこの225Uに対応する。

```sh
python3 -m unittest discover -s tests -p 'test_*output.py' -v
gcc -Ofast -march=native -fopenmp run.c -lm -o /tmp/run-optimized-output

# 同条件のOFF。outputも10人で処理する。
env -u GOMP_SPINCOUNT -u OMP_THREAD_LIMIT \
  OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=ACTIVE OMP_SCHEDULE='guided,16' \
  OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
  LLAMA_OUT_THREADS=0 LLAMA_OUT_P=0 LLAMA_OUT_WP=2 \
  taskset -c 0,2,4,5,6,7,8,9,10,11 \
  /tmp/run-optimized-output /path/to/stories15M.bin -z tokenizer.bin \
  -n 256 -t 0 -s 42 -i 'Once upon a time' -w 1 -a 1 -g 144
```

同じコマンドで`LLAMA_OUT_THREADS=10 LLAMA_OUT_P=2`にすると重み付き10→10。
`LLAMA_OUT_P=0`なら人数指定だけ、`LLAMA_OUT_P=2 LLAMA_OUT_WP=1`なら手動均等分配になる。
9→10を再現する場合は`OMP_NUM_THREADS=9`、`GOMP_SPINCOUNT=10000`とし、その他の条件を保持する。

探索全体と追加確認:

```sh
python3 benchmarks/optimized_output.py --model /path/to/stories15M.bin \
  --phase coarse --repeats 2 --output results/replay_optimized_coarse
python3 benchmarks/optimized_output.py --model /path/to/stories15M.bin \
  --phase tune --repeats 3 --previous results/replay_optimized_coarse \
  --output results/replay_optimized_tune
python3 benchmarks/optimized_output.py --model /path/to/stories15M.bin \
  --phase final --repeats 20 --previous results/replay_optimized_tune \
  --output results/replay_optimized_final
python3 benchmarks/optimized_output.py --model /path/to/stories15M.bin \
  --phase confirm --repeats 20 --warmup-seconds 30 \
  --previous results/replay_optimized_final --output results/replay_optimized_confirm
```

比較元は既定でmaster `256b0b24`。モデルとtokenizerのハッシュ、実際の全ソースのハッシュ、
コンパイラ、全CLI設定、CPU順、実行前後のAC・温度・周波数・loadavgを保存する。
各プロセスのOpenMP・LLAMA_OUT関連の親環境変数を除去してから条件を設定する。

## 検証と動作状態

本測定1440回すべてで、生成文のSHA256は未変更masterと一致した。
追加テストはO2/Ofastで、ゼロ重み、ランダム、同値・僅差の候補、極小・大きな入力、
グループサイズ24/72/144/288、全分配モードのargmaxと保持したlogitsをFP32参照と比較した。
実チームを2人に制限した場合、端数行、CPU配置、無効設定の拒否も確認した。
OpenMPなしのビルドも通過した。

AC状態は全実行前後で1のまま。最初の600回の確認では、各巡の全構成集計が
1〜7巡目の約1492〜1527 tok/sから、9巡目以降は約1315〜1340 tok/sへ低下した。原因は未特定。
追加確認では各巡の全構成集計は1504.19〜1536.36 tok/sで大きな低下はなく、
絶対速度が異なっても「重み付き10→10がOFFより約1.7%遅い」という方向は一致した。
巡集計は動作状態を見る目安であり、特定構成の性能値ではない。
追加確認の回復をウォームアップの因果効果として断定はしない。

改善率の区間は、同じ巡番号の対応を保って10000回再標本化したpaired bootstrap、seed 42。
別のCPU・モデル・g設定での最適性や、分配コストの一般的な大きさまでは主張しない。

## 記録

- [粗探索](../results/optimized_output_coarse_20261026/ranking.json)、[調整](../results/optimized_output_tune_20261026/ranking.json)。
- [最初の確認CSV](../results/optimized_output_final_20261026/summary.csv)、[比較・区間](../results/optimized_output_final_20261026/comparison.json)。
- **[追加確認CSV](../results/optimized_output_confirm_20261026/summary.csv)**、[全実行](../results/optimized_output_confirm_20261026/runs.jsonl)、[比較・区間](../results/optimized_output_confirm_20261026/comparison.json)。
- [メタデータ](../results/optimized_output_confirm_20261026/metadata.json)、[検証ログ](../results/optimized_output_confirm_20261026/tests.log)。

全段階に`cases.json`・`runs.jsonl`・`ranking.json`・`warmup.jsonl`があり、設定とstdout SHA256を追跡できる。
