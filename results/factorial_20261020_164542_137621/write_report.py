from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parent
RES=ROOT.parent

def read(path):
 with path.open() as f:return list(csv.DictReader(f))

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'

text='''# ワーカー管理・分配・融合の比較実験（2026-10-20）

QKV融合とFFN融合は両管理方式で有効だった。管理方式の差は最適化条件や人数切替に依存する。
今回のpthread実装は同じ融合とdynamic分配を備えており、以前の未最適化pthread版との
単純な方式比較から先へ進めた。ただし、ここでの勝敗を各ライブラリ一般の優劣とはしない。

## 実験の設計と範囲

- 速度測定700回＋詳細計測18回。別途バックエンド別のwarmupを6回実施。
- 一次探索: 2管理方式 × 2分配 × QKV有無 × FFN有無 × output1/4行 × CPU2構成 = 64セル。
  従来方式13構成、切替9構成も加え、86構成×5ブロック＝430回。
- 独立の追加確認13構成×10回＝130回。通常部分/outputの分配を分離した16構成×5回＝80回。
  人数切替だけの6構成×10回＝60回。詳細は代表6構成×3回＝18回。
- P4E4=(0,2,1,3,4,5,8,9)の8人、P2E8=(0,2,4,5,6,7,8,9,10,11)の10人。
  全14では上記P4E4に6,7,10,11,12,13を追加する。14は論理CPU数。
- stories15M、steps=256、temperature=0、seed=42、同一prompt。実行順は各完全ブロック内でシャッフル。
  他のベンチマークやテストとは並走させていない。全700回の生成文ハッシュが一致した。
- 新しいbackend=openmp/pthreadは同じ数値カーネルを共有。pthreadは既存の常駐プールとバリアを使用。
  staticは余りを前のrankから配る方法に揃え、Attentionも同じ静的分配。dynamicは行列演算に適用。
  層内16行、output32物理行を1タスクとし、4行カーネルでも32物理行になるよう粒度を揃えた。
- FFN要因はup/gate融合とSwiGLU並列化をまとめた処置。両者の内訳は今回独立には測っていない。
- 従来parallelはoutput4行時のchunk=32が128物理行。追加確認で32/128物理行を揃え直した。
  legacyのfactor列は設定値で、一部は実装で使われない。要因解析はfamily=factorialだけを対象とする。

## 各最適化の平均効果

単位は短縮ms/token。正なら改善。他の要因を均等に平均した周辺効果で、単一基準からの孤立効果ではない。
一次探索のeffects.csvが根拠。括弧は完全ブロックを再標本化した探索的95%bootstrap区間。

'''
effects=read(ROOT/'effects.csv')
rows=[]
for factor in ('qkv','ffn','dynamic','row4'):
 vals=[]
 for layout,scope in [('P4E4','openmp'),('P4E4','pthread'),('P2E8','openmp'),('P2E8','pthread')]:
  r=next(r for r in effects if (r['layout'],r['scope'],r['factor'])==(layout,scope,factor))
  vals.append(f"{float(r['saved_ms_per_token']):+.3f} [{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}]")
 rows.append([factor,*vals])
text+=table(['要因','OMP 8','pthread 8','OMP 10','pthread 10'],rows)
text+='''
QKVとFFNは4条件とも平均で改善し、区間もゼロをまたがない。
全行列のdynamic化は10人では有効だが、8人では平均効果が小さいか不確か。
4行化は管理方式・CPU構成・粒度に依存し、一律に有効とは言えない。
方式そのものの平均差はOpenMP有利が8人で0.010ms、10人で0.006ms程度だが、両方の区間はゼロをまたぐ。
一方、十分に最適化した特定条件では方式差が出るため、平均値だけでも結論しない。

## 重複・相乗効果

相乗短縮 = T(片方A)+T(片方B)-T(両方)-T(両方なし)。正なら単独短縮の合計より速く、負なら重複/阻害。
同一replicate内で、他の要因を揃えて計算する。要因2つの全組合せはinteractions.csvに保存。

QKV×FFNの周辺相互作用（他の分配/行処理設定を平均）:

'''
inter=read(ROOT/'interactions.csv')
rows=[]
for layout in ('P4E4','P2E8'):
 for scope in ('openmp','pthread'):
  r=next(r for r in inter if (r['layout'],r['scope'],r['factor_a'],r['factor_b'])==(layout,scope,'qkv','ffn'))
  rows.append([layout,scope,f"{float(r['extra_saved_ms_per_token']):+.3f}",f"[{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}]"])
text+=table(['CPU','方式','相乗短縮ms/token','探索的95%区間'],rows)
text+='''
全てゼロをまたぎ、QKVとFFNの強い相乗/阻害が全条件で一貫するとは言えない。
ただし条件を固定すると違いがある。static・1行のpthread10では相乗短縮が-0.118ms
（探索区間[-0.234,-0.001]）で、単独改善が併用時に重なる兆候が出た。
他の分配/4行条件へ一般化しないこと。conditional_interactions.csvに固定条件の結果も保存した。

通常部分のdynamicとoutputのdynamicを分けた追加実験では、P2E8 OpenMPで
平均短縮が層側0.019ms、output側0.076msだった。output側の寄与が大きい。
この条件の相乗短縮は-0.067ms（探索区間[-0.119,-0.014]）で、両方を有効にした効果は
単純な足し算より小さかった。pthread10では対応する相互作用は+0.005msで区間がゼロをまたぐ。
時間の非加算性は確認できても、物理的な原因を同期・キャッシュ・周波数のどれかに断定しない。
多くの相互作用を探索しており、多重比較補正はしていない。強い因果主張の根拠にはしない。

## 同じ融合・分配での人数切替

下表は独立した10回比較。同じQKV/FFN融合、dynamic(層16/output32物理行)、output1行を使用。
管理方式と人数以外の数値カーネル・分配を揃えた。単位tokens/s。

'''
switch=read(RES/'factorial_switch_20261020_165408_200692'/'summary.csv')
rows=[]
for variant in ('fixed8','switch8_14','fixed14'):
 rows.append([variant,*[f"{float(next(r for r in switch if r['name']==f'{b}_{variant}')['mean_tokens_per_second']):.2f}" for b in ('openmp','pthread')]])
text+=table(['通常→output','OpenMP','pthread'],rows)
text+='''
OpenMPの8→14切替は8固定から0.118ms悪化（区間[0.096,0.141]）。
pthreadは0.028ms改善傾向だが、区間[-0.0005,0.0525]で小差の確証はまだ弱い。
両方式の「切替効果の差」は0.146ms（区間[0.117,0.171]）で、今回の実装では明確に異なる。
これは常駐プール・OpenMP実装・起床/完了待ち・チーム人数変更を含む実装全体の差。
OSスレッドの生成終了だけの差を測ったものではない。OpenMPもワーカーを再利用し得る。
全14固定は両方式で遅いが、outputだけ14人を使う構成まで否定する結果ではない。

代表3回の詳細計測では、8→14切替のoutput区間はpthread約0.494ms、OpenMP約0.599ms。
差がoutput区間に現れることは分かるが、この全差を純粋な切替費用とみなさない。
固定10人の最適化版ではoutputは両方約0.48msだった。

## 追加確認と従来方式

次は別フェーズの10回比較。最上位の選択に使う補助結果で、上の切替表と絶対値を混ぜない。

'''
confirmation=read(RES/'factorial_confirm_20261020_165013_322636'/'summary.csv')
select=('legacy_upstream_P4E4','legacy_parallel_P2E8','P2E8_openmp_row4_chunk32','P2E8_pthread_row4_chunk32',
        'P2E8_openmp_row4_chunk128','P2E8_pthread_row4_chunk128','switch_pthread_8_14_d1')
text+=table(['構成','平均tok/s'],[[name,f"{float(next(r for r in confirmation if r['name']==name)['mean_tokens_per_second']):.2f}"] for name in select])
text+='''
固定10人・4行・32物理行ではOpenMPがpthreadより0.050ms速かった（区間[0.032,0.071]）。
8人・128物理行では方式差の区間がゼロをまたぐ。管理方式とタスク粒度にも相互作用がある。

従来の均等常駐、重み2の常駐、4→8切替、未改造本家、前回の融合OpenMPも一次探索で再測定した。
既存の静的4→8常駐では重み1が681.70 tok/s、重み2が650.90 tok/s。
融合済みの静的pthread4→8でも重み2は有利にならなかった。
これは指定した重み2についての結果で、適切な重み付き分配一般の否定ではない。

## 信頼性と残る問題

- 一次探索の全構成平均は、block1の1.330からblock5の1.561ms/tokenへ約17%悪化している。
  絶対速度は安定しておらず、原因は未計測。実行順を各ブロックでシャッフルし、完全ブロック内の
  対比較で効果と相互作用を推定した。温度/周波数/電力のどれかに原因を断定しない。
- 5/10ブロックのbootstrap区間は探索的であり、少ない反復と時系列変動の限界がある。
  小差や条件限定の相互作用は、状態を記録した追加試験が必要。
- 全700回の速度測定と18個の詳細JSONLの完全性を検証。生成文は一致。
  GQA、不均等な行数、各融合スイッチ、1/4行処理、dynamic/static、14人、人数切替をテスト。
  O2で全logit完全一致、Ofastでは絶対誤差1e-7以下。テスト10件成功。
- 詳細計測には時計取得と記録、pthreadのAttention完了確認用バリアの費用が加わる。
  速度比較は計測なしを優先。layer_totalと各演算は重複するため足し合わせない。
- 現在の常駐プールは条件変数によるdispatchとpthread_barrierを使う。
  spinningやバリア方式そのものは今回変えていない。理想的なpthread全般との比較ではない。
- 大型モデル、長文、複数リクエストは未検証。FFN融合とSwiGLU単独並列化の分離も次の実験候補。

## 再実行と成果物

`src/cpuprofiler/benchmark_factorial.py`をIDEなどから実行すると、PythonのBenchmarkConfigで
全要因の430回測定とeffects/interactionsの集計を行う。shellの環境変数設定や手動ビルドは不要。
追加確認・分配分離・人数切替はそれぞれの結果ディレクトリのreproduce.pyに条件を保存した。
追加スクリプトは一次探索と同じbuild/factorial-*バイナリを再利用する。実験を独立して再現するには
一次探索と同じソースをビルドし、builds/*/binary.sha256で一致を確認してから実行すること。

- plan.json、source/、sources.json、builds/: 計画・ソースとハッシュ・コンパイル条件。
- trials.csv / cells.csv: 生の速度・ブロックと実行順 / セル別集計。
- effects.csv / interactions.csv / conditional_interactions.csv: 周辺効果と相互作用。
- followup_contrasts.csv: 追加実験の対比較区間。block_means.csv: 時間経過の変動。
- 各runのconfig.json、stdout.txt、stderr.log、summary.json。
- ../factorial_confirm_20261020_165013_322636/: 追加確認130回。
- ../factorial_schedule_20261020_165250_800318/: 通常部分/outputの分配分離80回。
- ../factorial_switch_20261020_165408_200692/: 人数切替60回。
- ../factorial_detail_20261020_165200_548558/: 詳細18回、層別CSVと圧縮JSONL。
- summarize_followups.py / write_report.py: 対比較とこの報告を生成する処理。
'''
(ROOT/'README.md').write_text(text)
