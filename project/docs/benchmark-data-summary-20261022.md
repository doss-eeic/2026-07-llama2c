# pullしたベンチマークデータの要約（2026-10-22）

対象: `e02948e2` → `1597e2b1` のpullで追加された比較結果。主測定は2026-10-20。

## 結論

- 最速の測定条件は **P2E8_dynamic: 744.31 tok/s**。同条件の未改造版比 **+29.9%**（保存済みの対応ありbootstrap 95%区間 +25.4〜+35.2%）、融合前比 **+22.3%**。
- 次点は **P4E4_static: 709.49 tok/s**。未改造版比 **+22.8%**、融合前比 **+24.4%**。
- 現在版は融合前に対し全10条件で平均 **+5.1〜+32.9%**。QKVとFFNの個別寄与はこの実験から分離できない。
- 既定設定では未改造版より遅い。P4E4で **−20.8%**、All14で **−9.8%**。通常演算のscheduleとoutput設定を明示する必要がある。

## 全条件

各セルは10試行の算術平均tokens/s。改善率は平均値の比。

| 条件 | 未改造 | 融合前 | 現在 | 未改造比 | 融合前比 |
|---|---:|---:|---:|---:|---:|
| P2E8_dynamic | 573.03 | 608.49 | 744.31 | +29.9% | +22.3% |
| P4E4_static | 577.97 | 570.53 | 709.49 | +22.8% | +24.4% |
| P4E4_dynamic | 659.97 | 618.77 | 684.95 | +3.8% | +10.7% |
| P2E8_static | 585.09 | 620.77 | 652.56 | +11.5% | +5.1% |
| All14_dynamic | 435.98 | 475.11 | 631.64 | +44.9% | +32.9% |
| P4_to_output8 | 557.69 | 562.55 | 615.90 | +10.4% | +9.5% |
| P4_static | 555.38 | 539.01 | 578.68 | +4.2% | +7.4% |
| All14_static | 460.83 | 440.49 | 540.16 | +17.2% | +22.6% |
| P4E4_defaults | 600.66 | 440.70 | 475.43 | -20.8% | +7.9% |
| All14_defaults | 418.89 | 316.89 | 377.74 | -9.8% | +19.2% |

## 条件と解釈

Intel Core Ultra 5 225U、stories15M、256 steps、temperature=0、seed=42。同じgccの `-Ofast -march=native -fopenmp` でビルドし、10条件×3版×10反復を測定。P4は2物理PコアのSMT込み4論理CPU、P4E4はP4+E4の8論理CPU、P2E8はSMTを使わない10物理コア、All14はLP Eを含む14論理CPU。

`static`は通常演算/outputともstatic。`dynamic`は通常演算dynamic,16、output dynamic,128。`defaults`はOMP_SCHEDULEとOUTPUT_*を省略し、このlibgompの通常演算のruntime既定値はdynamic,1。未改造版はruntime scheduleとOUTPUT_*を使わないため、未改造版との比較には融合以外の実装差も含まれる。`P4_to_output8`でも未改造版は4→4、現在版と融合前は4→8となる。

P4E4_dynamicの未改造比95%区間は−0.4〜+8.2%で0をまたぐ。平均が上回ることだけで優位を確定しない。P2E8_dynamicとP4E4_staticの順位も、別セッションでの再現確認が必要。

全測定はバッテリー駆動で、記録上は推論中のthermal throttleカウンタ増加なし。短時間・単一モデル・単一機体の結果であり、AC固定や長時間生成への一般化は未確認。ACが途中で1→0に変わった予備測定 `current_vs_upstream_20261020_184403_185078` は主比較へ混ぜない。

## 今回の照合

300個のrun別summary.jsonとtrials.csvのtokens/sが一致。30群すべての平均・中央値・平均ms/tokenを個票から再計算し、summary.csvと一致。20比較の改善率も再計算して一致。各群10反復、case/source/replicateの重複なし、生成文ハッシュは全300試行で1種類。保存済みbootstrap区間は参照し、再推定はしていない。再ベンチマークは実施していない。

測定対象currentは `e02948e2` の保存ソースであり、現在の未コミットの `src/fused.h` を測定した結果ではない。

## 元データ

- [実験条件・評価](qkv-ffn-benchmark.md)
- [主測定README](../../results/current_vs_upstream_20261020_184747_142513/README.md)
- [集計](../../results/current_vs_upstream_20261020_184747_142513/summary.csv)
- [改善率・95%区間](../../results/current_vs_upstream_20261020_184747_142513/comparisons.csv)
- [全300試行](../../results/current_vs_upstream_20261020_184747_142513/trials.csv)
