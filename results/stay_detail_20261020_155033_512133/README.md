# 層・演算別の比較

6構成をランダム順で各10回測定。256ステップ、temperature=0、seed=42、
同じモデル・promptを使用。全60回の生成文は一致。時間は最初のforwardを除いた平均。
CPUはP4=(0,1,2,3)、P4E4=(0,1,2,3,4,5,8,9)。

| 構成 | tok/s | forward ms/token | QKV | Attention | FFN行列積 | output |
|---|---:|---:|---:|---:|---:|---:|
| Upstream_P4_HT | 606.55 | 1.615 | 0.265 | 0.061 | 0.425 | 0.770 |
| Upstream_P4E4 | 707.08 | 1.380 | 0.301 | 0.084 | 0.359 | 0.542 |
| P4_fixed | 617.54 | 1.585 | 0.221 | 0.062 | 0.420 | 0.785 |
| P4E4_fixed | 656.99 | 1.493 | 0.319 | 0.097 | 0.377 | 0.553 |
| Switch_equal | 677.85 | 1.439 | 0.254 | 0.069 | 0.434 | 0.544 |
| Switch_weight2 | 639.19 | 1.529 | 0.257 | 0.071 | 0.444 | 0.627 |

QKV以下の列もms/token。FFN行列積はup/gate/downの合計。
本家P4E4は本家P4よりFFNとoutputが速く、QKVとAttentionは遅い。
Switch_equalと本家P4E4のoutputはほぼ同じだが、層合計は0.850対0.836ms、
forwardから層合計とoutputを引いた残差は0.0445対0.0012ms。
この残差にはdispatch以外も含まれ、切り替えコスト単独の推定値ではない。

復元元と計測方法・オーバーヘッドの注意点は
`src/cpuprofiler/DETAIL.md`を参照。集計値は`aggregate.csv`、層別は
`layers_aggregate.csv`、生ログは各runの`profile.jsonl.gz`に保存している。
