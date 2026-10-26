# 層・演算別の再測定

`python3 src/cpuprofiler/benchmark_detail.py`を実行する。CPU構成・反復数などは
`benchmark_distribute.py`の`main()`で指定する。通常測定の入口は従来どおり。

- OpenMP側: `dev/wataru-2`の`66870ea:run_detail.c`をそのまま復元。
- stay側: 現在の`run.c`と`src/stay.h`を使用し、同じJSONL形式の計測点を追加。
- `LLAMA_PROFILE`付きビルドだけ計測する。通常ビルドでは追加計測・同期は無効。
- 各構成10回、seed 42で実行順をシャッフル。最初のforwardを除いて時間を集計。
- `layers.csv`は各層・演算別、`stages.csv`は全層合計の1トークン当たり時間。
- `summary.csv`は反復ごと、`aggregate.csv`と`layers_aggregate.csv`は反復平均。
- 生ログは`profile.jsonl.gz`。元の生成文、stderr、設定、ビルドログも保存する。

stayのoutputはcoordinator側でdispatchから完了まで計測する。層内の演算は
rank 0が計測し、Attention完了時には計測用の同期を追加する。プロファイル出力の
I/Oはforwardの外で行うが、時計取得・バッファ記録・同期の負荷は含まれる。
したがって計測なしのスループットとは区別して扱う。

`layer_total`と各演算は重複するため足し合わせない。
`outside_layers_output_ms_per_token`はforwardから全層とoutputを引いた残差であり、
embedding・final norm・dispatchなどを含む。純粋な切り替えコストではない。
