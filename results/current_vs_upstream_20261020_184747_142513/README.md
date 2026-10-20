# 現在版・融合前・未改造版の比較（主測定）

[評価と条件の説明](../../project/docs/qkv-ffn-benchmark.md)。300試行、全試行AC=0。

summary.csvは平均・中央値・平均ms/token、comparisons.csvは対応した反復のbootstrap比較、trials.csvは実行順つき個票。各条件/source/run_*に設定・生成文・stderr・実行前後の機体状態を保存。manifest.jsonにはコミット・ビルド条件・SHA-256・CPU配置・runtime probeを記録した。source/は実際にコンパイルしたソース。

verification.jsonに整合性検証、measurement_end.jsonに最後の試行の終了状態を保存。after.jsonは集計終了後であり、推論終了時ではない。sensitivity_*はthermalカウンタ増加のある3版の組を除く事後感度分析（除外0組）。

benchmark_compare.pyは実行時のスクリプトの記録。再実行にはリポジトリのsrc/cpuprofiler/benchmark_compare.pyを使う。
