# ブランチ横断で集めた docs の索引

整理日: 2026-10-26。ローカルと `origin` の全ブランチ（`backup/*` を除く）から、
master に対して追加・変更された Markdown を集めた。
`results/`・`tmp/` 内の結果用 README とトップレベルの `README.md` は対象外。

- 各パスでは、他の版の子孫にあたる版（履歴上で古い版）を除き、残った版を元のパスに置いた。
- 残った版が複数あるパスは、古い順に `git merge-file`（3-way）でマージし、衝突は下表の「備考」のとおり手で解決した。
- 文書は書かれた当時のブランチのコード・結果を前提にしている。`results/` などへのリンクは、
  このブランチに無いものがある。必要なら「取得元」のブランチを参照する。
- `dev/omp-tmp` b6ee9e86 と `merge/opt-to-master` a62bde1d は 2026-10-26 時点で未 push のローカルコミット。
- 「版数」は、そのパスで内容の違う版がいくつあったか。

|ファイル|取得元（ブランチ・コミット）|版数|備考|
|---|---|---:|---|
|[analyze/README.md](../analyze/README.md)|`dev/wataru` ed3a81dd|1||
|[bench_memopt/README.md](../bench_memopt/README.md)|`dev/memory-opt` 0dedad45|1||
|[doc/stories260K.md](../doc/stories260K.md)|`merge/opt-to-master` a62bde1d|1||
|[docs/README.md](../docs/README.md)|`dev/omp-tmp` b6ee9e86|1||
|[docs/master-threads-and-output-split.md](../docs/master-threads-and-output-split.md)|`dev/wataru2` 1e1fa037|1||
|[docs/optimized-output-benchmark.md](../docs/optimized-output-benchmark.md)|`dev/omp-tmp` b6ee9e86|1||
|[docs/output-tuning-rerun-20261026.md](../docs/output-tuning-rerun-20261026.md)|`dev/wataru2` 1e1fa037|1||
|[docs/output-tuning.md](../docs/output-tuning.md)|`dev/wataru2` 1e1fa037|1||
|[docs/weighted-output-benchmark.md](../docs/weighted-output-benchmark.md)|`dev/omp-tmp` b6ee9e86|2||
|[project/docs/AI向け_20261020_CPU並列化.md](../project/docs/AI向け_20261020_CPU並列化.md)|`dev/wataru-2-topology-20261022` b38b845a|1||
|[project/docs/AI向け_20261020_最適化要因比較.md](../project/docs/AI向け_20261020_最適化要因比較.md)|`dev/wataru-2-topology-20261022` 8518ee9e|2||
|[project/docs/AI向け_20261022_14CPU探索.md](../project/docs/AI向け_20261022_14CPU探索.md)|`dev/wataru-2-topology-20261022` 8518ee9e|1||
|[project/docs/AI向け_20261022_構成探索.md](../project/docs/AI向け_20261022_構成探索.md)|`dev/wataru-2-topology-20261022` 8518ee9e|1||
|[project/docs/AI用のメモ.md](../project/docs/AI用のメモ.md)|`dev/wataru-qkv-ffn` 1597e2b1, `dev/wataru-2-topology-20261022` 8518ee9e, `merge/opt-to-master` a62bde1d|5|マージ。冒頭の衝突は 2026-10-22 の引き継ぎを先、2026-10-20 の項を後に並べた|
|[project/docs/AI用のメモ2.md](../project/docs/AI用のメモ2.md)|`dev/wataru-2` 0900c3d2|1||
|[project/docs/autotune-detail.md](../project/docs/autotune-detail.md)|`dev/wataru-2` 70418b94|1||
|[project/docs/benchmark-data-summary-20261022.md](../project/docs/benchmark-data-summary-20261022.md)|`dev/wataru-qkv-ffn` 4fcaf5b3|1||
|[project/docs/benchmark.md](../project/docs/benchmark.md)|`dev/wataru-2` 0900c3d2|1||
|[project/docs/core_thread_hugepage_results.md](../project/docs/core_thread_hugepage_results.md)|`dev/sakurada` 7c125ed3|1||
|[project/docs/fused_matmul_w5.md](../project/docs/fused_matmul_w5.md)|`dev/sakurada` b631bd63|1||
|[project/docs/hugepage.md](../project/docs/hugepage.md)|`merge/opt-to-master` a62bde1d|2||
|[project/docs/kasikoi-search.md](../project/docs/kasikoi-search.md)|`dev/opt` ad5c4696|1||
|[project/docs/optimization-strategy-20261022.md](../project/docs/optimization-strategy-20261022.md)|`dev/wataru-qkv-ffn` 4fcaf5b3|1||
|[project/docs/output-tuning.md](../project/docs/output-tuning.md)|`dev/wataru2` f0e4ffbb|1||
|[project/docs/parallel-experiment-design.md](../project/docs/parallel-experiment-design.md)|`dev/wataru-qkv-ffn` 4fcaf5b3|1||
|[project/docs/persistent-openmp.md](../project/docs/persistent-openmp.md)|`dev/wataru2` f0e4ffbb|1||
|[project/docs/qkv-ffn-benchmark.md](../project/docs/qkv-ffn-benchmark.md)|`dev/wataru-qkv-ffn` 1597e2b1|1||
|[project/docs/qkv-ffn-fusion.md](../project/docs/qkv-ffn-fusion.md)|`dev/wataru-qkv-ffn` 1597e2b1, `merge/opt-to-master` a62bde1d|3|マージ。衝突は dev/wataru-qkv-ffn 側の本文を残し、merge/opt-to-master の hugepage への参照を追記|
|[project/docs/stage-switching-experiment.md](../project/docs/stage-switching-experiment.md)|`dev/wataru-2` 70418b94|1||
|[project/docs/stay.md](../project/docs/stay.md)|`dev/wataru-2-topology-20261022` 27553fc7|2||
|[project/docs/two-stage-argmax.md](../project/docs/two-stage-argmax.md)|`dev/opt` f8590c99|1||
|[project/docs/weights_memory_layout_report.md](../project/docs/weights_memory_layout_report.md)|`dev/sakurada` cc724ceb|1||
|[project/docs/ベンチマーク結果_fp32_int8-exact_20261022.md](../project/docs/ベンチマーク結果_fp32_int8-exact_20261022.md)|`bench/cpu-profiler-int8-exact-split-beam` d1cba56b|1||
|[project/docs/今後の実験方針.md](../project/docs/今後の実験方針.md)|`dev/wataru-qkv-ffn` 4fcaf5b3|1||
|[project/qqk,ffn最適化.md](../project/qqk,ffn最適化.md)|`merge/opt-to-master` a62bde1d|2||
|[project/src/src-docs.md](../project/src/src-docs.md)|`dev/sakurada` f251071f|1||
|[project/プロジェクトReadMe.md](../project/プロジェクトReadMe.md)|`dev/wataru-2-topology-20261022` 8518ee9e, `feature/cpu-profiler-strict-argmax` 3159ebd5, `bench/cpu-profiler-int8-exact-split-beam` da5461e5, `merge/opt-to-master` a62bde1d|5|マージ。セットアップ節の衝突は bench/cpu-profiler-int8-exact-split-beam 側（環境変数表・`-q int8-exact` を含む）を採用|
|[project/プロジェクトReatMe.md](../project/プロジェクトReatMe.md)|`dev/sakurada` f251071f|1||
|[src/cpuprofiler/CORE_TUNING.md](../src/cpuprofiler/CORE_TUNING.md)|`dev/wataru-2-topology-20261022` 8518ee9e|2||
|[src/cpuprofiler/DATASET.md](../src/cpuprofiler/DATASET.md)|`dev/wataru-2` 624606f3|1||
|[src/cpuprofiler/DETAIL.md](../src/cpuprofiler/DETAIL.md)|`dev/wataru-2-topology-20261022` 1fa118a6|1||
|[src/cpuprofiler/TOPOLOGY14.md](../src/cpuprofiler/TOPOLOGY14.md)|`dev/wataru-2-topology-20261022` 8518ee9e|1||
|[src/cpuprofiler/TOPOLOGY_SEARCH.md](../src/cpuprofiler/TOPOLOGY_SEARCH.md)|`dev/wataru-2-topology-20261022` 8518ee9e|1||
|[プロジェクト/docs/AI用のメモ.md](../プロジェクト/docs/AI用のメモ.md)|`dev/memory-opt` b5da09ce|1||
|[プロジェクト/output-layer-memory-optimization.md](../プロジェクト/output-layer-memory-optimization.md)|`dev/memory-opt` cbcfbb83|2||
|[プロジェクト/progress-2026-10-15.md](../プロジェクト/progress-2026-10-15.md)|`dev/memory-opt` 0dedad45|1||
|[プロジェクト/progress-2026-10-19-p4x-int8x.md](../プロジェクト/progress-2026-10-19-p4x-int8x.md)|`dev/memory-opt` cbcfbb83|1||
|[プロジェクト/progress-2026-10-19-schedule.md](../プロジェクト/progress-2026-10-19-schedule.md)|`dev/memory-opt` 7bb6e8b4|1||
|[プロジェクト/progress-2026-10-19-switch-c.md](../プロジェクト/progress-2026-10-19-switch-c.md)|`dev/memory-opt` 0dedad45|1||
|[プロジェクト/progress-2026-10-19.md](../プロジェクト/progress-2026-10-19.md)|`dev/memory-opt` 65ec3d13|1||
|[プロジェクト/プロジェクトReatMe.md](../プロジェクト/プロジェクトReatMe.md)|`dev/memory-opt` b5da09ce|1||
