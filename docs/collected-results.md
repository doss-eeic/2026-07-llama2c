# ブランチ横断で集めた結果データの索引

整理日: 2026-10-26。ローカルと `origin` の全ブランチ（`backup/*` を除く）から、
master に対して追加・変更された結果データを集めた。対象は `results/`・`benchmark*/`・`bench_memopt/`・
`memlab/`・`tmp/stay_probe/` の 52387 ファイル。

- 同じパスで中身が違うファイルは無かったため、マージや取捨選択はしていない。
- ベンチで使ったビルド済みバイナリ（`run-fused` など16個）と解析コードの `analyze/` は含めない。
  バイナリは各取得元ブランチにある。
- 結果を生成したスクリプト・ソースは取得元ブランチにある。docs の索引は [collected-docs.md](collected-docs.md)。
- 「取得元」は、そのディレクトリのファイルを含むブランチ（複数あれば全部）。

|ディレクトリ|ファイル数|取得元ブランチ|
|---|---:|---|
|`bench_memopt/`|9|`dev/memory-opt`, `dev/wataru-2`|
|`bench_memopt/env`|6|`dev/memory-opt`|
|`benchmark/`|43|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru`, `dev/wataru-2`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`, `feature/cpu-profiler-strict-argmax`, `merge/opt-to-master`|
|`benchmark_affinity_stories15M/`|8|`dev/wataru`|
|`benchmark_affinity_stories15M/20261012_175533`|116|`dev/wataru`|
|`benchmark_affinity_stories15M/20261012_183251_l2`|85|`dev/wataru`|
|`benchmark_affinity_stories15M/20261012_193505_l2_scaling`|361|`dev/wataru`|
|`benchmark_weights_layout/20261013_140739`|365|`dev/sakurada`|
|`benchmark_weights_layout/20261013_141838`|365|`dev/sakurada`|
|`memlab/`|10|`dev/sakurada`|
|`memlab/native_EE00025_20261013`|178|`dev/sakurada`|
|`memlab/native_core_EE00025_20261019`|5|`dev/sakurada`|
|`memlab/native_pcore_EE00025_20261019`|10|`dev/sakurada`|
|`memlab/native_power_EE00025_20261015`|5|`dev/sakurada`|
|`memlab/native_team2_EE00025_20261019`|10|`dev/sakurada`|
|`memlab/native_w0345_EE00025_20261019`|10|`dev/sakurada`|
|`results/`|1|`dev/wataru-2-topology-20261022`|
|`results/20261019_171347`|16|`dev/wataru-2`|
|`results/20261019_172823`|16|`dev/wataru-2`|
|`results/20261019_174659`|34|`dev/wataru-2`|
|`results/20261019_175702`|34|`dev/wataru-2`|
|`results/core_sweep_20261020`|114|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/cores_20261020_160902_626449`|187|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/cores_20261020_161043_147082`|190|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/cores_confirm_20261020_161145_042899`|333|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/cores_detail_20261020_161237_705922`|190|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/current_vs_upstream_20261020_184403_185078`|1532|`dev/wataru-qkv-ffn`|
|`results/current_vs_upstream_20261020_184747_142513`|1536|`dev/wataru-qkv-ffn`|
|`results/distribute_20261020_143658_005491`|27|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_144559_945859`|27|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_144745_053187`|27|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_144854_858646`|53|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_145457_135957`|79|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_145737_929213`|79|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_145922_351778`|242|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/distribute_20261020_150009_296759`|242|`bench/cpu-profiler-int8-exact-split-beam`, `codex/stay-minimal`, `dev/cpu-profiler`, `dev/cullisto`, `dev/opt`, `dev/wataru-2-topology-20261022`, `dev/wataru-qkv-ffn`, `dev/wataru-stay-minimal`, `dev/wataru2`, `feature/cpu-profiler-fixed-team`, `feature/cpu-profiler-int8-exact`, `feature/cpu-profiler-split-beam`, `feature/cpu-profiler-stay-team`|
|`results/factorial_20261020_164542_137621`|1779|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/factorial_confirm_20261020_165013_322636`|524|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/factorial_detail_20261020_165200_548558`|138|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/factorial_schedule_20261020_165250_800318`|324|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/factorial_switch_20261020_165408_200692`|244|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/master_threads_20261026`|4|`dev/wataru2`|
|`results/omp_tmp_coarse_20261026`|7|`dev/omp-tmp`|
|`results/omp_tmp_final_20261026`|10|`dev/omp-tmp`|
|`results/omp_tmp_tune_20261026`|7|`dev/omp-tmp`|
|`results/optimized_output_coarse_20261026`|7|`dev/omp-tmp`|
|`results/optimized_output_confirm_20261026`|10|`dev/omp-tmp`|
|`results/optimized_output_final_20261026`|9|`dev/omp-tmp`|
|`results/optimized_output_tune_20261026`|7|`dev/omp-tmp`|
|`results/output_tune_cas`|511|`dev/wataru2`|
|`results/output_tune_coarse`|1161|`dev/wataru2`|
|`results/output_tune_confirm`|727|`dev/wataru2`|
|`results/output_tune_final`|888|`dev/wataru2`|
|`results/output_tune_groups`|343|`dev/wataru2`|
|`results/output_tune_neighbors`|823|`dev/wataru2`|
|`results/output_tune_rerun_20261026`|890|`dev/wataru2`|
|`results/output_tune_schedule`|679|`dev/wataru2`|
|`results/output_tune_validation`|74|`dev/wataru2`|
|`results/probe_stay2_20261020_133611_450528`|518|`dev/wataru-2`|
|`results/probe_stay2_20261020_133723_755239`|284|`dev/wataru-2`|
|`results/probe_stay2_20261020_133807_534147`|284|`dev/wataru-2`|
|`results/probe_stay2_20261020_133855_042738`|202|`dev/wataru-2`|
|`results/stay2_output_20261020_132922_555802`|16|`dev/wataru-2`|
|`results/stay2_output_20261020_133120_502255`|16|`dev/wataru-2`|
|`results/stay2_output_20261020_133126_301350`|16|`dev/wataru-2`|
|`results/stay_20261019_180412_332608`|56|`dev/wataru-2`|
|`results/stay_20261019_180830_217316`|31|`dev/wataru-2`|
|`results/stay_20261019_180853_414823`|26|`dev/wataru-2`|
|`results/stay_20261019_182546_176581`|31|`dev/wataru-2`|
|`results/stay_20261019_183542_974814`|21|`dev/wataru-2`|
|`results/stay_20261019_183936_295707`|31|`dev/wataru-2`|
|`results/stay_20261019_183946_916954`|31|`dev/wataru-2`|
|`results/stay_20261019_183954_284260`|21|`dev/wataru-2`|
|`results/stay_20261020_153956_052618`|205|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/stay_20261020_154433_913377`|245|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/stay_detail_20261020_155033_512133`|428|`dev/wataru-2-topology-20261022`, `dev/wataru-stay-minimal`|
|`results/topology14_confirm_20261022`|3288|`dev/wataru-2-topology-20261022`|
|`results/topology14_oracle_20261022`|10992|`dev/wataru-2-topology-20261022`|
|`results/topology14_search_20261022`|7212|`dev/wataru-2-topology-20261022`|
|`results/topology_adoption_20261022`|10252|`dev/wataru-2-topology-20261022`|
|`results/topology_confirm_20261022`|879|`dev/wataru-2-topology-20261022`|
|`results/topology_detail_20261022`|306|`dev/wataru-2-topology-20261022`|
|`results/topology_online_normalized_20261022`|639|`dev/wataru-2-topology-20261022`|
|`results/topology_online_raw_20261022`|639|`dev/wataru-2-topology-20261022`|
|`tmp/stay_probe`|7|`dev/wataru-2`|
