from .benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .topology import CPU, CPUTopology
from .autotuner import AutoTuner
import os
import sys

def parse_workers():
    # コマンドラインの "-x <omp|pool>" を読む（run.c の -x と同じ）。省略すると omp
    if "-x" not in sys.argv:
        return "omp"
    i = sys.argv.index("-x")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1] not in ("omp", "pool"):
        raise SystemExit("-x must be omp or pool")
    return sys.argv[i + 1]

def main(*args):
    workers = parse_workers()
    if len(sys.argv) == 1 or sys.argv[1] == "topology":
        CPUTopology().print_summary()

    elif sys.argv[1] == "beam":
        print("Running beam search...")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner(), workers)
        results = tuner.beam_search()
        for config, result in results:
            print(f"Configuration: {config}, Performance: {result.tokens_per_second} tokens/sec")

    elif sys.argv[1] == "sample":
        print("Running benchmark...")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner(), workers)
        best_result = tuner.search()
        print(f"Best configuration: {best_result.config}")
        print(f"Best performance: {best_result.tokens_per_second} tokens/sec")

    elif sys.argv[1] == "kasikoi":
        print("Running kasikoi search...")
        topology = CPUTopology()
        split = "--split" in sys.argv  # 出力層とそれ以外の CPU（run.c の -b / -c）も探す
        tuner = AutoTuner(topology, BenchmarkRunner())
        config = BenchmarkConfig(
            cpus=tuple(topology.cpus), threads=topology.n_logical_cpus,
            optimize_weights=True, two_stage_argmax=True, workers=workers,
        )
        best_result = tuner.kasikoi_serch(config, split=split)
        print(f"Best configuration: {best_result.config}")
        print(f"Best performance: {best_result.tokens_per_second:.2f} tokens/sec")
        print(f"Search: {tuner.last_search['seconds']:.2f} sec, "
              f"{tuner.last_search['measurements']} measurements")
    
    else:
        print("Invalid argument.")