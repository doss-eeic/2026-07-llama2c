from .benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .topology import CPU, CPUTopology
from .autotuner import AutoTuner
import os


def main(*args):
    if len(args) == 0:
        CPUTopology().print_summary()
    else:
        print("Running benchmark...")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner())
        best_result = tuner.search()
        print(f"Best configuration: {best_result.config}")
        print(f"Best performance: {best_result.tokens_per_second} tokens/sec")
        