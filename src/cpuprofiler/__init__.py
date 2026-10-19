from .benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .topology import CPU, CPUTopology
from .autotuner import AutoTuner
import os
import sys

def main(*args):
    if len(sys.argv) == 1 or sys.argv[1] == "topology":
        CPUTopology().print_summary()

    elif sys.argv[1] == "collect":
        from .dataset import main as collect_main
        collect_main(sys.argv[2:])

    elif sys.argv[1] == "mock":
        from .mock import main as mock_main
        mock_main(sys.argv[2:])

    elif sys.argv[1] == "beam":
        print("Running beam search...")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner())
        results = tuner.beam_search()
        for config, result in results:
            print(f"Configuration: {config}, Performance: {result.tokens_per_second} tokens/sec")

    elif sys.argv[1] == "sample":
        print("Running benchmark...")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner())
        best_result = tuner.search()
        print(f"Best configuration: {best_result.config}")
        print(f"Best performance: {best_result.tokens_per_second} tokens/sec")
    
    else:
        print("Invalid argument.")