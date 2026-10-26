from .benchmark import CLASSIFIERS, WORKERS, BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .topology import CPU, CPUTopology
from .autotuner import AutoTuner
import os
import sys
from datetime import datetime

# 結果の保存先（src/cpuprofiler/output）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def save_output(name, lines):
    # lines を output/<name>_<日時>.txt に書き出し、そのパスを返す
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now()
    path_to_save = os.path.join(OUTPUT_DIR, f"{name}_{now:%Y%m%d_%H%M%S}.txt")
    with open(path_to_save, "w", encoding="utf-8") as f:
        f.write(f"# {name} {now:%Y-%m-%d %H:%M:%S}\n")
        for line in lines:
            f.write(line + "\n")
    return path_to_save

def parse_option(flag, choices, default):
    # コマンドラインの "<flag> <値>" を読む。省略すると default
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv) or sys.argv[i + 1] not in choices:
        raise SystemExit(f"{flag} must be one of {choices}")
    return sys.argv[i + 1]

def main(*args):
    classifier = parse_option("-q", CLASSIFIERS, "fp32")  # 出力層の計算方法（run.c の -q）
    workers = parse_option("-w", WORKERS, "omp")          # 区間を動かすスレッド（run.c の SECTION_WORKERS）

    if len(sys.argv) == 1 or sys.argv[1] == "topology":
        CPUTopology().print_summary()

    elif sys.argv[1] == "beam":
        print(f"Running beam search... (output layer: {classifier}, workers: {workers})")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner(), classifier, workers)
        results = tuner.beam_search()
        lines = []
        for config, result in sorted(results, key=lambda x: x[1].tokens_per_second, reverse=True):
            lines.append(f"Configuration: {config}, Performance: {result.tokens_per_second} tokens/sec")
            print(f"Configuration: {config}, Performance: {result.tokens_per_second} tokens/sec")

        path_to_save = save_output(f"beam_{classifier}_{workers}", lines)
        print(f"Saved to {path_to_save}")

    elif sys.argv[1] == "split":
        print(f"Running beam search for body / output layer... (output layer: {classifier}, workers: {workers})")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner(), classifier, workers)
        config, result = tuner.split_beam_search()

        lines = [
            f"Output layer: {classifier}",
            f"Workers: {workers}",
            f"Best body CPUs: {config.body_cpus}, {result.body_ms_per_token} ms/token",
            f"Best output layer CPUs: {config.cls_cpus}, {result.cls_ms_per_token} ms/token",
            f"Best performance: {result.tokens_per_second} tokens/sec",
        ]
        for line in lines:
            print(line)

        path_to_save = save_output(f"split_{classifier}_{workers}", lines)
        print(f"Saved to {path_to_save}")

    elif sys.argv[1] == "sample":
        print(f"Running benchmark... (output layer: {classifier}, workers: {workers})")
        tuner = AutoTuner(CPUTopology(), BenchmarkRunner(), classifier, workers)
        best_result = tuner.search()
        print(f"Best configuration: {best_result.config}")
        print(f"Best performance: {best_result.tokens_per_second} tokens/sec")
    
    else:
        print("Invalid argument.")