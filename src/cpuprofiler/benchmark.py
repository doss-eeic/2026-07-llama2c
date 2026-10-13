import re
import subprocess
import cpuprofiler.topology as Topology

def test_benchmark():
    # シンプルなテストを回すための関数。cpu4を使ってベンチマークを実行し、tpsを取得する。
    result = subprocess.run(
        [
            "taskset", "-c", "4",
            "./build/omp-run", "./build/stories15M.bin",
            "-n", "256",
            "-t", "0",
            "-s", "42",
            "-i", "Once upon a time",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"achieved tok/s:\s*([\d.]+)",
        result.stderr
    )

    if match is None:
        raise RuntimeError("Benchmark result not found")

    tps = float(match.group(1))
    print(f"{tps:.2f} tokens/s")

class BenchmarkRunner:
    def __init__(self):
        self.topology = Topology.CPUTopology()

if __name__ == "__main__":
    br = BenchmarkRunner()
    br.topology.print_summary()