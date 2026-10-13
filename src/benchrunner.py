import re
import subprocess

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