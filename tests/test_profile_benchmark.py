import gzip
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpuprofiler.benchmark_detail import parse_profile
from cpuprofiler.benchmark_distribute import run
from cpuprofiler.benchmarkconfig import BenchmarkConfig


class ProfileBenchmarkTest(unittest.TestCase):
    def test_rejects_incomplete_profile(self):
        with self.assertRaises(ValueError):
            parse_profile('{"event":"meta"}\n')

    def test_stay_and_reference_profiles_preserve_generation(self):
        cpus = tuple(sorted(os.sched_getaffinity(0))[:4])
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("stay", "upstream"):
                output = Path(directory) / backend
                summary = run(BenchmarkConfig(
                    cpus=cpus, threads=len(cpus), output=output,
                    binary=Path(directory) / (backend + "-run"),
                    backend=backend, profile=True, steps=8,
                ))
                self.assertGreater(summary["forward_ms_per_token"], 0)
                self.assertEqual(summary["measured_tokens"], summary["forward_calls"] - 1)
                with gzip.open(output / "profile.jsonl.gz", "rt") as file:
                    profile = parse_profile(file.read())
                self.assertEqual(len(profile["tokens"]), summary["forward_calls"])
                self.assertTrue((output / "layers.csv").is_file())
                self.assertTrue((output / "stages.csv").is_file())
                outputs.append((output / "stdout.txt").read_bytes())
            self.assertEqual(*outputs)


if __name__ == "__main__":
    unittest.main()
