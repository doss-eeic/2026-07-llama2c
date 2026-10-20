import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpuprofiler import benchmark_distribute as benchmark
from cpuprofiler.benchmarkconfig import BenchmarkConfig, UPSTREAM_REVISION


class StayBenchmarkTest(unittest.TestCase):
    def test_upstream_build_uses_exact_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = BenchmarkConfig((0, 1, 2, 3), 4, output, backend="upstream")
            command = benchmark.build_command(config, output, output / "upstream-run")
            original = subprocess.check_output(["git", "show", f"{UPSTREAM_REVISION}:run.c"], cwd=ROOT)
            self.assertEqual((output / "run.c").read_bytes(), original)
            self.assertIn("-fopenmp", command)
            self.assertNotIn("-DLLAMA_STAY", command)
            self.assertEqual(command[-4], str(output / "run.c"))
            self.assertEqual(json.loads((output / "source.json").read_text())["revision"], UPSTREAM_REVISION)

    def test_environment_does_not_inherit_other_experiments(self):
        config = BenchmarkConfig((0, 1, 2, 3), 4, Path("unused"), backend="upstream")
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "99", "OMP_WAIT_POLICY": "PASSIVE",
                                     "GOMP_SPINCOUNT": "0", "STAY_P_WEIGHT": "9"}):
            env = benchmark.execution_environment(config)
            self.assertEqual(env["OMP_NUM_THREADS"], "4")
            self.assertEqual(env["OMP_PLACES"], "{0},{1},{2},{3}")
            self.assertNotIn("OMP_WAIT_POLICY", env)
            self.assertNotIn("GOMP_SPINCOUNT", env)
            self.assertNotIn("STAY_P_WEIGHT", env)
            env = benchmark.execution_environment(BenchmarkConfig((0, 2, 4), 2, Path("unused")))
            self.assertEqual(env["STAY_LAYER_THREADS"], "2")
            self.assertEqual(env["STAY_CPUS"], "0,2,4")
            self.assertFalse(any(key.startswith(("OMP_", "GOMP_")) for key in env))

    def test_comparison_builds_each_backend_once_and_includes_upstream(self):
        calls = []

        def run(config):
            calls.append(config)
            config.output.mkdir(parents=True)
            return {"tokens_per_second": 500.0, "ms_per_token": 2.0}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "ROOT", Path(directory)), patch.object(benchmark, "run", side_effect=run):
                with contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
        self.assertEqual(len(calls), 50)
        self.assertEqual(sum(c.backend == "upstream" for c in calls), 10)
        self.assertEqual(sum(c.build for c in calls), 2)
        upstream = [c for c in calls if c.backend == "upstream"]
        self.assertTrue(all(c.threads == 4 and c.cpus == (0, 1, 2, 3) for c in upstream))
        self.assertEqual(len({c.binary for c in calls}), 2)


if __name__ == "__main__":
    unittest.main()
