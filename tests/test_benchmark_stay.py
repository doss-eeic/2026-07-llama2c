"""External policy execution and the standalone comparison workload."""

from contextlib import redirect_stdout
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpuprofiler.benchmark_stay import main, run
from cpuprofiler.benchmarkconfig import StayBenchmarkConfig
from test_benchmark_detail import example_log


class StayBenchmarkTests(unittest.TestCase):
    def test_external_policy_logs_and_timing_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "external.txt"
            content = "-2 default 0,1,2,3\n-1 output 0,1,2,3,4,5,8,9\n"
            policy.write_text(content)

            def child(command, **kwargs):
                self.assertEqual(command[0], str(config.binary.resolve()))
                self.assertEqual(Path(command[command.index("-c") + 1]).read_text(), content)
                self.assertEqual(command[command.index("-t") + 1], "0.25")
                self.assertEqual(kwargs["timeout"], 12)
                self.assertTrue(kwargs["check"])
                self.assertEqual("LLAMA_STAY_TIMING_ONLY" in kwargs["env"], config.timing_only)
                Path(kwargs["env"]["LLAMA_PROFILE_PATH"]).write_text(example_log())
                kwargs["stdout"].write(b"generated text\n")
                kwargs["stderr"].write(b"worker placement\n")

            for timing_only in (False, True):
                config = StayBenchmarkConfig(policy=policy, output=root / str(timing_only),
                                             temperature=0.25, timeout=12,
                                             stages=("q_proj",), timing_only=timing_only)
                with patch.dict(os.environ, {"LLAMA_STAY_TIMING_ONLY": "1"}), patch(
                        "cpuprofiler.benchmark_stay.subprocess.run", side_effect=child) as child_run:
                    summary, stages = run(config)
                child_run.assert_called_once()
                self.assertEqual(summary["measured_tokens"], 2)
                self.assertEqual(stages[0]["ms_per_token"], 2)
                self.assertEqual((config.output / "policy.txt").read_text(), content)
                self.assertEqual((config.output / "stdout.txt").read_text(), "generated text\n")
                self.assertEqual((config.output / "stderr.log").read_text(), "worker placement\n")
                with self.assertRaises(FileExistsError):
                    run(config)

    def test_failed_child_is_not_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.txt"
            policy.write_text("-2 default 0\n")
            with patch("cpuprofiler.benchmark_stay.subprocess.run",
                       side_effect=subprocess.CalledProcessError(1, ["stay-run-detail"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    run(StayBenchmarkConfig(policy=policy, output=root / "trial"))

    def test_main_compares_fixed_and_output_switch_policies(self):
        seen = {}

        def benchmark(config):
            seen[config.output.name] = config.policy.read_text()
            return {"forward_calls": 3, "measured_tokens": 2, "forward_ms_per_token": 4}, [
                {"stage": "output", "ms_per_token": 1}]

        with tempfile.TemporaryDirectory() as directory, patch(
                "cpuprofiler.benchmark_stay.ROOT", Path(directory)), patch(
                "cpuprofiler.benchmark_stay.run", side_effect=benchmark), redirect_stdout(io.StringIO()):
            main()
            with next(Path(directory).rglob("summary.csv")).open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 11)
            self.assertEqual(seen["P4_89"], "-2 default 0,1,2,3,8,9\n-1 output 0,1,2,3,8,9\n")
            self.assertEqual(seen["P4_HT_output_P4X4"],
                             "-2 default 0,1,2,3\n-1 output 0,1,2,3,4,5,8,9\n")
            self.assertEqual(rows[-1]["output_cpus"], "0,1,2,3,8,9")
            self.assertEqual(rows[-1]["output"], "1")


if __name__ == "__main__":
    unittest.main()
