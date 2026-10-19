"""Checks for log completion, time aggregation and child execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpuprofiler.benchmark_detail import parse_profile, run, summarize
from cpuprofiler.benchmarkconfig import BenchmarkConfig


def example_log():
    rows = [{"event": "meta", "schema": 1}]
    for fid in range(3):
        for layer in range(2):
            rows.append({"event": "timing", "forward_id": fid, "layer": layer,
                         "stage": "q_proj", "duration_ns": 1000000})
        rows.append({"event": "timing", "forward_id": fid, "layer": -1,
                     "stage": "forward_total", "duration_ns": 4000000})
    rows.append({"event": "end", "forward_calls": 3, "timing_records": 9})
    return "\n".join(json.dumps(row) for row in rows)


class BenchmarkTests(unittest.TestCase):
    def test_sum_layers_per_token_and_select_stages(self):
        profile = parse_profile(example_log())
        summary, stages = summarize(profile["timings"], 1, {"q_proj"})
        self.assertEqual(summary["forward_calls"], 3)
        self.assertEqual(summary["measured_tokens"], 2)
        self.assertEqual(summary["forward_ms_per_token"], 4.0)
        self.assertEqual(stages, [{"stage": "q_proj", "calls": 4,
                                   "mean_call_ms": 1.0, "ms_per_token": 2.0}])
        with self.assertRaises(ValueError):
            summarize(profile["timings"], 3)

    def test_incomplete_log_does_not_get_aggregated(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_profile(example_log().rsplit("\n", 1)[0])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            parse_profile(example_log().replace('"timing_records": 9', '"timing_records": 10'))

    def test_run_saves_logs_and_sets_cpu_placement(self):
        def child(command, **kwargs):
            self.assertEqual(command[:3], ["taskset", "-c", "4"])
            self.assertEqual(kwargs["env"]["OMP_PLACES"], "{4}")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertIn("0.25", command)
            Path(kwargs["env"]["LLAMA_PROFILE_PATH"]).write_text(example_log())
            kwargs["stdout"].write(b"generated text\n")

        with tempfile.TemporaryDirectory() as directory, patch(
                "cpuprofiler.benchmark_detail.subprocess.run", side_effect=child) as child_run:
            path = Path(directory) / "trial"
            config = BenchmarkConfig(cpus=(4,), threads=1, output=path, steps=32,
                                     temperature=0.25, stages=("q_proj",))
            summary, stages = run(config)
            child_run.assert_called_once()
            self.assertEqual(summary["forward_calls"], 3)
            self.assertTrue((path / "profile.jsonl").is_file())
            self.assertEqual((path / "stdout.txt").read_text(), "generated text\n")


if __name__ == "__main__":
    unittest.main()
