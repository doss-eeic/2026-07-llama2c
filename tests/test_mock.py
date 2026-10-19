from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cpuprofiler.benchmark import BenchmarkConfig, BenchmarkResult
from cpuprofiler.dataset import BenchmarkDataset
from cpuprofiler.mock import MockBenchmarkRunner


class MockTest(unittest.TestCase):
    def test_replay_cost_reset_and_no_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.sqlite3'
            config = BenchmarkConfig((0, 1), 2, steps=120)
            dataset = BenchmarkDataset(path)
            for repetition, tps in enumerate((10., 30., 20.)):
                dataset.record(BenchmarkResult(config, tps, str(repetition)), repetition, repetition + 1.)
            dataset.close()
            with patch('subprocess.run', side_effect=AssertionError('must not infer')):
                runner = MockBenchmarkRunner(path)
                self.assertEqual([runner.run(config).tokens_per_second for _ in range(4)], [10., 30., 20., 10.])
                self.assertEqual(runner.elapsed_seconds, 7.)
                self.assertEqual(runner.calls, 4)
                runner.reset()
                self.assertEqual(runner.calls, 0)
                self.assertEqual(runner.run(config).raw_stderr, '0')
                median = MockBenchmarkRunner(path, 'median').run(config)
                self.assertEqual((median.tokens_per_second, median.elapsed_seconds), (20., 3.))
                random = MockBenchmarkRunner(path, 'random', seed=7)
                first = [random.run(config).tokens_per_second for _ in range(10)]
                random.reset()
                self.assertEqual(first, [random.run(config).tokens_per_second for _ in range(10)])
                with self.assertRaises(KeyError):
                    runner.run(replace(config, steps=256))
                self.assertEqual(runner.calls, 1)
