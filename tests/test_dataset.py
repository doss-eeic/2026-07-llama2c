import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cpuprofiler.benchmark import BenchmarkResult, BenchmarkConfig
from cpuprofiler.dataset import BenchmarkDataset, ReplayRunner, collect, collection_lock


class DatasetTest(unittest.TestCase):
    def test_collection_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.sqlite3'
            with collection_lock(path):
                with self.assertRaises(RuntimeError):
                    with collection_lock(path):
                        self.fail('second collector acquired lock')
            with collection_lock(path):
                pass

    def test_resume_and_replay(self):
        topology = SimpleNamespace(cpus={0: None, 1: None, 2: None}, n_logical_cpus=3)
        calls = []

        class Runner:
            def run(self, config):
                calls.append(config)
                return BenchmarkResult(config, float(sum(config.cpus) + len(calls)), 'raw')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.sqlite3'
            dataset = BenchmarkDataset(path, {'machine': 'test'})
            # Simulate interruption after a committed measurement.
            config = BenchmarkConfig((0,), 1, steps=32)
            dataset.record(BenchmarkResult(config, 10., 'first'), 0)
            dataset.record(BenchmarkResult(config, 999., 'duplicate'), 0)
            self.assertEqual(dataset.samples(config), [(0, 10., 'first')])
            dataset.close()
            dataset = BenchmarkDataset(path, {'machine': 'test'})
            collect(topology, Runner(), dataset, (32, 64), 3, 3, 2)
            self.assertEqual(len(calls), 35)  # six configs x two budgets x three trials, minus saved sample
            self.assertEqual(len(dataset.samples(config)), 3)
            elapsed = dataset.db.execute('SELECT elapsed_seconds FROM samples WHERE elapsed_seconds IS NOT NULL').fetchall()
            self.assertEqual(len(elapsed), 35)
            self.assertTrue(all(row[0] >= 0 for row in elapsed))
            before = len(calls)
            collect(topology, Runner(), dataset, (32, 64), 3, 3, 2)
            self.assertEqual(len(calls), before)
            values = [row[1] for row in dataset.samples(config)]
            replay = ReplayRunner(dataset, 'cycle')
            self.assertEqual([replay.run(config).tokens_per_second for _ in range(4)], values + values[:1])
            self.assertEqual(ReplayRunner(dataset).run(config).tokens_per_second, sorted(values)[1])
            with self.assertRaises(KeyError):
                replay.run(BenchmarkConfig((0,), 1, steps=128))
            with self.assertRaises(ValueError):
                BenchmarkDataset(path, {'machine': 'other'})
            dataset.close()


if __name__ == '__main__':
    unittest.main()
