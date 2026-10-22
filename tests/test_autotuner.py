"""Search contracts without running a performance benchmark."""
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpuprofiler.autotuner import AutoTuner
from cpuprofiler.benchmark import BenchmarkConfig, BenchmarkResult


def topology():
    return SimpleNamespace(
        cpus=dict.fromkeys(range(14)),
        physical_cores=[[0, 1], [2, 3], *[[i] for i in range(4, 14)]],
        l2_groups=[[0, 1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13]],
    )


class FakeRunner:
    def __init__(self, model_path):
        self.model_path, self.calls = model_path, []

    def run(self, config):
        self.calls.append(config)
        latency = 1 + (config.threads - 9) ** 2 / 100
        latency += 0.02 * (config.omp_schedule == "static")
        return BenchmarkResult(config, 1000 / latency, "fake")


class AutoTunerTest(unittest.TestCase):
    def test_candidates_keep_smt_l2_and_all_counts(self):
        tuner = AutoTuner(topology(), None)
        candidates = tuner._topology_candidates()
        self.assertEqual(len(candidates), 269)
        self.assertEqual(set(map(len, candidates)), set(range(1, 15)))
        self.assertTrue(any(len(set(mask) & {0, 1, 2, 3}) == 3
                            and len(set(mask) & set(range(4, 12))) == 6
                            and not set(mask) & {12, 13} for mask in candidates))
        # Same E count, both compact and spread L2 allocations remain candidates.
        e_pairs = [set(mask) for mask in candidates if len(mask) == 2
                   and set(mask) <= set(range(4, 12))]
        self.assertTrue(any(mask <= set(range(4, 8)) or mask <= set(range(8, 12)) for mask in e_pairs))
        self.assertTrue(any(mask & set(range(4, 8)) and mask & set(range(8, 12)) for mask in e_pairs))
        preferred = (0, 1, 2, 4, 5, 6, 7, 8, 9)
        self.assertIn(preferred, tuner._topology_candidates(preferred))

    def test_budget_preserved_options_and_independent_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.bin"
            model.write_bytes(struct.pack("i", 288))
            runner = FakeRunner(model)
            tuner = AutoTuner(topology(), runner)
            config = BenchmarkConfig(cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11), threads=10,
                                     steps=32, prompt="test", optimize_weights=True,
                                     two_stage_argmax=True, omp_places="old places")
            result = tuner.kasikoi_serch(config, budget=40, top_k=3, repeats=3)
            self.assertEqual(set(c.threads for c in runner.calls[1:15]), set(range(1, 15)))
            exploration = runner.calls[1:41]
            self.assertEqual(len(set(exploration)), 40)
            self.assertEqual(tuner.last_search["exploration"], 40)
            baseline = tuner.last_search["baseline"]
            final = tuner.last_search["confirmed"]
            self.assertIn(baseline, [r.config for r in final])
            self.assertEqual(len(runner.calls), 1 + 40 + 3 * len(final))
            for block in range(3):
                start = 41 + block * len(final)
                self.assertEqual(set(runner.calls[start:start + len(final)]), {r.config for r in final})
            self.assertEqual(result, min(final, key=lambda r: r.ms_per_token))
            for case in runner.calls:
                self.assertEqual(case.threads, len(case.cpus))
                self.assertIsNone(case.omp_places)
                self.assertTrue(set(case.cpus) <= set(range(14)))
                self.assertEqual((case.steps, case.prompt), (32, "test"))
                self.assertTrue(case.optimize_weights and case.two_stage_argmax)
                self.assertIn(case.group_size, (96, 144, 288))
            again = AutoTuner(topology(), FakeRunner(model))
            again.kasikoi_serch(config, budget=40, top_k=3, repeats=3)
            self.assertEqual(runner.calls, again.runner.calls)

    def test_invalid_budget_does_not_run_benchmarks(self):
        runner = FakeRunner(None)
        tuner = AutoTuner(topology(), runner)
        with self.assertRaises(ValueError):
            tuner.kasikoi_serch(budget=13)
        self.assertFalse(runner.calls)


if __name__ == "__main__":
    unittest.main()
