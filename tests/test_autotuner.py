"""Search contracts without running a performance benchmark."""
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpuprofiler.autotuner import AutoTuner, DomainCalibrator
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

    def kasikoi2(self, model, **options):
        config = BenchmarkConfig(cpus=(0, 2, 4, 5, 6, 7, 8, 9, 10, 11), threads=10,
                                 steps=32, prompt="test", optimize_weights=True,
                                 two_stage_argmax=True, omp_places="old places")
        tuner = AutoTuner(topology(), FakeRunner(model))
        result = tuner.kasikoi2_serch(config, **options)
        return tuner, result

    def test_kasikoi2_grows_one_cpu_per_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.bin"
            model.write_bytes(struct.pack("i", 288))
            tuner, result = self.kasikoi2(model)
            calls, stages = tuner.runner.calls, tuner.last_search["stages"]
            self.assertEqual([len(next(iter(stage))) for stage in stages], list(range(1, 15)))
            # 1スレッドはP/E/LP-Eの3型だけ。以降は最良配置に1CPU足した配置だけ。
            self.assertEqual(len(stages[0]), 3)
            for before, after in zip(stages, stages[1:]):
                best = set(min(before, key=before.get))
                self.assertTrue(all(len(set(mask) - best) == 1 and best < set(mask) for mask in after))
            calibration = 14 * 3 + 4 * 3
            exploration = sum(map(len, stages))
            grown = calls[1 + calibration:1 + calibration + exploration]
            self.assertEqual([c.threads for c in grown], sorted(c.threads for c in grown))
            final = tuner.last_search["confirmed"]
            self.assertIn(tuner.last_search["baseline"], [r.config for r in final])
            self.assertEqual(len(calls), 1 + calibration + tuner.last_search["exploration"] + 3 * len(final))
            self.assertEqual(result, min(final, key=lambda r: r.ms_per_token))
            for case in calls:
                self.assertEqual(case.threads, len(case.cpus))
                self.assertIsNone(case.omp_places)
                self.assertTrue(case.optimize_weights and case.two_stage_argmax)

            # calibratedを渡せばキャリブレーションを再測定しない。結果は再現する。
            again, _ = self.kasikoi2(model, calibrated=tuner.last_search["calibrated"])
            self.assertEqual(again.runner.calls, calls[:1] + calls[1 + calibration:])

    def test_kasikoi2_invalid_options_do_not_run_benchmarks(self):
        runner = FakeRunner(None)
        with self.assertRaises(ValueError):
            AutoTuner(topology(), runner).kasikoi2_serch(top_k=0)
        self.assertFalse(runner.calls)


class CalibrationRunner:
    """CPU 8-11だけ単独性能を変え、slowで全体性能だけを変える。"""
    def __init__(self, e2_single=1.0, e2_full=1.0):
        self.e2_single, self.e2_full, self.calls = e2_single, e2_full, []

    def __call__(self, config):
        self.calls.append(config)
        latency = 10.0 if set(config.cpus) & {0, 1, 2, 3} else 20.0
        if set(config.cpus) <= set(range(8, 12)):
            latency *= self.e2_single if len(config.cpus) == 1 else self.e2_full
        return BenchmarkResult(config, 1000 / latency, "fake")


class DomainCalibratorTest(unittest.TestCase):
    def classes(self, runner):
        calibrated = DomainCalibrator(topology(), runner, repeats=3).calibrate()
        return calibrated, [tuple(d.id for d in c.domains) for c in calibrated.classes]

    def test_equal_domains_share_class(self):
        runner = CalibrationRunner()
        calibrated, classes = self.classes(runner)
        self.assertEqual(classes, [(0, 1), (2, 3), (4,)])
        # 単一CPU 14×3、比較相手のいるDomain全体 4×3
        self.assertEqual(len(runner.calls), 14 * 3 + 4 * 3)
        self.assertIsNone(calibrated.profiles[4].full_mean)
        self.assertEqual(calibrated.profiles[3].per_cpu_means, (20.0,) * 4)
        self.assertEqual(len(AutoTuner(topology(), None)._topology_candidates(calibrated=calibrated)), 269)

    def test_single_or_full_difference_splits_class(self):
        self.assertEqual(self.classes(CalibrationRunner(e2_single=1.2))[1], [(0, 1), (2,), (3,), (4,)])
        self.assertEqual(self.classes(CalibrationRunner(e2_full=1.2))[1], [(0, 1), (2,), (3,), (4,)])
        self.assertEqual(self.classes(CalibrationRunner(e2_single=1.03))[1], [(0, 1), (2, 3), (4,)])

    def test_split_class_adds_asymmetric_candidates(self):
        calibrated, _ = self.classes(CalibrationRunner(e2_single=1.2))
        tuner = AutoTuner(topology(), None)
        split = tuner._topology_candidates(calibrated=calibrated)
        self.assertGreater(len(split), 269)
        singles = [set(m) for m in split if len(m) == 1]
        # 分かれたE Domainはそれぞれ単独で試す
        self.assertTrue(any(m <= set(range(4, 8)) for m in singles))
        self.assertTrue(any(m <= set(range(8, 12)) for m in singles))


if __name__ == "__main__":
    unittest.main()
