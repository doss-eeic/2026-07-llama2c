"""Meaningful policy-selection and persistent-worker numerical regression checks."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import struct
from types import SimpleNamespace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpuprofiler.autotune_detail import (
    STAGES, analyze, blocked_jobs, generate_candidates, select_policy, validate, write_policy,
)


def trial(cpus, slow_stage, total):
    records = []
    for fid in range(3):
        for stage in STAGES:
            records.append({"event": "timing", "layer": -1 if stage == "output" else 0,
                            "stage": stage, "forward_id": fid, "pos": fid,
                            "input_token": fid + 1, "duration_ns": 9000000 if stage == slow_stage else 1000000})
        records.append({"event": "timing", "layer": -1, "stage": "forward_total",
                        "forward_id": fid, "pos": fid, "input_token": fid + 1,
                        "duration_ns": int(total * 1e6)})
    operators = {(row["layer"], row["stage"]): row["duration_ns"] / 1e6
                 for row in records if row["stage"] in STAGES}
    return {"cpus": cpus, "operators": operators, "forward_ms": total,
            "profile": {"meta": {"event": "meta", "n_layers": 1}, "timings": records}}


class TunerTests(unittest.TestCase):
    def test_exhaustive_subset_domain(self):
        self.assertEqual(generate_candidates((4, 1, 7), 2),
                         [(1,), (4,), (7,), (1, 4), (1, 7), (4, 7)])
        with self.assertRaises(ValueError):
            generate_candidates((1, 1))
        with self.assertRaises(ValueError):
            generate_candidates(tuple(range(20)))

    def test_operator_choices_and_real_forward_baseline(self):
        result = select_policy([trial((1,), "output", 20), trial((2, 3), "q_proj", 25)])
        self.assertEqual(result["baseline_cpus"], [1])
        choices = {c["stage"]: c["cpus"] for c in result["choices"]}
        self.assertEqual(choices["output"], [2, 3])
        self.assertEqual(choices["q_proj"], [1])
        self.assertEqual(result["estimated_operator_ms"], len(STAGES))

    def test_trial_median_resists_one_outlier(self):
        a = trial((1,), "output", 20)
        outlier = deepcopy(a)
        outlier["operators"][(0, "q_proj")] = 1000
        other = trial((2,), "output", 21)
        other["operators"][(0, "q_proj")] = 2
        result = select_policy([a, a, outlier, other])
        self.assertEqual(next(c for c in result["choices"] if c["stage"] == "q_proj")["cpus"], [1])

    def test_rejects_changed_workload(self):
        a, b = trial((1,), "output", 20), trial((2,), "q_proj", 25)
        b["profile"]["timings"][-1]["input_token"] = 99
        with self.assertRaisesRegex(ValueError, "workload"):
            select_policy([a, b])

    def test_analyze_saved_raw_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for i, cpus in enumerate(((1,), (2, 3))):
                directory = output / f"trial-{i}"
                directory.mkdir()
                item = trial(cpus, "output" if i == 0 else "q_proj", 20 + i)
                (directory / "config.json").write_text(json.dumps({"cpus": cpus, "threads": len(cpus)}))
                records = item["profile"]["timings"]
                rows = [item["profile"]["meta"], *records,
                        {"event": "end", "forward_calls": 3, "timing_records": len(records)}]
                (directory / "profile.jsonl").write_text("\n".join(map(json.dumps, rows)))
            result, _ = analyze(output, 1)
            self.assertEqual(result["baseline_cpus"], [1])
            self.assertIn("-1 output 2,3", (output / "policy.txt").read_text())
            raw = output / "trial-0/profile.jsonl"
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            rows = [r for r in rows if r.get("stage") != "attention"]
            rows[-1]["timing_records"] -= 3
            raw.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "incomplete operator"):
                analyze(output, 1)

    def test_randomization_keeps_repeat_blocks(self):
        jobs = blocked_jobs(["a", "b", "c"], 3, 42)
        self.assertEqual(jobs, blocked_jobs(["a", "b", "c"], 3, 42))
        for repeat in range(3):
            self.assertEqual({r for r, _ in jobs[repeat * 3:(repeat + 1) * 3]}, {repeat})
            self.assertEqual({c for _, c in jobs[repeat * 3:(repeat + 1) * 3]}, {"a", "b", "c"})


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.binary = cls.directory / "stay-test"
        command = [compiler, "-O3", "-pthread", "-Wno-unknown-pragmas"]
        if sys.platform != "linux":
            # Numerical/synchronization tests only. Production builds never bypass pinning.
            command.append("-DSTAY_TEST_UNPINNED")
        subprocess.run([*command, str(ROOT / "tests/stay_correctness.c"), "-lm", "-o", str(cls.binary)], check=True)
        cls.cpus = sorted(os.sched_getaffinity(0))[:3] if hasattr(os, "sched_getaffinity") else [0, 1, 2]
        cls.cli = cls.directory / "stay-cli"
        subprocess.run([*command, "-DLLAMA_PROFILE", str(ROOT / "run_stay.c"),
                        "-lm", "-o", str(cls.cli)], check=True)
        cls.model = cls.directory / "model.bin"
        weights = [0.01 * ((i % 11) - 5) for i in range(4096)]
        cls.model.write_bytes(struct.pack("7i", 8, 12, 2, 2, 1, 16, 8) + struct.pack("4096f", *weights))
        cls.tokenizer = cls.directory / "tokenizer.bin"
        cls.tokenizer.write_bytes(struct.pack("i", 1) + (struct.pack("fi", 0.0, 1) + b"X") * 16)


    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, policy):
        path = self.directory / "policy.txt"
        path.write_text(policy)
        return subprocess.run([str(self.binary), str(path)], capture_output=True, text=True, timeout=20)

    def test_disjoint_workers_and_layer_overrides_preserve_logits(self):
        a, b = self.cpus[0], self.cpus[-1]
        all_cpus = ",".join(map(str, self.cpus))
        result = self.execute(f"-2 default {a}\n-2 q_proj {all_cpus}\n0 q_proj {b}\n1 attention {b}\n-1 output {all_cpus}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bit-identical", result.stdout)

    def test_replay_validation_and_refinement_with_real_c_program(self):
        if len(self.cpus) < 2:
            self.skipTest("two available CPUs required")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trials = [trial((self.cpus[0],), "output", 20),
                      trial((self.cpus[-1],), "q_proj", 25)]
            selection = select_policy(trials)
            write_policy(output / "policy.txt", selection["baseline_cpus"], selection["choices"])
            args = SimpleNamespace(repeats=1, order_seed=42, stay_binary=self.cli,
                                   model=self.model, tokenizer=self.tokenizer, seed=42,
                                   timeout=10, skip_tokens=1, refine_rounds=1)
            with redirect_stdout(io.StringIO()):
                report = validate(output, args, selection, trials)
            self.assertTrue(report["greedy_tokens_match"])
            self.assertEqual(len(report["trials"]), 3)
            self.assertEqual(report["refinement"]["steps"], len(STAGES))
            self.assertTrue((output / "recommended-policy.txt").is_file())
            self.assertTrue((output / "refinement.json").is_file())
            rows = [json.loads(line) for line in next(output.glob("validation-*/profile.jsonl")).read_text().splitlines()]
            timings = [row for row in rows if row["event"] == "timing"]
            self.assertEqual(len(timings), 3)
            self.assertEqual({row["stage"] for row in timings}, {"forward_total"})

    def test_invalid_policy_fails_before_execution(self):
        cpu = self.cpus[0]
        for policy in (f"-2 default {cpu},{cpu}\n", f"-2 unknown {cpu}\n",
                       f"-2 default {cpu}\n-2 default {cpu}\n", f"0 output {cpu}\n",
                       f"-2 default {cpu},\n"):
            result = self.execute(policy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run_stay:", result.stderr)


if __name__ == "__main__":
    unittest.main()
