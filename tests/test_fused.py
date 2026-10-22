"""Compare all logits, gates, and KV-cache entries with unmodified upstream."""
from array import array
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "350e04fe35433e6d2941dce5a1f53308f87058eb"


class FusedTest(unittest.TestCase):
    def test_forward_matches_unfused(self):
        compiler = shlex.split(os.environ.get("CC", "cc"))
        openmp = os.environ.get("TEST_OPENMP") == "1"
        harness = (ROOT / "tests/fused_forward.c").read_text()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            baseline = work / "baseline.c"
            baseline.write_bytes(subprocess.check_output(
                ["git", "show", f"{BASE_REVISION}:run.c"], cwd=ROOT
            ))
            for flags in (("-O2",), ("-Ofast", "-march=native")):
                binaries = []
                for name, source in (("reference", baseline), ("fused", ROOT / "run.c")):
                    test_source, binary = work / f"{name}_test.c", work / name
                    test_source.write_text(harness.replace('#include "../run.c"', f'#include "{source}"'))
                    subprocess.run(compiler + list(flags) + (["-fopenmp"] if openmp else []) + [
                        str(test_source), "-lm", "-o", str(binary)
                    ], check=True, capture_output=True, timeout=60)
                    binaries.append(binary)
                for heads in (2, 6):
                    for threads in ((1, 4, 14) if openmp else (1,)):
                        with self.subTest(flags=flags, kv_heads=heads, threads=threads):
                            env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_", "OUTPUT_"))}
                            env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                                       OMP_PROC_BIND="FALSE")
                            outputs = [subprocess.check_output([binary, str(heads)], env=env, timeout=30)
                                       for binary in binaries]
                            reference, actual = (array("f", output) for output in outputs)
                            self.assertEqual(len(actual), len(reference))
                            self.assertTrue(all(math.isfinite(v) for v in (*reference, *actual)))
                            if flags == ("-O2",):
                                self.assertEqual(outputs[0], outputs[1])
                            else:
                                for expected, value in zip(reference, actual):
                                    self.assertAlmostEqual(value, expected, delta=1e-6 + 1e-5 * abs(expected))


if __name__ == "__main__":
    unittest.main()
