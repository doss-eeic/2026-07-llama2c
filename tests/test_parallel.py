"""Check fused OpenMP kernels against every reference logit, including GQA."""
import os
import array
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ParallelTest(unittest.TestCase):
    def test_logits_match_reference(self):
        cpus = sorted(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as directory:
            reference, parallel = [Path(directory) / name for name in ("reference", "parallel")]
            for flags in (("-O2",), ("-Ofast", "-march=native")):
                for binary, extra in ((reference, []), (parallel, ["-DLLAMA_PARALLEL", "-fopenmp"])):
                    subprocess.run(["gcc", *flags, *extra, str(ROOT / "tests/stay_forward.c"),
                                    "-lm", "-o", str(binary)], check=True, capture_output=True)
                for kv_heads in (2, 6):
                    expected = subprocess.check_output([reference, str(kv_heads)])
                    for threads in sorted({1, min(4, len(cpus)), len(cpus)}):
                        for fuse in (0, 1):
                            for block in (1, 4):
                                env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                                           OMP_SCHEDULE="dynamic,2", OUTPUT_THREADS=str(threads),
                                           OUTPUT_CHUNK="2", PARALLEL_FUSE=str(fuse), PARALLEL_OUTPUT_BLOCK=str(block))
                                actual = subprocess.check_output([parallel, str(kv_heads)], env=env, timeout=30)
                                with self.subTest(flags=flags, kv_heads=kv_heads, threads=threads, fuse=fuse):
                                    if flags == ("-O2",):
                                        self.assertEqual(actual, expected)
                                    else:
                                        # -Ofast may choose a different reduction/vectorization
                                        # inside the OpenMP outlined kernels (observed < 4e-9).
                                        lhs, rhs = array.array("f"), array.array("f")
                                        lhs.frombytes(actual)
                                        rhs.frombytes(expected)
                                        self.assertEqual(len(lhs), len(rhs))
                                        self.assertTrue(all(math.isfinite(a) and abs(a-b) <= 1e-7
                                                            for a, b in zip(lhs, rhs)))
