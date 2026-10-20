"""Compare every logit with the shared upstream kernel, without model downloads."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StayTest(unittest.TestCase):
    def test_logits_and_settings(self):
        compiler = shlex.split(os.environ.get("CC", "cc"))
        cpus = sorted(os.sched_getaffinity(0))[:8] if hasattr(os, "sched_getaffinity") else list(range(8))
        with tempfile.TemporaryDirectory() as directory:
            reference, stay = (Path(directory) / name for name in ("reference", "stay"))
            for flags in (("-O2",), ("-Ofast", "-march=native")):
                for binary, extra in ((reference, []), (stay, ["-DLLAMA_STAY", "-DSTAY_TEST_UNPINNED", "-pthread"])):
                    subprocess.run(compiler + list(flags) + extra + [
                        str(ROOT / "tests/stay_forward.c"), "-lm", "-o", str(binary)
                    ], check=True, capture_output=True, timeout=60)
                for kv_heads in (2, 6):
                    expected = subprocess.check_output([reference, str(kv_heads)], timeout=15)
                    for count in sorted({1, min(4, len(cpus)), len(cpus)}):
                        for layers in sorted({1, min(4, count), count}):
                            for weight in (1, 2):
                                env = dict(os.environ, STAY_CPUS=",".join(map(str, cpus[:count])),
                                           STAY_LAYER_THREADS=str(layers), STAY_P_WEIGHT=str(weight))
                                with self.subTest(flags=flags, kv_heads=kv_heads, count=count, layers=layers, weight=weight):
                                    actual = subprocess.check_output([stay, str(kv_heads)], env=env, timeout=15)
                                    self.assertEqual(actual, expected)
            base = dict(os.environ, STAY_CPUS=str(cpus[0]), STAY_LAYER_THREADS="1", STAY_P_WEIGHT="2")
            for settings in ({"STAY_CPUS": ""}, {"STAY_CPUS": f"{cpus[0]},{cpus[0]}"},
                             {"STAY_CPUS": "0,"}, {"STAY_LAYER_THREADS": "2"}, {"STAY_P_WEIGHT": "0"}):
                result = subprocess.run([stay], env=dict(base, **settings), capture_output=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"stay:", result.stderr)


if __name__ == "__main__":
    unittest.main()
