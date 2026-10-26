"""Resident team identity, prefix participation, synchronization and numerical parity."""
from array import array
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "OpenMP resident pool on Linux")
class PersistentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.binaries = []
        for name, flags in (("O2", ["-O2"]), ("fast", ["-Ofast", "-march=native"])):
            binary = Path(cls.work.name) / name
            subprocess.run(["gcc", *flags, "-fopenmp", str(ROOT / "tests/persistent_forward.c"),
                            "-lm", "-o", str(binary)], check=True, capture_output=True, timeout=60)
            cls.binaries.append((name, binary))

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def environment(self, layer, output, schedule, wait="ACTIVE", enabled=True):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMP_", "GOMP_", "OUTPUT_", "PERSISTENT_"))}
        env.update(OMP_NUM_THREADS=str(layer), OMP_DYNAMIC="FALSE", OMP_SCHEDULE=schedule,
                   OMP_WAIT_POLICY=wait, OUTPUT_THREADS=str(output),
                   PERSISTENT_OPENMP=str(int(enabled)))
        return env

    def test_forward_matches_existing_openmp(self):
        for name, binary in self.binaries:
            for heads in (2, 6):
                for argmax in (0, 1):
                    args = [str(binary), str(heads), str(argmax)]
                    reference = subprocess.check_output(args, env=self.environment(1, 1, "static", enabled=False), timeout=20)
                    for schedule in ("static", "static,3", "dynamic,1", "guided,4"):
                        for layer, output in ((1, 1), (2, 4), (4, 2), (8, 14), (14, 8)):
                            with self.subTest(flags=name, heads=heads, argmax=argmax,
                                              schedule=schedule, layer=layer, output=output):
                                actual = subprocess.check_output(args, env=self.environment(layer, output, schedule),
                                                                 stderr=subprocess.PIPE, timeout=20)
                                if name == "O2":
                                    self.assertEqual(actual, reference)
                                else:
                                    expected_values, actual_values = array("f", reference), array("f", actual)
                                    self.assertEqual(len(expected_values), len(actual_values))
                                    for expected, value in zip(expected_values, actual_values):
                                        self.assertAlmostEqual(value, expected, delta=1e-6 + 1e-5 * abs(expected))

    def test_fixed_team_and_prefix_workers(self):
        binary = self.binaries[0][1]
        for wait in ("ACTIVE", "PASSIVE"):
            for schedule in ("static", "static,3", "dynamic,1", "guided,4"):
                subprocess.run([str(binary), "audit"], env=self.environment(8, 14, schedule, wait),
                               check=True, capture_output=True, timeout=20)

    def test_insufficient_team_and_dynamic_fail_without_hanging(self):
        binary = self.binaries[0][1]
        for extra in ({"OMP_THREAD_LIMIT": "4"}, {"OMP_DYNAMIC": "TRUE"}):
            env = self.environment(8, 14, "static") | extra
            result = subprocess.run([str(binary), "audit"], env=env, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
