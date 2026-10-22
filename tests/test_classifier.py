"""Screened argmax matches full fp32, including ties, zero rows and small gaps."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "Linux classifier allocation")
class ClassifierTest(unittest.TestCase):
    def test_argmax_and_retained_logits(self):
        harness = r'''
#define TESTING
#include "RUN_SOURCE"
static unsigned rng = 42;
static float random_float(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 16) - 32768) / 32768.0f;
}
int main(int argc, char **argv) {
    const int n = 288, d = 67;
    float w[d*n], x[n], full[d], screened[d];
    for (int scenario = 0; scenario < 3; scenario++) {
        for (int i = 0; i < d*n; i++) w[i] = scenario == 0 ? 0.0f : random_float();
        // Equal top rows and near-equal scores exercise candidate retention and tie order.
        if (scenario == 2) {
            for (int i = 0; i < d; i++) {
                memcpy(w+i*n, w, n*sizeof(float));
                if (i > 1) w[i*n] += i * 1e-7f;
            }
        }
        setup_classifier(w, n, d, atoi(argv[1]), 0);
        for (int trial = 0; trial < 40; trial++) {
            for (int j = 0; j < n; j++) x[j] = random_float() * (trial % 2 ? 1e4f : 1e-4f);
            matmul(full, x, w, n, d);
            classifier_int8(screened, x, w, n, d);
            if (sample_argmax(full, d) != sample_argmax(screened, d)) {
                fprintf(stderr, "argmax mismatch: scenario=%d trial=%d\n", scenario, trial);
                return 1;
            }
            for (int i = 0; i < d; i++) {
                if (screened[i] != -FLT_MAX && memcmp(&screened[i], &full[i], sizeof(float))) {
                    fprintf(stderr, "logit mismatch: scenario=%d trial=%d row=%d\n", scenario, trial, i);
                    return 2;
                }
            }
        }
        free_classifier();
    }
    return 0;
}
'''.replace("RUN_SOURCE", str(ROOT / "run.c"))
        with tempfile.TemporaryDirectory() as directory:
            source, binary = Path(directory) / "check.c", Path(directory) / "check"
            source.write_text(harness)
            openmp = os.environ.get("TEST_OPENMP") == "1"
            for flags in (("-O2",), ("-Ofast", "-march=native")):
                subprocess.run(shlex.split(os.environ.get("CC", "cc")) + list(flags) +
                               (["-fopenmp"] if openmp else []) +
                               [str(source), "-lm", "-o", str(binary)],
                               check=True, capture_output=True, timeout=60)
                for threads in ((1, 4, 14) if openmp else (1,)):
                    for schedule in (("static", "dynamic,16") if openmp else ("static",)):
                        env = {k: v for k, v in os.environ.items()
                               if not k.startswith(("OMP_", "GOMP_", "OUTPUT_"))}
                        env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                                   OMP_SCHEDULE=schedule)
                        for gs in (1, 16, 144, 288):
                            with self.subTest(flags=flags, threads=threads, schedule=schedule, gs=gs):
                                subprocess.run([str(binary), str(gs)], env=env, check=True,
                                               capture_output=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
