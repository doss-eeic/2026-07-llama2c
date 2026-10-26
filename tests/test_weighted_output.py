"""Check weighted output's row coverage, numerical parity, and CPU placement."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WeightedOutputTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "sched_getaffinity") and
                         set(range(14)).issubset(os.sched_getaffinity(0)),
                         "CPU placement audit requires Linux CPUs 0-13")
    def test_rows_and_placement(self):
        source = r'''
#define _GNU_SOURCE
#define TESTING
#include <omp.h>
#include <sched.h>
static int cpus[32];
static int audited_rank(void) {
    int rank = omp_get_thread_num();
    cpus[rank] = sched_getcpu();
    return rank;
}
#define omp_get_thread_num audited_rank
#include "SOURCE"
int main(void) {
    const int n = 31;
    float x[n], w[257*n], reference[257], actual[257];
    for (int j=0; j<n; j++) x[j] = (j-17)*0.03125f;
    for (int j=0; j<257*n; j++) w[j] = ((j*37)%101-50)*0.015625f;
    for (int threads=3; threads<=14; threads++) {
        out_threads = threads;
        for (int p=1; p<threads && p<=4; p++) {
            out_p = p;
            for (int weight=1; weight<=6; weight++) {
                out_wp = weight;
                for (int d=1; d<=257; d=d*2+1) {
                    matmul(reference,x,w,n,d);
                    for (int i=0;i<d;i++) actual[i] = NAN;
                    matmul_output(actual,x,w,n,d);
                    for (int i=0;i<d;i++) {
                        if (!isfinite(actual[i]) || fabsf(actual[i]-reference[i])>1e-5f) return 1;
                    }
                }
            }
        }
    }
    for (int t=0;t<14;t++) if(cpus[t] != t) return 2;
    puts("row coverage, numerical parity, and 14-rank CPU placement OK");
    return 0;
}
'''.replace("SOURCE", str(ROOT / "run.c"))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_"))}
        env.update(OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close",
                   OMP_PLACES=",".join(f"{{{c}}}" for c in range(14)), GOMP_SPINCOUNT="1000")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "check.c").write_text(source)
            for flags in (("-O2",), ("-Ofast", "-march=native")):
                subprocess.run(["gcc", *flags, "-fopenmp", str(path / "check.c"), "-lm", "-o", str(path / "check")],
                               check=True, capture_output=True)
                subprocess.run([str(path / "check")], env=env, check=True, capture_output=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
