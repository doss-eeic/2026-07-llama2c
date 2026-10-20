"""Same row kernels under pthread/OpenMP, including independent fusion switches."""
import array
import itertools
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FactorialTest(unittest.TestCase):
    def test_all_switches_and_pool_sizes(self):
        cpus = sorted(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as directory:
            binaries = {name: Path(directory) / name for name in ('reference', 'openmp', 'pthread')}
            for flags in (['-O2'], ['-Ofast', '-march=native']):
                for name, binary in binaries.items():
                    extra = {'reference': [], 'openmp': ['-DLLAMA_PARALLEL', '-DLLAMA_EXPERIMENT', '-fopenmp'],
                             'pthread': ['-DLLAMA_STAY', '-DLLAMA_EXPERIMENT', '-pthread']}[name]
                    subprocess.run(['gcc', *flags, *extra, str(ROOT / 'tests/stay_forward.c'),
                                    '-lm', '-o', str(binary)], check=True, capture_output=True)
                for kv in (2, 6):
                    expected = subprocess.check_output([binaries['reference'], str(kv)])
                    reference = array.array('f'); reference.frombytes(expected)
                    for count in sorted({1, min(4, len(cpus)), len(cpus)}):
                        for backend, dynamic, qkv, ffn, block in itertools.product(
                                ('openmp', 'pthread'), (0, 1), (0, 1), (0, 1), (1, 4)):
                            env = dict(os.environ, OMP_NUM_THREADS=str(count), OMP_DYNAMIC='FALSE',
                                       OMP_SCHEDULE='dynamic,2' if dynamic else 'static',
                                       STAY_CPUS=','.join(map(str, cpus[:count])), STAY_LAYER_THREADS=str(count),
                                       STAY_P_WEIGHT='1', OUTPUT_THREADS=str(count), OUTPUT_CHUNK='4',
                                       OUTPUT_DYNAMIC=str(dynamic), PARALLEL_DYNAMIC=str(dynamic),
                                       PARALLEL_CHUNK='2', PARALLEL_FUSE_QKV=str(qkv), PARALLEL_FUSE_FFN=str(ffn),
                                       PARALLEL_OUTPUT_BLOCK=str(block), PARALLEL_ATTENTION_THREADS='0')
                            data = subprocess.check_output([binaries[backend], str(kv)], env=env, timeout=15)
                            actual = array.array('f'); actual.frombytes(data)
                            with self.subTest(flags=flags, kv=kv, count=count, backend=backend,
                                              dynamic=dynamic, qkv=qkv, ffn=ffn, block=block):
                                if flags == ['-O2']:
                                    self.assertEqual(data, expected)
                                else:
                                    self.assertEqual(len(actual), len(reference))
                                    self.assertTrue(all(math.isfinite(a) and abs(a-b) <= 1e-7
                                                        for a,b in zip(actual, reference)))
                    # Exercise the independent output queue with only some workers in the layers.
                    count = min(4, len(cpus))
                    env.update(STAY_CPUS=','.join(map(str, cpus[:count])), STAY_LAYER_THREADS='1',
                               OUTPUT_DYNAMIC='1', PARALLEL_DYNAMIC='1', PARALLEL_FUSE_QKV='1',
                               PARALLEL_FUSE_FFN='1')
                    actual = array.array('f')
                    actual.frombytes(subprocess.check_output([binaries['pthread'], str(kv)], env=env, timeout=15))
                    self.assertTrue(all(math.isfinite(a) and abs(a-b) <= 1e-7 for a,b in zip(actual,reference)))
