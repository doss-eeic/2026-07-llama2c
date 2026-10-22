"""Weight placement preserves checkpoint bytes, logits, gates, and KV caches."""
from array import array
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "Linux anonymous hugepage mappings")
class WeightsTest(unittest.TestCase):
    def test_placement_preserves_weights_and_forward(self):
        harness = r'''
#define TESTING
#include "RUN_SOURCE"
int main(int argc, char **argv) {
    Transformer t;
    int optimize_weights = atoi(argv[2]);
    build_transformer(&t, argv[1]);
    size_t nbytes = t.file_size - sizeof(Config);
    if (optimize_weights) {
        use_hugepages(&t);
        if ((uintptr_t)t.weights.token_embedding_table % (2 * 1024 * 1024) != 0) return 2;
    }
    fwrite(t.weights.token_embedding_table, 1, nbytes, stdout);
    for (int pos = 0; pos < t.config.seq_len; pos++) {
        float *logits = forward(&t, pos * 7 % t.config.vocab_size, pos);
        fwrite(logits, sizeof(float), t.config.vocab_size, stdout);
        fwrite(t.state.hb2, sizeof(float), t.config.hidden_dim, stdout);
    }
    int cache_size = t.config.n_layers * t.config.seq_len * t.config.dim
                     * t.config.n_kv_heads / t.config.n_heads;
    fwrite(t.state.key_cache, sizeof(float), cache_size, stdout);
    fwrite(t.state.value_cache, sizeof(float), cache_size, stdout);
    free_transformer(&t);
    return 0;
}
'''.replace("RUN_SOURCE", str(ROOT / "run.c"))
        payload = array("f", ((i % 251 - 125) / 1024 for i in range(65536))).tobytes()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, binary = work / "check.c", work / "check"
            source.write_text(harness)
            flags = ["-fopenmp"] if os.environ.get("TEST_OPENMP") == "1" else []
            subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
                "-O2", *flags, str(source), "-lm", "-o", str(binary)
            ], check=True, capture_output=True, timeout=60)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("OMP_", "GOMP_", "OUTPUT_"))}
            env.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", OMP_SCHEDULE="dynamic,16")
            for shared in (False, True):
                for kv_heads in (2, 6):
                    with self.subTest(shared=shared, kv_heads=kv_heads):
                        checkpoint = work / "model.bin"
                        checkpoint.write_bytes(struct.pack("7i", 24, 37, 2, 6, kv_heads,
                                                           31 if shared else -31, 9) + payload)
                        outputs = [subprocess.check_output([
                            str(binary), str(checkpoint), str(int(enabled))
                        ], env=env, timeout=30) for enabled in (False, True)]
                        self.assertEqual(outputs[0][:len(payload)], payload)
                        for output in outputs[1:]:
                            self.assertEqual(output, outputs[0])


if __name__ == "__main__":
    unittest.main()
