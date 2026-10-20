/* A deterministic model with uneven row counts, GQA, and fewer heads than workers. */
#define TESTING
#include "../run.c"

int main(int argc, char **argv) {
    #if defined(LLAMA_PARALLEL) || defined(LLAMA_EXPERIMENT)
    parallel_init();
    #endif
    Transformer t = {0};
    t.config = (Config){24, 37, 2, 6, argc > 1 ? atoi(argv[1]) : 2, 31, 9};
    float *weights = calloc(65536, sizeof(float));
    if (!weights) return 1;
    unsigned random = 42;
    for (int i = 0; i < 65536; i++) {
        random = random * 1664525u + 1013904223u;
        weights[i] = ((int)(random >> 16) - 32768) / 327680.0f;
    }
    memory_map_weights(&t.weights, &t.config, weights, 0);
    malloc_run_state(&t.state, &t.config);
    for (int repeat = 0; repeat < 3; repeat++) {
        for (int pos = 0; pos < t.config.seq_len; pos++) {
            float *logits = forward(&t, (pos * 7 + repeat) % t.config.vocab_size, pos);
            for (int i = 0; i < t.config.vocab_size; i++)
                if (!isfinite(logits[i])) return 2;
            fwrite(logits, sizeof(float), t.config.vocab_size, stdout);
        }
    }
    free_run_state(&t.state);
    free(weights);
    return 0;
}
