/* Deterministic MHA/GQA model with uneven FFN rows and a complete KV cache. */
#define TESTING
#include "../run.c"

int main(int argc, char **argv) {
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
    for (int i = 0; i < t.config.n_layers * t.config.dim; i++) {
        t.weights.rms_att_weight[i] = 1.0f;
        t.weights.rms_ffn_weight[i] = 1.0f;
    }
    for (int i = 0; i < t.config.dim; i++) t.weights.rms_final_weight[i] = 1.0f;
    malloc_run_state(&t.state, &t.config);
    for (int repeat = 0; repeat < 3; repeat++) {
        for (int pos = 0; pos < t.config.seq_len; pos++) {
            float *logits = forward(&t, (pos * 7 + repeat) % t.config.vocab_size, pos);
            fwrite(logits, sizeof(float), t.config.vocab_size, stdout);
            fwrite(t.state.hb2, sizeof(float), t.config.hidden_dim, stdout);
        }
    }
    int cache_size = t.config.n_layers * t.config.seq_len * t.config.dim
                     * t.config.n_kv_heads / t.config.n_heads;
    fwrite(t.state.key_cache, sizeof(float), cache_size, stdout);
    fwrite(t.state.value_cache, sizeof(float), cache_size, stdout);
    free_run_state(&t.state);
    free(weights);
    return 0;
}
