/* Build without OpenMP so reference_forward is the original serial kernel. */
#define STAY_TESTING
#include "/home/denjo/2026-07-llama2c/tmp/stay_probe/probe.c"

static int visited[32];
static void visit(void *unused, int row) {
    (void)unused;
    visited[row]++;
}
int main(int argc, char **argv) {
    if (argc != 2) return 10;
    Config config = {.dim=8, .hidden_dim=12, .n_layers=2, .n_heads=2,
                     .n_kv_heads=1, .vocab_size=16, .seq_len=8};
    float weights[4096];
    for (int i = 0; i < 4096; i++) weights[i] = sinf(i * 0.17f) * 0.1f;
    Transformer original = {.config=config}, mixed = {.config=config};
    memory_map_weights(&original.weights, &config, weights, 0);
    memory_map_weights(&mixed.weights, &config, weights, 0);
    malloc_run_state(&original.state, &config);
    malloc_run_state(&mixed.state, &config);
    stay_init(argv[1], config.n_layers);
    stay_layers_count = stay_rule(9, -2)->count;
    stay_check(pthread_barrier_init(&stay_layers_barrier, NULL, stay_layers_count));
    // Verify exact coverage even when rows < participants and with disjoint rules.
    stay_dispatch("q_proj", 0, 1, visit, NULL);
    stay_dispatch("output", -1, 31, visit, NULL);
    for (int i = 0; i < 32; i++) {
        int expected = i == 0 ? 2 : i < 31 ? 1 : 0;
        if (visited[i] != expected) return 11;
    }
    for (int pos = 0; pos < 8; pos++) {
        int token = (pos * 3 + 1) % config.vocab_size;
        float *reference = reference_forward(&original, token, pos);
        float *actual = forward(&mixed, token, pos);
        if (memcmp(reference, actual, config.vocab_size * sizeof(float))) {
            fprintf(stderr, "logits differ at pos=%d\n", pos);
            return 12;
        }
    }
    stay_shutdown();
    stay_check(pthread_barrier_destroy(&stay_layers_barrier));
    free_run_state(&original.state);
    free_run_state(&mixed.state);
    puts("all logits bit-identical");
    return 0;
}
