#define TESTING
#include "../run.c"
#include <sys/syscall.h>

static void forward_iterations(void* context) {
    Transformer* t = context;
    for (int repeat = 0; repeat < 2; repeat++) {
        for (int pos = 0; pos < t->config.seq_len; pos++) {
            float* logits = forward(t, (pos * 7 + repeat) % t->config.vocab_size, pos);
            fwrite(logits, sizeof(float), t->config.vocab_size, stdout);
            fwrite(t->state.hb2, sizeof(float), t->config.hidden_dim, stdout);
        }
    }
    int n = t->config.n_layers * t->config.seq_len * t->config.dim
            * t->config.n_kv_heads / t->config.n_heads;
    fwrite(t->state.key_cache, sizeof(float), n, stdout);
    fwrite(t->state.value_cache, sizeof(float), n, stdout);
}

static int visits[257], owner[257];
static long native_ids[64];

static void audit_rows(int begin, int end, void* context) {
    int rank = omp_pool_rank();
    long id = syscall(SYS_gettid);
    if (omp_get_level() != 1 || (native_ids[rank] && native_ids[rank] != id)) exit(2);
    native_ids[rank] = id;
    for (int row = begin; row < end; row++) {
        __atomic_fetch_add(&visits[row], 1, __ATOMIC_RELAXED);
        owner[row] = rank;
    }
}

static void audit(void* context) {
    if (omp_get_num_threads() != omp_pool_total) exit(3);
    for (int job = 0; job < 300; job++) {
        omp_pool_output_stage = job % 2;
        int active = omp_pool_output_stage ? omp_pool_output_threads : omp_pool_layer_threads;
        memset(visits, 0, sizeof(visits));
        omp_pool_rows(257, audit_rows, NULL);
        for (int row = 0; row < 257; row++) {
            if (visits[row] != 1 || owner[row] >= active) exit(4);
        }
    }
    omp_pool_output_stage = 0;
}

static void no_jobs(void* context) {}

int main(int argc, char** argv) {
    omp_pool_init();
    if (!strcmp(argv[1], "audit")) {
        omp_pool_run(audit, NULL);
        omp_pool_run(no_jobs, NULL);  // shutdown before the first numerical job
        memset(native_ids, 0, sizeof(native_ids));
        omp_pool_run(audit, NULL);    // a new invocation must not observe the old stop flag
        return 0;
    }
    Transformer t = {0};
    t.config = (Config){24, 37, 2, 6, atoi(argv[1]), 31, 6};
    float* weights = calloc(65536, sizeof(float));
    if (!weights) return 1;
    unsigned random = 42;
    for (int i = 0; i < 65536; i++) {
        random = random * 1664525u + 1013904223u;
        weights[i] = ((int)(random >> 16) - 32768) / 327680.0f;
    }
    memory_map_weights(&t.weights, &t.config, weights, 0);
    for (int i = 0; i < t.config.n_layers * t.config.dim; i++) {
        t.weights.rms_att_weight[i] = t.weights.rms_ffn_weight[i] = 1.0f;
    }
    for (int i = 0; i < t.config.dim; i++) t.weights.rms_final_weight[i] = 1.0f;
    malloc_run_state(&t.state, &t.config);
    if (atoi(argv[2])) setup_classifier(t.weights.wcls, 24, 31, 6, 0);
    omp_pool_run(forward_iterations, &t);
    free_classifier();
    free_run_state(&t.state);
    free(weights);
    return 0;
}
