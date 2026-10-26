/* OpenMP experiments for heterogeneous cores. No changes to the stay backend. */
#include <omp.h>
#include <errno.h>
#include <limits.h>

static int parallel_output_threads, parallel_output_chunk = 32;
static int parallel_fuse = 1, parallel_attention_threads, parallel_output_block = 4;

static int parallel_setting(const char *name, int fallback, int minimum) {
    const char *value = getenv(name);
    if (!value) return fallback;
    char *end;
    errno = 0;
    long n = strtol(value, &end, 10);
    if (errno || end == value || *end || n < minimum || n > INT_MAX) {
        fprintf(stderr, "Invalid %s\n", name);
        exit(EXIT_FAILURE);
    }
    return (int)n;
}

static void parallel_init(void) {
    parallel_output_threads = parallel_setting("OUTPUT_THREADS", omp_get_max_threads(), 1);
    parallel_output_chunk = parallel_setting("OUTPUT_CHUNK", 32, 1);
    parallel_fuse = parallel_setting("PARALLEL_FUSE", 1, 0);
    parallel_attention_threads = parallel_setting("PARALLEL_ATTENTION_THREADS", 0, 0);
    parallel_output_block = parallel_setting("PARALLEL_OUTPUT_BLOCK", 4, 1);
}

static void parallel_output(float *out, float *x, float *w, int n, int d) {
    if (parallel_output_block != 4) {
    #pragma omp parallel for num_threads(parallel_output_threads) schedule(dynamic, parallel_output_chunk)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        out[i] = val;
    }
    return;
    }
    #pragma omp parallel for num_threads(parallel_output_threads) schedule(dynamic, parallel_output_chunk)
    for (int block = 0; block < (d + 3) / 4; block++) {
        int i = block * 4;
        if (i + 3 < d) {
            float a = 0, b = 0, c = 0, e = 0;
            for (int j = 0; j < n; j++) {
                float v = x[j];
                a += w[i * n + j] * v;
                b += w[(i + 1) * n + j] * v;
                c += w[(i + 2) * n + j] * v;
                e += w[(i + 3) * n + j] * v;
            }
            out[i] = a; out[i+1] = b; out[i+2] = c; out[i+3] = e;
        } else {
            for (; i < d; i++) {
                float val = 0;
                for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
                out[i] = val;
            }
        }
    }
}

static void parallel_qkv(float *q, float *k, float *v, float *x,
                         float *wq, float *wk, float *wv, int dim, int kv_dim) {
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < dim + 2 * kv_dim; row++) {
        int i;
        float *out, *w;
        if (row < dim) { i = row; out = q; w = wq; }
        else if (row < dim + kv_dim) { i = row - dim; out = k; w = wk; }
        else { i = row - dim - kv_dim; out = v; w = wv; }
        float val = 0.0f;
        for (int j = 0; j < dim; j++) val += w[i * dim + j] * x[j];
        out[i] = val;
    }
}

static void parallel_ffn(float *out, float *x, float *up, float *gate, int dim, int hidden) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < hidden; i++) {
        float val = 0.0f, g = 0.0f;
        // Keep each row's accumulation order identical to matmul().
        for (int j = 0; j < dim; j++) val += up[i * dim + j] * x[j];
        for (int j = 0; j < dim; j++) g += gate[i * dim + j] * x[j];
        val *= 1.0f / (1.0f + expf(-val));
        out[i] = val * g;
    }
}
