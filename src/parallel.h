/* Shared numerical kernels for OpenMP and the experimental persistent pool.
 * LLAMA_EXPERIMENT selects matched scheduling; legacy parallel/stay remain baselines.
 */
#ifndef LLAMA_STAY
#include <omp.h>
#endif
#include <errno.h>
#include <limits.h>

static int parallel_output_threads, parallel_output_chunk = 32;
static int parallel_fuse_qkv = 1, parallel_fuse_ffn = 1;
static int parallel_attention_threads, parallel_output_block = 1;
static int parallel_dynamic, parallel_output_dynamic = 1, parallel_layer_chunk = 16;

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
#ifdef LLAMA_STAY
    int threads = 1;  // The pool uses STAY_CPUS/STAY_LAYER_THREADS instead.
#else
    int threads = omp_get_max_threads();
#endif
    parallel_output_threads = parallel_setting("OUTPUT_THREADS", threads, 1);
    parallel_output_chunk = parallel_setting("OUTPUT_CHUNK", 32, 1);
    int fuse = parallel_setting("PARALLEL_FUSE", 1, 0);
    parallel_fuse_qkv = parallel_setting("PARALLEL_FUSE_QKV", fuse, 0);
    parallel_fuse_ffn = parallel_setting("PARALLEL_FUSE_FFN", fuse, 0);
    parallel_attention_threads = parallel_setting("PARALLEL_ATTENTION_THREADS", 0, 0);
    parallel_output_block = parallel_setting("PARALLEL_OUTPUT_BLOCK", 1, 1);
    parallel_dynamic = parallel_setting("PARALLEL_DYNAMIC", 0, 0);
    parallel_output_dynamic = parallel_setting("OUTPUT_DYNAMIC", 1, 0);
    parallel_layer_chunk = parallel_setting("PARALLEL_CHUNK", 16, 1);
    if (parallel_output_block != 1 && parallel_output_block != 4) {
        fprintf(stderr, "Invalid PARALLEL_OUTPUT_BLOCK (use 1 or 4)\n");
        exit(EXIT_FAILURE);
    }
}

#if defined(LLAMA_STAY) && defined(LLAMA_EXPERIMENT)
static int experiment_static_begin(int rows) {
    int n = stay_output_mode ? stay_count : stay_layer_count;
    if (stay_output_mode && stay_p_weight != 1) return stay_boundary(rows, stay_rank);
    int rem = rows % n;
    return stay_rank * (rows / n) + (stay_rank < rem ? stay_rank : rem);
}
static int experiment_static_end(int rows) {
    int n = stay_output_mode ? stay_count : stay_layer_count;
    if (stay_output_mode && stay_p_weight != 1) return stay_boundary(rows, stay_rank + 1);
    return experiment_static_begin(rows) + rows / n + (stay_rank < rows % n);
}
typedef struct { int begin, end, rows, chunk, dynamic; } ExperimentRange;
static ExperimentRange experiment_next(ExperimentRange range) {
    if (!range.dynamic) { range.begin = range.end; return range; }
    range.begin = atomic_fetch_add_explicit(&stay_row_cursor, range.chunk, memory_order_relaxed);
    range.end = range.begin + range.chunk;
    if (range.end > range.rows) range.end = range.rows;
    return range;
}
static ExperimentRange experiment_first(int rows, int chunk, int dynamic) {
    if (!stay_output_mode) {
        STAY_SERIAL atomic_store_explicit(&stay_row_cursor, 0, memory_order_relaxed);
        STAY_SYNC();
    }
    // Output's queue is reset by the coordinator before waking workers.
    ExperimentRange range = {experiment_static_begin(rows), experiment_static_end(rows), rows, chunk, dynamic};
    return dynamic ? experiment_next(range) : range;
}
#define PARALLEL_ROWS(i, rows) \
    for (ExperimentRange range = experiment_first((rows), parallel_layer_chunk, parallel_dynamic); \
         range.begin < range.end; range = experiment_next(range)) \
        for (int i = range.begin; i < range.end; i++)
#define PARALLEL_FINISH() STAY_SYNC()
#else
#define PARALLEL_ROWS(i, rows) \
    _Pragma("omp parallel for schedule(runtime)") \
    for (int i = 0; i < (rows); i++)
#define PARALLEL_FINISH() ((void)0)
#endif

static void parallel_matmul(float *out, float *x, float *w, int n, int d) {
    PARALLEL_ROWS(i, d) {
        float val = 0;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        out[i] = val;
    }
    PARALLEL_FINISH();
}

static void parallel_output_unit(float *out, float *x, float *w, int n, int d, int block) {
    int i = block * parallel_output_block;
    if (parallel_output_block == 4 && i + 3 < d) {
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
        int end = i + parallel_output_block;
        if (end > d) end = d;
        for (; i < end; i++) {
            float val = 0;
            for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
            out[i] = val;
        }
    }
}

static void parallel_output(float *out, float *x, float *w, int n, int d) {
    int units = (d + parallel_output_block - 1) / parallel_output_block;
    int chunk = parallel_output_chunk;
#ifdef LLAMA_EXPERIMENT
    // Hold physical row granularity constant when comparing 1-row and 4-row kernels.
    chunk = (chunk + parallel_output_block - 1) / parallel_output_block;
#endif
#if defined(LLAMA_STAY) && defined(LLAMA_EXPERIMENT)
    for (ExperimentRange range = experiment_first(units, chunk, parallel_output_dynamic);
         range.begin < range.end; range = experiment_next(range))
        for (int unit = range.begin; unit < range.end; unit++)
            parallel_output_unit(out, x, w, n, d, unit);
#else
    if (parallel_output_dynamic) {
        #pragma omp parallel for num_threads(parallel_output_threads) schedule(dynamic, chunk)
        for (int unit = 0; unit < units; unit++) parallel_output_unit(out, x, w, n, d, unit);
    } else {
        #pragma omp parallel for num_threads(parallel_output_threads) schedule(static)
        for (int unit = 0; unit < units; unit++) parallel_output_unit(out, x, w, n, d, unit);
    }
#endif
}

static void parallel_qkv(float *q, float *k, float *v, float *x,
                         float *wq, float *wk, float *wv, int dim, int kv_dim) {
    PARALLEL_ROWS(row, dim + 2 * kv_dim) {
        int i;
        float *out, *w;
        if (row < dim) { i = row; out = q; w = wq; }
        else if (row < dim + kv_dim) { i = row - dim; out = k; w = wk; }
        else { i = row - dim - kv_dim; out = v; w = wv; }
        float val = 0.0f;
        for (int j = 0; j < dim; j++) val += w[i * dim + j] * x[j];
        out[i] = val;
    }
    PARALLEL_FINISH();
}

static void parallel_ffn(float *out, float *x, float *up, float *gate, int dim, int hidden) {
    PARALLEL_ROWS(i, hidden) {
        float val = 0.0f, g = 0.0f;
        for (int j = 0; j < dim; j++) val += up[i * dim + j] * x[j];
        for (int j = 0; j < dim; j++) g += gate[i * dim + j] * x[j];
        val *= 1.0f / (1.0f + expf(-val));
        out[i] = val * g;
    }
    PARALLEL_FINISH();
}
