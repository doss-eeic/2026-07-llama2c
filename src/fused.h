/* Independent output rows share one OpenMP work-sharing region.
 * Each dot product retains matmul's accumulation order; no weight repacking.
 */
typedef struct { float *q, *k, *v, *x, *wq, *wk, *wv; int dim, kv_dim; } QkvJob;

static void qkv_rows(void *ctx, int r0, int r1, int rank) {
    (void)rank;
    const QkvJob *job = ctx;
    int dim = job->dim, kv_dim = job->kv_dim;
    float *x = job->x;
    for (int row = r0; row < r1; row++) {
        int i;
        float *out, *w;
        if (row < dim) { i = row; out = job->q; w = job->wq; }
        else if (row < dim + kv_dim) { i = row - dim; out = job->k; w = job->wk; }
        else { i = row - dim - kv_dim; out = job->v; w = job->wv; }
        float val = 0.0f;
        for (int j = 0; j < dim; j++) val += w[i * dim + j] * x[j];
        out[i] = val;
    }
}

static void matmul_qkv(float *q, float *k, float *v, float *x,
                       float *wq, float *wk, float *wv, int dim, int kv_dim) {
    QkvJob job = { q, k, v, x, wq, wk, wv, dim, kv_dim };
    if (pool_on) { pool_run(qkv_rows, &job, dim + 2 * kv_dim); return; }
    #pragma omp parallel for schedule(runtime) num_threads(section_num_threads())
    for (int row = 0; row < dim + 2 * kv_dim; row++) { qkv_rows(&job, row, row + 1, 0); }
}

typedef struct { float *hb, *hb2, *x, *up, *gate; int dim; } UpGateJob;

static void up_gate_rows(void *ctx, int i0, int i1, int rank) {
    (void)rank;
    const UpGateJob *job = ctx;
    int dim = job->dim;
    float *x = job->x, *up = job->up, *gate = job->gate;
    for (int i = i0; i < i1; i++) {
        float val = 0.0f, g = 0.0f;
        for (int j = 0; j < dim; j++) val += up[i * dim + j] * x[j];
        for (int j = 0; j < dim; j++) g += gate[i * dim + j] * x[j];
        job->hb2[i] = g;
        val *= 1.0f / (1.0f + expf(-val));
        job->hb[i] = val * g;
    }
}

static void ffn_up_gate(float *hb, float *hb2, float *x, float *up, float *gate,
                        int dim, int hidden_dim) {
    UpGateJob job = { hb, hb2, x, up, gate, dim };
    if (pool_on) { pool_run(up_gate_rows, &job, hidden_dim); return; }
    #pragma omp parallel for schedule(runtime) num_threads(section_num_threads())
    for (int i = 0; i < hidden_dim; i++) { up_gate_rows(&job, i, i + 1, 0); }
}
