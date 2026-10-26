/* Independent rows keep the original dot-product order. */
typedef struct {
    float *q, *k, *v, *x, *wq, *wk, *wv;
    int dim, kv_dim;
} QkvRows;

static void qkv_rows(int begin, int end, void* context) {
    QkvRows* c = context;
    for (int row = begin; row < end; row++) {
        int i;
        float *out, *w;
        if (row < c->dim) { i = row; out = c->q; w = c->wq; }
        else if (row < c->dim + c->kv_dim) { i = row - c->dim; out = c->k; w = c->wk; }
        else { i = row - c->dim - c->kv_dim; out = c->v; w = c->wv; }
        float val = 0.0f;
        for (int j = 0; j < c->dim; j++) val += w[i * c->dim + j] * c->x[j];
        out[i] = val;
    }
}

static void matmul_qkv(float *q, float *k, float *v, float *x,
                       float *wq, float *wk, float *wv, int dim, int kv_dim) {
    QkvRows context = {q, k, v, x, wq, wk, wv, dim, kv_dim};
    if (omp_pool_running) { omp_pool_rows(dim + 2 * kv_dim, qkv_rows, &context); return; }
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < dim + 2 * kv_dim; row++) qkv_rows(row, row + 1, &context);
}

typedef struct { float *hb, *hb2, *x, *up, *gate; int dim; } FfnRows;

static void ffn_rows(int begin, int end, void* context) {
    FfnRows* c = context;
    for (int i = begin; i < end; i++) {
        float val = 0.0f, g = 0.0f;
        for (int j = 0; j < c->dim; j++) val += c->up[i * c->dim + j] * c->x[j];
        for (int j = 0; j < c->dim; j++) g += c->gate[i * c->dim + j] * c->x[j];
        c->hb2[i] = g;
        val *= 1.0f / (1.0f + expf(-val));
        c->hb[i] = val * g;
    }
}

static void ffn_up_gate(float *hb, float *hb2, float *x, float *up, float *gate,
                        int dim, int hidden_dim) {
    FfnRows context = {hb, hb2, x, up, gate, dim};
    if (omp_pool_running) { omp_pool_rows(hidden_dim, ffn_rows, &context); return; }
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < hidden_dim; i++) ffn_rows(i, i + 1, &context);
}
