/* Independent output rows share one OpenMP work-sharing region.
 * Each dot product retains matmul's accumulation order; no weight repacking.
 */
static void matmul_qkv(float *q, float *k, float *v, float *x,
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

static void ffn_up_gate(float *hb, float *hb2, float *x, float *gate, float *up,
                        int dim, int hidden_dim) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < hidden_dim; i++) {
        float val = 0.0f, u = 0.0f;
        for (int j = 0; j < dim; j++) val += gate[i * dim + j] * x[j];
        for (int j = 0; j < dim; j++) u += up[i * dim + j] * x[j];
        hb2[i] = u;
        val *= 1.0f / (1.0f + expf(-val));
        hb[i] = val * u;
    }
}
