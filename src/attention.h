/* Attention heads use the same resident layer workers as the matmuls. */
typedef struct {
    Config* p;
    RunState* s;
    int pos, loff, kv_dim, kv_mul, head_size;
} AttentionRows;

static void attention_rows(int begin, int end, void* context) {
    AttentionRows* c = context;
    Config* p = c->p;
    RunState* s = c->s;
    int pos = c->pos, loff = c->loff, kv_dim = c->kv_dim, kv_mul = c->kv_mul, head_size = c->head_size;
    for (int h = begin; h < end; h++) {
        // get the query vector for this head
        float* q = s->q + h * head_size;
        // attention scores for this head
        float* att = s->att + h * p->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}

static void attention_run(Config* p, RunState* s, int pos, int loff, int kv_dim, int kv_mul, int head_size) {
    AttentionRows context = {p, s, pos, loff, kv_dim, kv_mul, head_size};
    if (omp_pool_running) { omp_pool_rows(p->n_heads, attention_rows, &context); return; }
    #pragma omp parallel for
    for (int h = 0; h < p->n_heads; h++) attention_rows(h, h + 1, &context);
}
