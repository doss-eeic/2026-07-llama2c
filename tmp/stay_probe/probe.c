/* Output-only switching: fixed team executes all layers in one pool job.
 * Original numerical kernels from run_detail.c.
 * Policy: -2 default <cpus> for the fixed layer team; -1 output <cpus>
 * (or -2 output <cpus>) for the classifier. No other stage overrides.
 * Workers remain pinned; each token dispatches one layer job and one output job.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <errno.h>
#include <limits.h>
#ifdef __linux__
#include <sched.h>
#endif
#define TESTING
#define forward reference_forward
#include "run_detail.c"
#undef forward
#undef TESTING

#define STAY_MAX_CPUS 1024
static const char *stay_stages[] = {
    "q_proj", "k_proj", "v_proj", "attention", "attn_out",
    "ffn_up", "ffn_gate", "ffn_down", "output", "default"
};
typedef void (*RowFunction)(void *, int);
typedef struct {
    int layer, stage, count, cpus[STAY_MAX_CPUS];
} StayRule;
typedef struct {
    pthread_t thread;
    pthread_cond_t wake;
    int cpu, ready, begin, end, actual_cpu;
    RowFunction function;
    void *context;
    unsigned long long jobs, rows;
    int execution_cpu;
    unsigned long long trace_dispatch;
    int trace_stage, trace_count;
    struct { unsigned long long dispatch, start, finish, cpu_ns; int stage, rows; } trace[1024];
} StayWorker;
static StayRule *stay_rules;
static int stay_rule_count, stay_worker_count, stay_pending, stay_stopping;
static int stay_started, stay_pin_failed;
static int stay_diagnostics;
static int probe_trace, probe_output_only;
static int stay_reported[10];
static StayWorker stay_workers[STAY_MAX_CPUS];
static pthread_mutex_t stay_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t stay_done = PTHREAD_COND_INITIALIZER;

static void stay_fail(const char *message) {
    fprintf(stderr, "run_stay2: %s\n", message);
    exit(EXIT_FAILURE);
}
static void stay_check(int error) {
    if (error) { errno = error; perror("run_stay pthread"); exit(EXIT_FAILURE); }
}
static void *stay_worker(void *argument) {
    StayWorker *worker = argument;
    int pin_failed = 0;
    worker->actual_cpu = -1;
#ifdef __linux__
    cpu_set_t mask, actual;
    CPU_ZERO(&mask);
    CPU_SET(worker->cpu, &mask);
    pin_failed = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (!pin_failed) {
        pin_failed = pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual);
        worker->actual_cpu = sched_getcpu();
        if (!pin_failed && (CPU_COUNT(&actual) != 1 ||
            !CPU_ISSET(worker->cpu, &actual) || worker->actual_cpu != worker->cpu)) pin_failed = 1;
    }
#elif !defined(STAY_TEST_UNPINNED)
    pin_failed = 1;
#endif
    stay_check(pthread_mutex_lock(&stay_mutex));
    stay_pin_failed |= !!pin_failed;
    stay_started++;
    stay_check(pthread_cond_signal(&stay_done));
    for (;;) {
        while (!worker->ready && !stay_stopping)
            stay_check(pthread_cond_wait(&worker->wake, &stay_mutex));
        if (stay_stopping) break;
        worker->ready = 0;
        stay_check(pthread_mutex_unlock(&stay_mutex));
        if (stay_diagnostics) {
#ifdef __linux__
            worker->execution_cpu = sched_getcpu();
            if (worker->execution_cpu != worker->cpu) stay_fail("worker execution CPU mismatch");
#endif
            worker->jobs++;
            worker->rows += worker->end - worker->begin;
        }
        unsigned long long probe_start = 0, probe_cpu_start = 0;
        struct timespec probe_cpu;
        if (probe_trace) {
            probe_start = profile_now();
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &probe_cpu);
            probe_cpu_start = (unsigned long long)probe_cpu.tv_sec * 1000000000ULL + probe_cpu.tv_nsec;
        }
        for (int i = worker->begin; i < worker->end; i++)
            worker->function(worker->context, i);
        if (probe_trace && worker->trace_count < 1024) {
            unsigned long long finish = profile_now();
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &probe_cpu);
            int n = worker->trace_count++;
            worker->trace[n].dispatch = worker->trace_dispatch;
            worker->trace[n].start = probe_start;
            worker->trace[n].finish = finish;
            worker->trace[n].cpu_ns = (unsigned long long)probe_cpu.tv_sec * 1000000000ULL + probe_cpu.tv_nsec - probe_cpu_start;
            worker->trace[n].stage = worker->trace_stage;
            worker->trace[n].rows = worker->end - worker->begin;
        }
        stay_check(pthread_mutex_lock(&stay_mutex));
        if (--stay_pending == 0) stay_check(pthread_cond_signal(&stay_done));
    }
    stay_check(pthread_mutex_unlock(&stay_mutex));
    return NULL;
}
static int stay_stage_id(const char *name) {
    for (int i = 0; i < 10; i++) if (!strcmp(name, stay_stages[i])) return i;
    stay_fail("unknown policy stage");
    return -1;
}
static StayWorker *stay_find_worker(int cpu) {
    for (int i = 0; i < stay_worker_count; i++)
        if (stay_workers[i].cpu == cpu) return &stay_workers[i];
    stay_fail("CPU missing from pool");
    return NULL;
}
static StayRule *stay_rule(int stage, int layer) {
    StayRule *selected = NULL;
    int priority = -1;
    for (int i = 0; i < stay_rule_count; i++) {
        StayRule *r = &stay_rules[i];
        int score = r->stage == 9 ? 0 : r->layer == -2 ? 1 : 2;
        if ((r->stage == 9 || r->stage == stage) &&
            (r->layer == -2 || r->layer == layer) && score > priority) {
            selected = r;
            priority = score;
        }
    }
    if (!selected) stay_fail("policy has no default");
    return selected;
}
static void stay_init(const char *path, int layers) {
    probe_trace = getenv("LLAMA_PROBE_TRACE") != NULL;
    probe_output_only = getenv("LLAMA_PROBE_OUTPUT_ONLY") != NULL;
    stay_diagnostics = getenv("LLAMA_STAY_DIAGNOSTICS") != NULL;
    if (!path) stay_fail("specify -c policy.txt or LLAMA_STAY_POLICY");
#if !defined(__linux__) && !defined(STAY_TEST_UNPINNED)
    stay_fail("exact CPU pinning requires Linux");
#endif
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); exit(EXIT_FAILURE); }
    size_t capacity = (size_t)(layers + 2) * 10;
    stay_rules = calloc(capacity, sizeof(*stay_rules));
    if (!stay_rules) stay_fail("policy allocation failed");
    char line[16384], name[64], cpus[8192], extra;
    int have_default = 0;
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) stay_fail("cannot read allowed CPUs");
#endif
    while (fgets(line, sizeof(line), file)) {
        char *first = line;
        while (isspace((unsigned char)*first)) first++;
        if (!*first || *first == '#') continue;
        if ((size_t)stay_rule_count >= capacity) stay_fail("too many rules");
        StayRule *r = &stay_rules[stay_rule_count];
        if (sscanf(first, "%d %63s %8191s %c", &r->layer, name, cpus, &extra) != 3)
            stay_fail("expected: layer stage cpu,cpu,...");
        r->stage = stay_stage_id(name);
        if (r->stage != 9 && r->stage != 8)
            stay_fail("only default and output policies are supported");
        if (r->layer < -2 || r->layer >= layers ||
            (r->stage == 9 && r->layer != -2) ||
            (r->stage == 8 && r->layer != -2 && r->layer != -1) ||
            (r->stage < 8 && r->layer == -1)) stay_fail("invalid layer/stage");
        for (int i = 0; i < stay_rule_count; i++)
            if (stay_rules[i].layer == r->layer && stay_rules[i].stage == r->stage)
                stay_fail("duplicate policy rule");
        have_default |= r->stage == 9;
        char *cursor = cpus;
        for (;;) {
            char *end;
            long cpu = strtol(cursor, &end, 10);
            if (end == cursor || cpu < 0 || cpu >= STAY_MAX_CPUS ||
                (*end && *end != ',') || r->count == STAY_MAX_CPUS) stay_fail("invalid CPU list");
#ifdef __linux__
            if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) stay_fail("CPU outside allowed affinity");
#endif
            for (int i = 0; i < r->count; i++)
                if (r->cpus[i] == cpu) stay_fail("duplicate CPU in rule");
            r->cpus[r->count++] = (int)cpu;
            int found = 0;
            for (int i = 0; i < stay_worker_count; i++) found |= stay_workers[i].cpu == cpu;
            if (!found) stay_workers[stay_worker_count++].cpu = (int)cpu;
            if (!*end) break;
            cursor = end + 1;
        }
        stay_rule_count++;
    }
    if (ferror(file)) stay_fail("policy read failed");
    fclose(file);
    if (!have_default) stay_fail("required rule: -2 default cpu,cpu,...");
#ifdef __linux__
    // Coordinator stays on the first default CPU and sleeps while workers run.
    cpu_set_t coordinator;
    CPU_ZERO(&coordinator);
    CPU_SET(stay_rule(9, -2)->cpus[0], &coordinator);
    if (sched_setaffinity(0, sizeof(coordinator), &coordinator)) stay_fail("coordinator pin failed");
#endif
    for (int i = 0; i < stay_worker_count; i++) {
        stay_check(pthread_cond_init(&stay_workers[i].wake, NULL));
        stay_check(pthread_create(&stay_workers[i].thread, NULL, stay_worker, &stay_workers[i]));
    }
    stay_check(pthread_mutex_lock(&stay_mutex));
    while (stay_started < stay_worker_count)
        stay_check(pthread_cond_wait(&stay_done, &stay_mutex));
    stay_check(pthread_mutex_unlock(&stay_mutex));
    if (stay_pin_failed) stay_fail("worker CPU pin/verification failed");
    for (int i = 0; i < stay_worker_count; i++)
        fprintf(stderr, "stay worker=%d allowed_cpu=%d actual_cpu=%d\n",
                i, stay_workers[i].cpu, stay_workers[i].actual_cpu);
}
static void stay_dispatch(const char *stage, int layer, int rows,
                          RowFunction function, void *context) {
    int stage_id = stay_stage_id(stage);
    StayRule *rule = stay_rule(stage_id, layer);
    int report = stay_diagnostics && !stay_reported[stage_id];
    stay_reported[stage_id] = 1;
    stay_check(pthread_mutex_lock(&stay_mutex));
    // Only participants are woken; all other CPU-pinned threads stay asleep.
    int probe_weight = getenv("LLAMA_PROBE_P_WEIGHT") ? atoi(getenv("LLAMA_PROBE_P_WEIGHT")) : 1;
    if (probe_weight < 1 || probe_weight > 8) stay_fail("invalid probe P weight");
    int probe_weights[STAY_MAX_CPUS], probe_total = 0, probe_prefix = 0;
    const char *probe_p_cpus = getenv("LLAMA_PROBE_P_CPUS");
    for (int i = 0; i < rule->count; i++) {
        char needle[32]; snprintf(needle, sizeof(needle), ",%d,", rule->cpus[i]);
        int is_p = probe_p_cpus && strstr(probe_p_cpus, needle);
        probe_weights[i] = stage_id == 8 && is_p ? probe_weight : 1;
        probe_total += probe_weights[i];
    }
    stay_pending = rule->count;
    unsigned long long probe_dispatch = probe_trace ? profile_now() : 0;
    for (int rank = 0; rank < rule->count; rank++) {
        StayWorker *worker = stay_find_worker(rule->cpus[rank]);
        worker->begin = (int)((long long)rows * probe_prefix / probe_total);
        probe_prefix += probe_weights[rank];
        worker->end = (int)((long long)rows * probe_prefix / probe_total);
        worker->function = function;
        worker->context = context;
        worker->trace_dispatch = probe_dispatch;
        worker->trace_stage = stage_id;
        worker->ready = 1;
        stay_check(pthread_cond_signal(&worker->wake));
    }
    while (stay_pending) stay_check(pthread_cond_wait(&stay_done, &stay_mutex));
    if (report) {
        for (int rank = 0; rank < rule->count; rank++) {
            StayWorker *worker = stay_find_worker(rule->cpus[rank]);
            fprintf(stderr, "stay dispatch stage=%s layer=%d participants=%d cpu=%d execution_cpu=%d begin=%d end=%d\n",
                    stage, layer, rule->count, worker->cpu, worker->execution_cpu,
                    worker->begin, worker->end);
        }
    }
    stay_check(pthread_mutex_unlock(&stay_mutex));
}
static void stay_shutdown(void) {
    stay_check(pthread_mutex_lock(&stay_mutex));
    stay_stopping = 1;
    for (int i = 0; i < stay_worker_count; i++) stay_check(pthread_cond_signal(&stay_workers[i].wake));
    stay_check(pthread_mutex_unlock(&stay_mutex));
    for (int i = 0; i < stay_worker_count; i++) {
        stay_check(pthread_join(stay_workers[i].thread, NULL));
        if (stay_diagnostics)
            fprintf(stderr, "stay work worker=%d cpu=%d execution_cpu=%d jobs=%llu rows=%llu\n",
                    i, stay_workers[i].cpu, stay_workers[i].execution_cpu,
                    stay_workers[i].jobs, stay_workers[i].rows);
        for (int n = 0; n < stay_workers[i].trace_count; n++) {
            StayWorker *w = &stay_workers[i];
            fprintf(stderr, "probe,%d,%s,%d,%llu,%llu,%llu,%llu\n", w->cpu,
                    stay_stages[w->trace[n].stage], w->trace[n].rows,
                    w->trace[n].dispatch, w->trace[n].start,
                    w->trace[n].finish, w->trace[n].cpu_ns);
        }
        stay_check(pthread_cond_destroy(&stay_workers[i].wake));
    }
    free(stay_rules);
}
typedef struct { float *out, *x, *weights; int n; } MatmulJob;
static void matmul_row(void *context, int i) {
    MatmulJob *job = context;
    float val = 0.0f;
    for (int j = 0; j < job->n; j++) val += job->weights[(size_t)i * job->n + j] * job->x[j];
    job->out[i] = val;
}
static void stay_matmul(const char *stage, int layer, float *out, float *x, float *w, int n, int d) {
    MatmulJob job = {out, x, w, n};
    stay_dispatch(stage, layer, d, matmul_row, &job);
}
// Validation runs can retain only forward_total to avoid per-operator timer overhead.
#ifdef LLAMA_PROFILE
static int stay_timing_only;
#undef PROFILE_START
#undef PROFILE_END
#define PROFILE_START(name) unsigned long long name = stay_timing_only ? 0 : profile_now()
#define PROFILE_END(stage, layer, start) do { \
    if (!stay_timing_only || !strcmp(stage, "forward_total")) \
        profile_record(stage, pos, layer, start); \
} while (0)
#endif

typedef struct { Transformer *t; int pos, layer; } AttentionJob;
static void attention_row(void *context, int h) {
    AttentionJob *job = context;
    Config *p = &job->t->config;
    RunState *s = &job->t->state;
    int pos = job->pos, head_size = p->dim / p->n_heads;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int loff = job->layer * p->seq_len * kv_dim;
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

// Detailed measurements synchronize stage boundaries. Normal builds only
// synchronize data dependencies inside team kernels.
#ifdef LLAMA_PROFILE
#define STAY_STAGE_SYNC() do { \
    if (!stay_timing_only) pthread_barrier_wait(&stay_layers_barrier); \
} while (0)
#else
#define STAY_STAGE_SYNC() ((void)0)
#endif
typedef struct { Transformer *transformer; int token, pos; } LayersJob;
static pthread_barrier_t stay_layers_barrier;
static int stay_layers_count;
static void stay_team_matmul(int rank, float *out, float *x, float *w, int n, int d) {
    pthread_barrier_wait(&stay_layers_barrier);
    MatmulJob job = {out, x, w, n};
    int begin = (int)((long long)d * rank / stay_layers_count);
    int end = (int)((long long)d * (rank + 1) / stay_layers_count);
    for (int i = begin; i < end; i++) matmul_row(&job, i);
    pthread_barrier_wait(&stay_layers_barrier);
}
static void stay_layers(void *context, int rank) {
    LayersJob *job = context;
    Transformer *transformer = job->transformer;
    int token = job->token, pos = job->pos;

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    PROFILE_START(stage_start);

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    if (rank == 0) memcpy(x, content_row, dim*sizeof(*x));

    STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("embedding", -1, stage_start); }

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        PROFILE_START(layer_start);
        PROFILE_START(attention_norm_start);
        // attention rmsnorm
        if (rank == 0) rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("attention_norm", l, attention_norm_start); }
        PROFILE_START(q_start);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        if (rank == 0) {
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        }
        pthread_barrier_wait(&stay_layers_barrier);

        // qkv matmuls for this position
        stay_team_matmul(rank, s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("q_proj", l, q_start); }
        PROFILE_START(k_start);
        stay_team_matmul(rank, s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("k_proj", l, k_start); }
        PROFILE_START(v_start);
        stay_team_matmul(rank, s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("v_proj", l, v_start); }
        PROFILE_START(rope_start);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        if (rank == 0) for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("rope", l, rope_start); }
        PROFILE_START(attention_start);

        AttentionJob attention_job = {transformer, pos, (int)l};
        pthread_barrier_wait(&stay_layers_barrier);
        for (int h = rank; h < p->n_heads; h += stay_layers_count)
            attention_row(&attention_job, h);
        pthread_barrier_wait(&stay_layers_barrier);

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("attention", l, attention_start); }
        PROFILE_START(attention_output_start);

        // final matmul to get the output of the attention
        stay_team_matmul(rank, s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        if (rank == 0) for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("attn_out", l, attention_output_start); }
        PROFILE_START(ffn_norm_start);

        // ffn rmsnorm
        if (rank == 0) rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("ffn_norm", l, ffn_norm_start); }
        PROFILE_START(ffn_up_start);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        stay_team_matmul(rank, s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("ffn_up", l, ffn_up_start); }
        PROFILE_START(ffn_gate_start);
        stay_team_matmul(rank, s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("ffn_gate", l, ffn_gate_start); }
        PROFILE_START(swiglu_start);

        // SwiGLU non-linearity
        if (rank == 0) for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("swiglu", l, swiglu_start); }
        PROFILE_START(ffn_down_start);

        // final matmul to get the output of the ffn
        stay_team_matmul(rank, s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        if (rank == 0) for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("ffn_down", l, ffn_down_start); }
        STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("layer_total", l, layer_start); }
    }

    PROFILE_START(final_norm_start);
    // final rmsnorm
    if (rank == 0) rmsnorm(x, x, w->rms_final_weight, dim);

    STAY_STAGE_SYNC();
    if (rank == 0) { PROFILE_END("final_norm", -1, final_norm_start); }
}

float* forward(Transformer *transformer, int token, int pos) {
    profile_reserve(transformer->config.n_layers, token);
#ifdef LLAMA_PROFILE
    unsigned long long forward_start = profile_now();
#endif
    LayersJob job = {transformer, token, pos};
    if (!probe_output_only) stay_dispatch("default", -2, stay_layers_count, stay_layers, &job);
    else memcpy(transformer->state.x, transformer->weights.token_embedding_table + token * transformer->config.dim,
                transformer->config.dim * sizeof(float));
    PROFILE_START(output_start);
    stay_matmul("output", -1, transformer->state.logits, transformer->state.x,
                transformer->weights.wcls, transformer->config.dim, transformer->config.vocab_size);
    PROFILE_END("output", -1, output_start);
    PROFILE_END("forward_total", -1, forward_start);
    return transformer->state.logits;
}

// Fixed input sequence for fair end-to-end comparisons, independent of sampling/EOS.
static void stay_replay(Transformer *transformer, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); exit(EXIT_FAILURE); }
    int *tokens = malloc((size_t)transformer->config.seq_len * sizeof(int));
    if (!tokens) stay_fail("replay allocation failed");
    int count = 0, token, status;
    while ((status = fscanf(file, "%d", &token)) == 1) {
        if (count == transformer->config.seq_len || token < 0 || token >= transformer->config.vocab_size)
            stay_fail("replay exceeds context or token outside vocabulary");
        tokens[count++] = token;
    }
    if (status != EOF || ferror(file) || !count) stay_fail("invalid or empty replay");
    fclose(file);
    for (int pos = 0; pos < count; pos++) {
        float *logits = forward(transformer, tokens[pos], pos);
        int best = 0;
        for (int i = 1; i < transformer->config.vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        profile_token(pos, tokens[pos], best);
    }
    free(tokens);
}
#ifndef STAY_TESTING
void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "  -c <path>   CPU policy: -2 default <cpus>, -1 output <cpus> (or LLAMA_STAY_POLICY)\n");
    fprintf(stderr, "  -r <path>   replay whitespace-separated input token IDs\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    const char *policy_path = getenv("LLAMA_STAY_POLICY");
    const char *replay_path = NULL;

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'c') { policy_path = argv[i + 1]; }
        else if (argv[i][1] == 'r') { replay_path = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    stay_init(policy_path, transformer.config.n_layers);
    stay_layers_count = stay_rule(9, -2)->count;
    stay_check(pthread_barrier_init(&stay_layers_barrier, NULL, stay_layers_count));
    generation_forward = forward;
#ifdef LLAMA_PROFILE
    stay_timing_only = getenv("LLAMA_STAY_TIMING_ONLY") != NULL;
#endif

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (replay_path) {
        stay_replay(&transformer, replay_path);
    } else if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    stay_shutdown();
    stay_check(pthread_barrier_destroy(&stay_layers_barrier));
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif
