// 出力層を fp16 / int8 / int8-exact にしたときの精度の評価。
// 方針: プロジェクト/output-layer-memory-optimization.md 4.4 節（B1 / B2 / B4 の正しさの確認）
//
// fp16 / int8 は出力が fp32 と変わるので sha256 では確認できない。代わりに次を出す。
//   [1] logits の誤差: fp32 で greedy 生成しながら、各位置で同じ x から fp16 / int8 の logits も計算し、
//       fp32 との最大誤差・平均誤差と、argmax（次のトークン）が一致した位置の数を数える。
//       simd_vs_scalar 列は SIMD 版とスカラー版の logits の最大差（SIMD のカーネルの確認）
//   [2] greedy 生成の一致: 各形式だけで greedy 生成し、生成したトークンが fp32 と先頭から何個同じか
//   [3] int8-exact（B4）の確認: [1] と同じ位置で
//       - argmax が fp32 と一致するか
//       - 2段目で計算し直した行（候補）の数
//       - 候補の行の値が fp32 の matmul とビット単位で同じか、候補でない行が int8 の近似値のままか
//       - int8 の近似値と fp32 の差が、誤差の上限に収まっているか（上限を超えた行の数と、差 / 上限 の最大）
//
// ビルド（リポジトリ直下で）:
//   gcc -Ofast -fopenmp -march=native bench_memopt/eval_cls.c -lm -o eval_cls
// 使い方:
//   OMP_NUM_THREADS=4 ./eval_cls stories15M.bin [-n steps] [-z tokenizer.bin] [-i prompt]...
//   -i は複数回指定できる（省略時は下の default_prompts）

#define TESTING
#include "../run.c"

#define MAX_PROMPTS 32

static char* default_prompts[] = {
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the park",
    "",
};

static const char* cls_type_names[] = {"fp32", "fp16", "int8", "int8-exact"};
static const char* cls_isa_names[] = {"scalar", "avx2", "avx-vnni"};

// [1] で誤差を測る形式と、[2] で greedy 生成を比べる形式
static const ClsType approx_types[] = {CLS_FP16, CLS_INT8};
#define N_APPROX (int)(sizeof(approx_types) / sizeof(approx_types[0]))
static const ClsType gen_types[] = {CLS_FP16, CLS_INT8, CLS_INT8_EXACT};
#define N_GEN (int)(sizeof(gen_types) / sizeof(gen_types[0]))

typedef struct {
    double max_abs;     // fp32 との最大誤差
    double sum_abs;     // fp32 との誤差の和（平均誤差用）
    long n_vals;
    int agree;          // argmax が fp32 と一致した位置の数
    int n_pos;
    double max_simd;    // SIMD 版とスカラー版の最大差
} ErrStats;

typedef struct {
    int agree, n_pos;       // argmax が fp32 と一致した位置の数
    long long n_cand;       // 候補の行数の合計
    int max_cand;           // 1 位置での候補の行数の最大
    long long cand_diff;    // 候補の行で、fp32 の matmul とビットが違った数
    long long other_diff;   // 候補でない行で、int8 の近似値から変わった数
    long long violations;   // |fp32 - int8| が上限を超えた行の数
    long long n_rows;
    double max_ratio;       // |fp32 - int8| / 上限 の最大
} ExactStats;

// 実際に使われるカーネル（fp16 には VNNI 版がないので AVX2 版になる）
static const char* kernel_name(ClsType type, ClsIsa isa) {
    if (isa == CLS_ISA_AVXVNNI && type == CLS_FP16) { return cls_isa_names[CLS_ISA_AVX2]; }
    return cls_isa_names[isa];
}

static double max_abs_diff(const float* a, const float* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > m) { m = d; }
    }
    return m;
}

static int same_bits(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

typedef struct {
    float* buf;         // 評価する形式の logits
    float* buf_scalar;  // スカラー版の logits
    float* buf_int8;    // int8 の近似値（int8-exact の 1 段目と同じ値）
    ErrStats err[N_APPROX];
    ExactStats ex;
    double max_logit;
} Eval;

// int8-exact の確認。ref は fp32 の logits、e->buf_int8 は同じ x での int8 の近似値
static void check_exact(Transformer* t, const float* ref, int ref_top, Eval* e) {
    Classifier* c = &t->cls;
    int vocab = t->config.vocab_size;
    ExactStats* st = &e->ex;
    c->type = CLS_INT8_EXACT;
    classifier_matmul(t, e->buf, t->state.x);
    c->type = CLS_FP32;

    // cls_rescore() と同じ基準で候補を数え直す
    int n_blocks = (vocab + CLS_BLOCK - 1) / CLS_BLOCK;
    float lo = c->blk_lo[0];
    for (int b = 1; b < n_blocks; b++) { lo = fmaxf(lo, c->blk_lo[b]); }
    int n_cand = 0;
    for (int i = 0; i < vocab; i++) {
        float approx = e->buf_int8[i];
        if (approx + c->bound[i] >= lo) {
            n_cand++;
            st->cand_diff += !same_bits(e->buf[i], ref[i]);
        } else {
            st->other_diff += !same_bits(e->buf[i], approx);
        }
        double diff = fabs((double)ref[i] - approx);
        if (diff > c->bound[i]) { st->violations++; }
        if (c->bound[i] > 0.0f && diff / c->bound[i] > st->max_ratio) { st->max_ratio = diff / c->bound[i]; }
    }
    st->n_rows += vocab;
    st->n_cand += n_cand;
    if (n_cand > st->max_cand) { st->max_cand = n_cand; }
    st->agree += sample_argmax(e->buf, vocab) == ref_top;
    st->n_pos++;
}

// forward() の直後に呼ぶ。s->x は最後の rmsnorm を通った後の値なので、同じ x で各形式の logits を計算できる
static void measure_errors(Transformer* t, const float* ref, Eval* e) {
    int vocab = t->config.vocab_size;
    int ref_top = sample_argmax((float*)ref, vocab);
    for (int i = 0; i < vocab; i++) {
        if (fabs(ref[i]) > e->max_logit) { e->max_logit = fabs(ref[i]); }
    }
    for (int k = 0; k < N_APPROX; k++) {
        ErrStats* st = &e->err[k];
        t->cls.type = approx_types[k];
        classifier_matmul(t, e->buf, t->state.x);
        for (int i = 0; i < vocab; i++) {
            double d = fabs((double)e->buf[i] - ref[i]);
            if (d > st->max_abs) { st->max_abs = d; }
            st->sum_abs += d;
        }
        st->n_vals += vocab;
        st->agree += sample_argmax(e->buf, vocab) == ref_top;
        st->n_pos++;
        if (approx_types[k] == CLS_INT8) { memcpy(e->buf_int8, e->buf, vocab * sizeof(float)); }
        if (t->cls.isa != CLS_ISA_SCALAR) {
            ClsIsa isa = t->cls.isa;
            t->cls.isa = CLS_ISA_SCALAR;
            classifier_matmul(t, e->buf_scalar, t->state.x);
            t->cls.isa = isa;
            double d = max_abs_diff(e->buf, e->buf_scalar, vocab);
            if (d > st->max_simd) { st->max_simd = d; }
        }
    }
    t->cls.type = CLS_FP32;
    check_exact(t, ref, ref_top, e);
}

// run.c の generate() と同じ手順で greedy 生成する。seq[0] はプロンプトの先頭、seq[pos + 1] は pos での次のトークン。
// BOS（=1）が出たら止める。戻り値は seq の長さ。e が NULL でなければ（fp32 のとき）各位置で誤差を測る
static int run_greedy(Transformer* t, ClsType type, const int* prompt, int n_prompt, int steps, int* seq, Eval* e) {
    int vocab = t->config.vocab_size;
    int len = 1;
    seq[0] = prompt[0];
    for (int pos = 0; pos < steps; pos++) {
        t->cls.type = type;
        float* logits = forward(t, seq[pos], pos);
        t->cls.type = CLS_FP32;
        int next = pos < n_prompt - 1 ? prompt[pos + 1] : sample_argmax(logits, vocab);
        if (e != NULL) { measure_errors(t, logits, e); }
        seq[len++] = next;
        if (next == 1) { break; }
    }
    return len;
}

static void usage(void) {
    fprintf(stderr, "Usage: eval_cls <checkpoint> [-n steps] [-z tokenizer.bin] [-i prompt]...\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    char* checkpoint_path = NULL;
    char* tokenizer_path = "tokenizer.bin";
    int steps = 256;
    char* prompts[MAX_PROMPTS];
    int n_prompts = 0;

    if (argc < 2) { usage(); }
    checkpoint_path = argv[1];
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) { usage(); }
        if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'i') {
            if (n_prompts == MAX_PROMPTS) { fprintf(stderr, "too many prompts (max %d)\n", MAX_PROMPTS); exit(EXIT_FAILURE); }
            prompts[n_prompts++] = argv[i + 1];
        }
        else { usage(); }
    }
    if (n_prompts == 0) {
        n_prompts = (int)(sizeof(default_prompts) / sizeof(default_prompts[0]));
        for (int i = 0; i < n_prompts; i++) { prompts[i] = default_prompts[i]; }
    }

    Transformer t;
    build_transformer(&t, checkpoint_path);
    if (steps <= 0 || steps > t.config.seq_len) { steps = t.config.seq_len; }
    for (int k = 0; k < N_GEN; k++) { build_classifier(&t, gen_types[k]); }
    t.cls.type = CLS_FP32;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, t.config.vocab_size);

    int vocab = t.config.vocab_size;
    Eval e;
    memset(&e, 0, sizeof(e));
    e.buf = malloc(vocab * sizeof(float));
    e.buf_scalar = malloc(vocab * sizeof(float));
    e.buf_int8 = malloc(vocab * sizeof(float));
    int* seq_ref = malloc((steps + 1) * sizeof(int));
    int* seq = malloc((steps + 1) * sizeof(int));
    if (!e.buf || !e.buf_scalar || !e.buf_int8 || !seq_ref || !seq) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }

    // [2] の結果: same[p][k] = 生成したトークンが fp32 と先頭から一致した数、gen[p] = fp32 で生成した数
    int same[MAX_PROMPTS][N_GEN];
    int gen[MAX_PROMPTS];
    int n_full[N_GEN] = {0};

    for (int p = 0; p < n_prompts; p++) {
        int* prompt_tokens = malloc((strlen(prompts[p]) + 3) * sizeof(int));
        int n_prompt = 0;
        encode(&tokenizer, prompts[p], 1, 0, prompt_tokens, &n_prompt);

        int len_ref = run_greedy(&t, CLS_FP32, prompt_tokens, n_prompt, steps, seq_ref, &e);
        // 生成したトークンは seq[n_prompt] から
        gen[p] = len_ref - n_prompt;
        for (int k = 0; k < N_GEN; k++) {
            int len = run_greedy(&t, gen_types[k], prompt_tokens, n_prompt, steps, seq, NULL);
            int i = n_prompt;
            while (i < len && i < len_ref && seq[i] == seq_ref[i]) { i++; }
            same[p][k] = i - n_prompt;
            n_full[k] += (len == len_ref && i == len);
        }
        free(prompt_tokens);
    }

    printf("model: %s (dim %d, vocab %d), steps: %d, prompts: %d\n",
           checkpoint_path, t.config.dim, vocab, steps, n_prompts);
    printf("\n[1] logits error vs fp32 (same x, along the fp32 greedy sequence; max |logit| = %.3f)\n", e.max_logit);
    printf("%-6s %-9s %10s %10s %15s %15s\n", "format", "kernel", "max_abs", "mean_abs", "argmax_agree", "simd_vs_scalar");
    for (int k = 0; k < N_APPROX; k++) {
        ErrStats* st = &e.err[k];
        char agree[32], simd[32];
        snprintf(agree, sizeof(agree), "%d/%d", st->agree, st->n_pos);
        if (t.cls.isa == CLS_ISA_SCALAR) { snprintf(simd, sizeof(simd), "-"); }
        else { snprintf(simd, sizeof(simd), "%.3e", st->max_simd); }
        printf("%-6s %-9s %10.3e %10.3e %15s %15s\n", cls_type_names[approx_types[k]],
               kernel_name(approx_types[k], t.cls.isa), st->max_abs,
               st->sum_abs / (st->n_vals > 0 ? st->n_vals : 1), agree, simd);
    }

    printf("\n[2] greedy generation: generated tokens matching fp32 from the start\n");
    printf("%-40s %8s", "prompt", "fp32");
    for (int k = 0; k < N_GEN; k++) { printf(" %12s", cls_type_names[gen_types[k]]); }
    printf("\n");
    for (int p = 0; p < n_prompts; p++) {
        char label[64];
        snprintf(label, sizeof(label), "\"%.36s\"", prompts[p]);
        printf("%-40s %8d", label, gen[p]);
        for (int k = 0; k < N_GEN; k++) {
            char cell[32];
            snprintf(cell, sizeof(cell), "%d%s", same[p][k], same[p][k] == gen[p] ? " (all)" : "");
            printf(" %12s", cell);
        }
        printf("\n");
    }
    printf("%-40s %8s", "fully identical", "");
    for (int k = 0; k < N_GEN; k++) {
        char cell[32];
        snprintf(cell, sizeof(cell), "%d/%d", n_full[k], n_prompts);
        printf(" %12s", cell);
    }
    printf("\n");

    ExactStats* ex = &e.ex;
    printf("\n[3] int8-exact (two-stage exact argmax, kernel %s)\n", kernel_name(CLS_INT8_EXACT, t.cls.isa));
    printf("argmax == fp32 argmax:            %d/%d\n", ex->agree, ex->n_pos);
    printf("candidates per position:          mean %.2f, max %d (of %d)\n",
           ex->n_pos > 0 ? (double)ex->n_cand / ex->n_pos : 0.0, ex->max_cand, vocab);
    printf("candidate rows != fp32 (bits):    %lld of %lld\n", ex->cand_diff, ex->n_cand);
    printf("other rows != int8 approx (bits): %lld of %lld\n", ex->other_diff, ex->n_rows - ex->n_cand);
    printf("bound violations:                 %lld of %lld rows (max |fp32 - int8| / bound = %.3f)\n",
           ex->violations, ex->n_rows, ex->max_ratio);
    int ok = ex->agree == ex->n_pos && ex->cand_diff == 0 && ex->other_diff == 0 && ex->violations == 0;
    printf("result:                           %s\n", ok ? "OK" : "FAILED");

    free(e.buf);
    free(e.buf_scalar);
    free(e.buf_int8);
    free(seq_ref);
    free(seq);
    free_tokenizer(&tokenizer);
    free_transformer(&t);
    return ok ? 0 : 1;
}
