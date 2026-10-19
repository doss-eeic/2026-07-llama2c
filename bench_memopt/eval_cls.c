// 出力層を fp16 / int8 にしたときの精度の評価。
// 方針: プロジェクト/output-layer-memory-optimization.md 4.4 節（B1 / B2 の正しさの確認）
//
// 出力が fp32 と変わるので sha256 では確認できない。代わりに次の3つを出す。
//   [1] logits の誤差: fp32 で greedy 生成しながら、各位置で同じ x から fp16 / int8 の logits も計算し、
//       fp32 との最大誤差・平均誤差と、argmax（次のトークン）が一致した位置の数を数える
//   [2] greedy 生成の一致: fp16 / int8 だけで greedy 生成し、生成したトークンが fp32 と先頭から何個同じか
//   [3] SIMD 版とスカラー版の logits の最大差（[1] の表の simd_vs_scalar 列。SIMD のカーネルの確認）
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

static const char* cls_type_names[] = {"fp32", "fp16", "int8"};
static const char* cls_isa_names[] = {"scalar", "avx2", "avx-vnni"};

// 評価する形式（fp32 と比べる側）
static const ClsType eval_types[] = {CLS_FP16, CLS_INT8};
#define N_EVAL (int)(sizeof(eval_types) / sizeof(eval_types[0]))

typedef struct {
    double max_abs;     // fp32 との最大誤差
    double sum_abs;     // fp32 との誤差の和（平均誤差用）
    long n_vals;
    int agree;          // argmax が fp32 と一致した位置の数
    int n_pos;
    double max_simd;    // SIMD 版とスカラー版の最大差
} ErrStats;

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

// forward() の直後に呼ぶ。s->x は最後の rmsnorm を通った後の値なので、同じ x で各形式の logits を計算できる
static void measure_errors(Transformer* t, const float* ref, float* buf, float* buf_scalar, ErrStats* st) {
    int vocab = t->config.vocab_size;
    int ref_top = sample_argmax((float*)ref, vocab);
    for (int k = 0; k < N_EVAL; k++) {
        t->cls.type = eval_types[k];
        classifier_matmul(t, buf, t->state.x);
        for (int i = 0; i < vocab; i++) {
            double d = fabs((double)buf[i] - ref[i]);
            if (d > st[k].max_abs) { st[k].max_abs = d; }
            st[k].sum_abs += d;
        }
        st[k].n_vals += vocab;
        st[k].agree += sample_argmax(buf, vocab) == ref_top;
        st[k].n_pos++;
        if (t->cls.isa != CLS_ISA_SCALAR) {
            ClsIsa isa = t->cls.isa;
            t->cls.isa = CLS_ISA_SCALAR;
            classifier_matmul(t, buf_scalar, t->state.x);
            t->cls.isa = isa;
            double d = max_abs_diff(buf, buf_scalar, vocab);
            if (d > st[k].max_simd) { st[k].max_simd = d; }
        }
    }
    t->cls.type = CLS_FP32;
}

// run.c の generate() と同じ手順で greedy 生成する。seq[0] はプロンプトの先頭、seq[pos + 1] は pos での次のトークン。
// BOS（=1）が出たら止める。戻り値は seq の長さ。st が NULL でなければ（fp32 のとき）各位置で誤差を測る
static int run_greedy(Transformer* t, ClsType type, const int* prompt, int n_prompt, int steps,
                      int* seq, float* buf, float* buf_scalar, ErrStats* st, double* max_logit) {
    int vocab = t->config.vocab_size;
    int len = 1;
    seq[0] = prompt[0];
    for (int pos = 0; pos < steps; pos++) {
        t->cls.type = type;
        float* logits = forward(t, seq[pos], pos);
        t->cls.type = CLS_FP32;
        int next = pos < n_prompt - 1 ? prompt[pos + 1] : sample_argmax(logits, vocab);
        if (st != NULL) {
            for (int i = 0; i < vocab; i++) {
                if (fabs(logits[i]) > *max_logit) { *max_logit = fabs(logits[i]); }
            }
            measure_errors(t, logits, buf, buf_scalar, st);
        }
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
    for (int k = 0; k < N_EVAL; k++) { build_classifier(&t, eval_types[k]); }
    t.cls.type = CLS_FP32;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, t.config.vocab_size);

    int vocab = t.config.vocab_size;
    float* buf = malloc(vocab * sizeof(float));
    float* buf_scalar = malloc(vocab * sizeof(float));
    int* seq_ref = malloc((steps + 1) * sizeof(int));
    int* seq = malloc((steps + 1) * sizeof(int));
    if (!buf || !buf_scalar || !seq_ref || !seq) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }

    ErrStats st[N_EVAL];
    memset(st, 0, sizeof(st));
    double max_logit = 0.0;
    // [2] の結果: same[p][k] = 生成したトークンが fp32 と先頭から一致した数、gen[p] = fp32 で生成した数
    int same[MAX_PROMPTS][N_EVAL];
    int gen[MAX_PROMPTS];
    int n_full[N_EVAL] = {0};

    for (int p = 0; p < n_prompts; p++) {
        int* prompt_tokens = malloc((strlen(prompts[p]) + 3) * sizeof(int));
        int n_prompt = 0;
        encode(&tokenizer, prompts[p], 1, 0, prompt_tokens, &n_prompt);

        int len_ref = run_greedy(&t, CLS_FP32, prompt_tokens, n_prompt, steps,
                                 seq_ref, buf, buf_scalar, st, &max_logit);
        // 生成したトークンは seq[n_prompt] から
        gen[p] = len_ref - n_prompt;
        for (int k = 0; k < N_EVAL; k++) {
            int len = run_greedy(&t, eval_types[k], prompt_tokens, n_prompt, steps,
                                 seq, NULL, NULL, NULL, NULL);
            int i = n_prompt;
            while (i < len && i < len_ref && seq[i] == seq_ref[i]) { i++; }
            same[p][k] = i - n_prompt;
            n_full[k] += (len == len_ref && i == len);
        }
        free(prompt_tokens);
    }

    printf("model: %s (dim %d, vocab %d), steps: %d, prompts: %d\n",
           checkpoint_path, t.config.dim, vocab, steps, n_prompts);
    printf("\n[1] logits error vs fp32 (same x, along the fp32 greedy sequence; max |logit| = %.3f)\n", max_logit);
    printf("%-6s %-9s %10s %10s %15s %15s\n", "format", "kernel", "max_abs", "mean_abs", "argmax_agree", "simd_vs_scalar");
    for (int k = 0; k < N_EVAL; k++) {
        char agree[32], simd[32];
        snprintf(agree, sizeof(agree), "%d/%d", st[k].agree, st[k].n_pos);
        if (t.cls.isa == CLS_ISA_SCALAR) { snprintf(simd, sizeof(simd), "-"); }
        else { snprintf(simd, sizeof(simd), "%.3e", st[k].max_simd); }
        printf("%-6s %-9s %10.3e %10.3e %15s %15s\n", cls_type_names[eval_types[k]],
               kernel_name(eval_types[k], t.cls.isa), st[k].max_abs,
               st[k].sum_abs / (st[k].n_vals > 0 ? st[k].n_vals : 1), agree, simd);
    }

    printf("\n[2] greedy generation: generated tokens matching fp32 from the start\n");
    printf("%-40s %8s", "prompt", "fp32");
    for (int k = 0; k < N_EVAL; k++) { printf(" %12s", cls_type_names[eval_types[k]]); }
    printf("\n");
    for (int p = 0; p < n_prompts; p++) {
        char label[64];
        snprintf(label, sizeof(label), "\"%.36s\"", prompts[p]);
        printf("%-40s %8d", label, gen[p]);
        for (int k = 0; k < N_EVAL; k++) {
            char cell[32];
            snprintf(cell, sizeof(cell), "%d%s", same[p][k], same[p][k] == gen[p] ? " (all)" : "");
            printf(" %12s", cell);
        }
        printf("\n");
    }
    printf("%-40s %8s", "fully identical", "");
    for (int k = 0; k < N_EVAL; k++) {
        char cell[32];
        snprintf(cell, sizeof(cell), "%d/%d", n_full[k], n_prompts);
        printf(" %12s", cell);
    }
    printf("\n");

    free(buf);
    free(buf_scalar);
    free(seq_ref);
    free(seq);
    free_tokenizer(&tokenizer);
    free_transformer(&t);
    return 0;
}
