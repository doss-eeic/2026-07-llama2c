/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// x86 の GCC / clang では、出力層の SIMD カーネルを target 属性で作り、実行時に選ぶ。
// それ以外（MSVC、ARM など）はスカラー版だけになる
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    #define CLS_X86 1
    #include <immintrin.h>
    #include <cpuid.h>
    // AVX-VNNI の intrinsics（_mm256_dpbusd_avx_epi32）は GCC 11 / clang 12 から
    #if defined(__clang__)
        #if !defined(__apple_build_version__) && __clang_major__ >= 12
            #define CLS_VNNI 1
        #endif
    #elif __GNUC__ >= 11
        #define CLS_VNNI 1
    #endif
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif

// 層の matmul / attention で計算するスレッド数 (-L)。0なら全スレッドを使う。
static int g_layer_threads = 0;

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

// 出力層（classifier）の重みの形式。-q で選ぶ。
// CLS_INT8_EXACT は int8 で全行を近似し、argmax になりうる行だけを fp32 で計算し直す（greedy 専用）
typedef enum { CLS_FP32, CLS_FP16, CLS_INT8, CLS_INT8_EXACT } ClsType;
#define CLS_GROUP 32 // int8 のスケールを共有する要素数
#define CLS_BLOCK 64 // 出力層の並列化の単位の行数（fp16 の 4 行同時処理に合わせて 4 の倍数）
#define CLS_BOUND_MARGIN (1.0f + 1.0f / 64) // CLS_INT8_EXACT の誤差の上限に掛ける余裕（丸めの分）
// 出力層のカーネルで使う命令セット。起動時に CPU を調べて選ぶ
typedef enum { CLS_ISA_SCALAR, CLS_ISA_AVX2, CLS_ISA_AVXVNNI } ClsIsa;

typedef struct {
    ClsType type;  // 出力層で使う形式（CLS_FP32 なら wcls をそのまま使う）
    ClsIsa isa;    // fp16 / int8 のカーネルの命令セット
    uint16_t* w16; // fp16 の wcls (vocab_size, dim)
    int8_t* q8;    // int8 の wcls (vocab_size, dim)
    float* q8s;    // int8 の wcls のスケール (vocab_size, dim / CLS_GROUP)
    int8_t* xq;    // int8 に量子化した x (dim,)
    float* xs;     // x のスケール (dim / CLS_GROUP,)
    // 以下は CLS_INT8_EXACT 用
    float* xk;     // 誤差の上限の、x のグループごとの係数 (dim / CLS_GROUP,)
    float* bound;  // 行ごとの int8 の近似値の誤差の上限 (vocab_size,)
    float* blk_lo; // 並列化の単位ごとの (近似値 - 上限) の最大 (n_blocks,)
    int* blk_cand; // 並列化の単位ごとの候補の行数 (n_blocks,)
    long long n_calls, n_cand; // 呼び出し回数と候補の行数の合計（統計用）
    int max_cand;              // 1 回の呼び出しでの候補の行数の最大（統計用）
} Classifier;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    Classifier cls; // reduced-precision copies of wcls for the output layer
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    // the output layer uses wcls as is (fp32) until build_classifier() is called
    memset(&t->cls, 0, sizeof(t->cls));
}

void free_classifier(Classifier* c) {
    free(c->w16);
    free(c->q8);
    free(c->q8s);
    free(c->xq);
    free(c->xs);
    free(c->xk);
    free(c->bound);
    free(c->blk_lo);
    free(c->blk_cand);
    memset(c, 0, sizeof(*c));
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
    free_classifier(&t->cls);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


// 並列区間の中で呼び、このスレッドが担当する [lo, hi) を返す。
// 先頭 nt スレッドに均等に分け、それ以外のスレッドは空（何もせず区間の終わりで待つ）
static inline void thread_share(int n, int nt, int* lo, int*hi) {
    #ifdef _OPENMP
        int t = omp_get_thread_num(), T = omp_get_num_threads();
    #else
        int t = 0, T = 1;
    #endif
        if (nt <= 0 || nt > T) { nt = T; }
        if (t >= nt) { *lo = *hi = 0; return; }
        *lo = (int)((long long)n * t / nt);
        *hi = (int)((long long)n * (t + 1) / nt);
}

static void matmul_nt(float* xout, float* x, float* w, int n, int d, int nt) {
    #pragma omp parallel
    {
        int lo, hi;
        thread_share(d, nt, &lo, &hi);
        for (int i = lo; i < hi; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }
}


void matmul(float* xout, float* x, float* w, int n, int d) {
    matmul_nt(xout, x, w, n, d, g_layer_threads);
}

// ----------------------------------------------------------------------------
// Output layer (classifier) in fp16 / int8
// 方針: プロジェクト/output-layer-memory-optimization.md 第4節（B1, B2）
//
// 出力層は 1 トークンごとに wcls（vocab_size x dim）を丸ごと読むので、メモリ帯域で律速する。
// 読み込み時に wcls の fp16 / int8 のコピーを作り、読む量を 1/2、約 1/3.5 に減らす。
// 埋め込み表としては fp32 の元データを使い続ける（1 トークンにつき 1 行しか読まないため）。

// fp32 -> fp16（最近接偶数への丸め）。重みの変換に使う
static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint16_t sign = (x >> 16) & 0x8000;
    uint32_t ax = x & 0x7fffffff;
    if (ax >= 0x7f800000) { return sign | 0x7c00 | (ax > 0x7f800000 ? 0x200 : 0); } // inf, nan
    if (ax >= 0x477ff000) { return sign | 0x7c00; } // 65520 以上は inf に丸まる
    if (ax < 0x38800000) {
        // fp16 では非正規化数になる（2^-14 未満）: 2^-24 単位に丸める
        float a;
        memcpy(&a, &ax, sizeof(a));
        return sign | (uint16_t)lrintf(a * 16777216.0f);
    }
    // 指数のバイアスを 127 から 15 に付け替え、仮数の下位 13 ビットを最近接偶数に丸める
    ax += 0xc8000fff + ((ax >> 13) & 1);
    return sign | (uint16_t)(ax >> 13);
}

// fp16 -> fp32（誤差なし）。スカラー版のカーネルで使う
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    if (exp == 0) { // 0 または非正規化数
        float f = ldexpf((float)man, -24);
        return sign ? -f : f;
    }
    uint32_t x = sign | (man << 13);
    x |= (exp == 0x1f) ? 0x7f800000 : (exp + 112) << 23; // inf, nan / 正規化数
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// CLS_GROUP 要素を int8 に量子化する（絶対値の最大が 127 になるスケール。runq.c の Q8_0 と同じ方式）
static void cls_quantize_group(int8_t* q, float* scale, const float* x) {
    float amax = 0.0f;
    for (int j = 0; j < CLS_GROUP; j++) { amax = fmaxf(amax, fabsf(x[j])); }
    float s = amax / 127.0f;
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int j = 0; j < CLS_GROUP; j++) { q[j] = (int8_t)lrintf(x[j] * inv); }
    *scale = s;
}

// カーネルは行 [r0, r1) の logits を計算する。int8 は x の代わりに量子化済みの c->xq, c->xs を使う。
// CLS_INT8_EXACT のときは、各行の誤差の上限も c->bound に書く
typedef void (*ClsRowsFn)(float* out, const float* x, const Classifier* c, int n, int r0, int r1);

// int8 の近似値の誤差の上限: Σ_g (重みのスケール) x (x の係数)。導出は classifier_matmul() の上のコメント
static inline float cls_row_bound(const float* rs, const float* xk, int gpr) {
    float b = 0.0f;
    for (int g = 0; g < gpr; g++) { b += rs[g] * xk[g]; }
    return b;
}

// スカラー版はどのマシンでも動く。SIMD 版の正しさの基準にもする
static void cls_rows_fp16_scalar(float* out, const float* x, const Classifier* c, int n, int r0, int r1) {
    for (int i = r0; i < r1; i++) {
        const uint16_t* row = c->w16 + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) { val += fp16_to_fp32(row[j]) * x[j]; }
        out[i] = val;
    }
}

static void cls_rows_int8_scalar(float* out, const float* x, const Classifier* c, int n, int r0, int r1) {
    (void)x;
    int gpr = n / CLS_GROUP;
    float* bound = c->type == CLS_INT8_EXACT ? c->bound : NULL;
    for (int i = r0; i < r1; i++) {
        const int8_t* row = c->q8 + (size_t)i * n;
        const float* rs = c->q8s + (size_t)i * gpr;
        float val = 0.0f;
        for (int g = 0; g < gpr; g++) {
            int32_t isum = 0;
            for (int j = 0; j < CLS_GROUP; j++) {
                isum += (int32_t)row[g * CLS_GROUP + j] * c->xq[g * CLS_GROUP + j];
            }
            val += (float)isum * (rs[g] * c->xs[g]);
        }
        out[i] = val;
        if (bound) { bound[i] = cls_row_bound(rs, c->xk, gpr); }
    }
}

#ifdef CLS_X86
#define CLS_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))

CLS_TARGET_AVX2 static inline float cls_hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// fp16 の重みを F16C で fp32 に変換して FMA する。4 行を同時に計算し、x の読み込みを共有する
CLS_TARGET_AVX2 static void cls_rows_fp16_avx2(float* out, const float* x, const Classifier* c, int n, int r0, int r1) {
    int n8 = n & ~7;
    for (int i = r0; i < r1; i += 4) {
        int nr = r1 - i < 4 ? r1 - i : 4;
        const uint16_t* row = c->w16 + (size_t)i * n;
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int j = 0; j < n8; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            for (int r = 0; r < nr; r++) {
                __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + (size_t)r * n + j)));
                acc[r] = _mm256_fmadd_ps(wv, xv, acc[r]);
            }
        }
        for (int r = 0; r < nr; r++) {
            float val = cls_hsum256(acc[r]);
            for (int j = n8; j < n; j++) { val += fp16_to_fp32(row[(size_t)r * n + j]) * x[j]; }
            out[i + r] = val;
        }
    }
}

// 符号付き x 符号付きの積和を、|w| x sign(x, w) の符号なし x 符号付きに直して maddubs で計算する。
// 量子化した値は |w|, |x| <= 127 なので、16 ビットの途中の和は飽和しない
CLS_TARGET_AVX2 static inline __m256i cls_dot_i8_avx2(__m256i w, __m256i x) {
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

CLS_TARGET_AVX2 static void cls_rows_int8_avx2(float* out, const float* x, const Classifier* c, int n, int r0, int r1) {
    (void)x;
    int gpr = n / CLS_GROUP;
    float* bound = c->type == CLS_INT8_EXACT ? c->bound : NULL;
    for (int i = r0; i < r1; i++) {
        const int8_t* row = c->q8 + (size_t)i * n;
        const float* rs = c->q8s + (size_t)i * gpr;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < gpr; g++) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(row + g * CLS_GROUP));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(c->xq + g * CLS_GROUP));
            __m256 p = _mm256_cvtepi32_ps(cls_dot_i8_avx2(wv, xv));
            acc = _mm256_fmadd_ps(p, _mm256_set1_ps(rs[g] * c->xs[g]), acc);
        }
        out[i] = cls_hsum256(acc);
        if (bound) { bound[i] = cls_row_bound(rs, c->xk, gpr); }
    }
}

#ifdef CLS_VNNI
#define CLS_TARGET_VNNI __attribute__((target("avx2,fma,f16c,avxvnni")))

// AVX2 版と同じ計算を、maddubs + madd の代わりに vpdpbusd 1 命令で行う
CLS_TARGET_VNNI static void cls_rows_int8_vnni(float* out, const float* x, const Classifier* c, int n, int r0, int r1) {
    (void)x;
    int gpr = n / CLS_GROUP;
    float* bound = c->type == CLS_INT8_EXACT ? c->bound : NULL;
    for (int i = r0; i < r1; i++) {
        const int8_t* row = c->q8 + (size_t)i * n;
        const float* rs = c->q8s + (size_t)i * gpr;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < gpr; g++) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(row + g * CLS_GROUP));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(c->xq + g * CLS_GROUP));
            __m256i dot = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),
                                                  _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(rs[g] * c->xs[g]), acc);
        }
        out[i] = cls_hsum256(acc);
        if (bound) { bound[i] = cls_row_bound(rs, c->xk, gpr); }
    }
}
#endif // CLS_VNNI
#endif // CLS_X86

// 実行中の CPU と OS が使える命令セットを調べる（-march=native なしのビルドでも SIMD 版を使うため）
static ClsIsa cls_detect_isa(void) {
#ifdef CLS_X86
    unsigned int a, b, c, d;
    if (__get_cpuid_max(0, NULL) < 7) { return CLS_ISA_SCALAR; }
    __cpuid(1, a, b, c, d);
    int fma = (c >> 12) & 1, osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, f16c = (c >> 29) & 1;
    if (!(fma && osxsave && avx && f16c)) { return CLS_ISA_SCALAR; }
    // OS が YMM レジスタを退避・復帰するか（XCR0 の bit 1, 2）
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    if ((xcr0_lo & 6) != 6) { return CLS_ISA_SCALAR; }
    __cpuid_count(7, 0, a, b, c, d);
    if (!((b >> 5) & 1)) { return CLS_ISA_SCALAR; } // AVX2
#ifdef CLS_VNNI
    if (a >= 1) { // leaf 7 の subleaf 1 があれば、その EAX bit 4 が AVX-VNNI
        __cpuid_count(7, 1, a, b, c, d);
        if ((a >> 4) & 1) { return CLS_ISA_AVXVNNI; }
    }
#endif
    return CLS_ISA_AVX2;
#else
    return CLS_ISA_SCALAR;
#endif
}

// CLS_INT8 と CLS_INT8_EXACT は同じ int8 のカーネルを使う
static ClsRowsFn cls_rows_fn(ClsType type, ClsIsa isa) {
#ifdef CLS_X86
    if (type == CLS_FP16 && isa != CLS_ISA_SCALAR) { return cls_rows_fp16_avx2; }
#ifdef CLS_VNNI
    if (type != CLS_FP16 && isa == CLS_ISA_AVXVNNI) { return cls_rows_int8_vnni; }
#endif
    if (type != CLS_FP16 && isa != CLS_ISA_SCALAR) { return cls_rows_int8_avx2; }
#endif
    (void)isa;
    return type == CLS_FP16 ? cls_rows_fp16_scalar : cls_rows_int8_scalar;
}

// 出力層で使う形式を選び、必要なら wcls の fp16 / int8 のコピーを作る。
// 作ったコピーは free_transformer() まで残るので、両方作っておけば c->type の書き換えだけで切り替えられる
void build_classifier(Transformer* t, ClsType type) {
    Config* p = &t->config;
    Classifier* c = &t->cls;
    const float* w = t->weights.wcls;
    size_t n = (size_t)p->vocab_size * p->dim;
    c->isa = cls_detect_isa();
    if (type == CLS_FP16 && c->w16 == NULL) {
        c->w16 = malloc(n * sizeof(uint16_t));
        if (!c->w16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        for (size_t i = 0; i < n; i++) { c->w16[i] = fp32_to_fp16(w[i]); }
    }
    if ((type == CLS_INT8 || type == CLS_INT8_EXACT) && c->q8 == NULL) {
        if (p->dim % CLS_GROUP != 0) {
            fprintf(stderr, "int8 output layer requires dim to be a multiple of %d (dim = %d)\n", CLS_GROUP, p->dim);
            exit(EXIT_FAILURE);
        }
        size_t groups = n / CLS_GROUP;
        c->q8 = malloc(n * sizeof(int8_t));
        c->q8s = malloc(groups * sizeof(float));
        c->xq = malloc(p->dim * sizeof(int8_t));
        c->xs = malloc(p->dim / CLS_GROUP * sizeof(float));
        if (!c->q8 || !c->q8s || !c->xq || !c->xs) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        for (size_t g = 0; g < groups; g++) {
            cls_quantize_group(c->q8 + g * CLS_GROUP, c->q8s + g, w + g * CLS_GROUP);
        }
    }
    if (type == CLS_INT8_EXACT && c->bound == NULL) {
        int n_blocks = (p->vocab_size + CLS_BLOCK - 1) / CLS_BLOCK;
        c->xk = malloc(p->dim / CLS_GROUP * sizeof(float));
        c->bound = malloc(p->vocab_size * sizeof(float));
        c->blk_lo = malloc(n_blocks * sizeof(float));
        c->blk_cand = malloc(n_blocks * sizeof(int));
        if (!c->xk || !c->bound || !c->blk_lo || !c->blk_cand) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    c->type = type;
}

// x を int8 に量子化する。CLS_INT8_EXACT のときは誤差の上限の係数 c->xk も計算する
static void cls_quantize_x(Classifier* c, const float* x, int n) {
    for (int g = 0; g < n / CLS_GROUP; g++) {
        const float* xg = x + g * CLS_GROUP;
        int8_t* qg = c->xq + g * CLS_GROUP;
        cls_quantize_group(qg, c->xs + g, xg);
        if (c->type == CLS_INT8_EXACT) {
            float sum_x = 0.0f, sum_e = 0.0f;
            for (int j = 0; j < CLS_GROUP; j++) {
                sum_x += fabsf(xg[j]);
                sum_e += fabsf(xg[j] - c->xs[g] * qg[j]);
            }
            c->xk[g] = CLS_BOUND_MARGIN * (0.5f * sum_x + 127.0f * sum_e);
        }
    }
}

// B4: 2段階の厳密な argmax（CLS_INT8_EXACT、方針メモ 4.5 節）
//
// 1段目: int8 で全行の近似値 y'_i を計算する。w = s_w q_w + e_w（|e_w| <= s_w / 2）、
//   x = s_x q_x + e_x と書くと、真の値との差は y_i - y'_i = Σ_j (s_w q_w,j e_x,j + e_w,j x_j) なので、
//   32 要素のグループ g ごとに |q_w| <= 127 を使って
//     |y_i - y'_i| <= Σ_g s_w,g (Σ_j |x_j| / 2 + 127 Σ_j |e_x,j|)
//   となる。括弧の中は x だけで決まるので、トークンごとに c->xk に計算しておく。
//   浮動小数点の丸め（量子化、近似値、fp32 の matmul 自体の足し算）による差は、最悪でも上限の 0.5% 程度なので、
//   上限を CLS_BOUND_MARGIN 倍して含める。
// 2段目: 下限 y'_i - B_i の最大を L とし、上限 y'_i + B_i が L 以上の行だけを fp32 で計算し直す。
//   それ以外の行は fp32 で計算しても L 未満なので argmax にならない。計算し直す行は matmul と同じ形の
//   ループで計算し、fp32 の matmul とビット単位で同じ値にするので、greedy の出力は fp32 と完全に一致する。
//   候補でない行には近似値が残るが、どれも L 未満なので argmax は変わらない（temperature > 0 では使えない）。
static void cls_rescore(Transformer* t, float* xout, float* x, int n_blocks) {
    Classifier* c = &t->cls;
    int n = t->config.dim;
    int d = t->config.vocab_size;
    float* w = t->weights.wcls;
    float lo = c->blk_lo[0];
    for (int b = 1; b < n_blocks; b++) { lo = fmaxf(lo, c->blk_lo[b]); }
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < n_blocks; b++) {
        int r0 = b * CLS_BLOCK;
        int r1 = r0 + CLS_BLOCK < d ? r0 + CLS_BLOCK : d;
        int cand = 0;
        for (int i = r0; i < r1; i++) {
            if (xout[i] + c->bound[i] < lo) { continue; }
            // matmul と同じ形のループ（足し算の順番を揃え、結果のビットを fp32 の matmul と同じにする）
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
            cand++;
        }
        c->blk_cand[b] = cand;
    }
    int total = 0;
    for (int k = 0; k < n_blocks; k++) { total += c->blk_cand[k]; }
    c->n_calls++;
    c->n_cand += total;
    if (total > c->max_cand) { c->max_cand = total; }
}

// xout (vocab_size,) = wcls (vocab_size, dim) @ x (dim,)。形式は t->cls.type で選ぶ
static void classifier_matmul(Transformer* t, float* xout, float* x) {
    Classifier* c = &t->cls;
    int n = t->config.dim;
    int d = t->config.vocab_size;
    if (c->type == CLS_FP32) {
        matmul_nt(xout, x, t->weights.wcls, n, d, 0);
        return;
    }
    if (c->type != CLS_FP16) { cls_quantize_x(c, x, n); }
    ClsRowsFn rows = cls_rows_fn(c->type, c->isa);
    int exact = c->type == CLS_INT8_EXACT;
    int n_blocks = (d + CLS_BLOCK - 1) / CLS_BLOCK;
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < n_blocks; b++) {
        int r0 = b * CLS_BLOCK;
        int r1 = r0 + CLS_BLOCK < d ? r0 + CLS_BLOCK : d;
        rows(xout, x, c, n, r0, r1);
        if (exact) {
            float lo = xout[r0] - c->bound[r0];
            for (int i = r0 + 1; i < r1; i++) { lo = fmaxf(lo, xout[i] - c->bound[i]); }
            c->blk_lo[b] = lo;
        }
    }
    if (exact) { cls_rescore(t, xout, x, n_blocks); }
}

// ============================================================
// Token Embedding
// ============================================================

static void embedding_lookup(Transformer* t, int token) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    float* row = w->token_embedding_table + token * p->dim;
    memcpy(s->x, row, p->dim * sizeof(float));
}


// ============================================================
// Attention: RMSNorm
// ============================================================

static void attention_norm(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    rmsnorm(
        s->xb,
        s->x,
        w->rms_att_weight + l * dim,
        dim
    );
}


// ============================================================
// Attention: Q, K, V Projection
// ============================================================

static void qkv_projection(Transformer* t, int l, int pos) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int kv_dim = dim * p->n_kv_heads / p->n_heads;

    // K, V の出力先は KV Cache 内の現在位置
    int loff = l * p->seq_len * kv_dim;

    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // Q = Wq * xb
    matmul(
        s->q, s->xb,
        w->wq + l * dim * dim,
        dim, dim
    );

    // K = Wk * xb（KV Cacheに直接書き込む）
    matmul(
        s->k, s->xb,
        w->wk + l * dim * kv_dim,
        dim, kv_dim
    );

    // V = Wv * xb（KV Cacheに直接書き込む）
    matmul(
        s->v, s->xb,
        w->wv + l * dim * kv_dim,
        dim, kv_dim
    );
}


// ============================================================
// Attention: RoPE
// ============================================================

static void apply_rope(Transformer* t, int pos) {
    Config* p = &t->config;
    RunState* s = &t->state;

    int dim = p->dim;
    int head_size = dim / p->n_heads;
    int kv_dim = dim * p->n_kv_heads / p->n_heads;

    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;

        float freq = 1.0f / powf(
            10000.0f,
            head_dim / (float)head_size
        );

        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        // Query と Key に適用（Valueには適用しない）
        int rotn = i < kv_dim ? 2 : 1;

        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k;

            float v0 = vec[i];
            float v1 = vec[i + 1];

            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}


// ============================================================
// Attention: Score, Softmax, Weighted Sum
// ============================================================

static void compute_attention(Transformer* t, int l, int pos) {
    Config* p = &t->config;
    RunState* s = &t->state;

    int head_size = p->dim / p->n_heads;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int loff = l * p->seq_len * kv_dim;

    #pragma omp parallel
    {
        int h0, h1;
        thread_share(p->n_heads, g_layer_threads, &h0, &h1);
        for (int h = h0; h < h1; h++) {

            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;

            // 1. Attention Score
            for (int pos2 = 0; pos2 <= pos; pos2++) {
                float* k = s->key_cache
                        + loff
                        + pos2 * kv_dim
                        + (h / kv_mul) * head_size;

                float score = 0.0f;

                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }

                score /= sqrtf(head_size);
                att[pos2] = score;
            }

            // 2. Softmax
            softmax(att, pos + 1);

            // 3. Weighted Sum
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            for (int pos2 = 0; pos2 <= pos; pos2++) {
                float* v = s->value_cache
                        + loff
                        + pos2 * kv_dim
                        + (h / kv_mul) * head_size;

                float a = att[pos2];

                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
    }

}


// ============================================================
// Attention: Output Projection + Residual
// ============================================================

static void attention_output(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    // Wo
    matmul(
        s->xb2, s->xb,
        w->wo + l * dim * dim,
        dim, dim
    );

    // Residual
    for (int i = 0; i < dim; i++) {
        s->x[i] += s->xb2[i];
    }
}


// ============================================================
// FFN: RMSNorm
// ============================================================

static void ffn_norm(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    rmsnorm(
        s->xb,
        s->x,
        w->rms_ffn_weight + l * dim,
        dim
    );
}


// ============================================================
// FFN: W1, W3 Projection
// ============================================================

static void ffn_up_projection(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    // W1
    matmul(
        s->hb, s->xb,
        w->w1 + l * dim * hidden_dim,
        dim, hidden_dim
    );

    // W3
    matmul(
        s->hb2, s->xb,
        w->w3 + l * dim * hidden_dim,
        dim, hidden_dim
    );
}


// ============================================================
// FFN: SwiGLU
// ============================================================

static void ffn_swiglu(Transformer* t) {
    Config* p = &t->config;
    RunState* s = &t->state;

    for (int i = 0; i < p->hidden_dim; i++) {
        float val = s->hb[i];

        // SiLU
        val *= 1.0f / (1.0f + expf(-val));

        // Elementwise multiplication
        val *= s->hb2[i];

        s->hb[i] = val;
    }
}


// ============================================================
// FFN: W2 Projection + Residual
// ============================================================

static void ffn_down_projection(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    // W2
    matmul(
        s->xb, s->hb,
        w->w2 + l * dim * hidden_dim,
        hidden_dim, dim
    );

    // Residual
    for (int i = 0; i < dim; i++) {
        s->x[i] += s->xb[i];
    }
}


// ============================================================
// Final RMSNorm + Logits
// ============================================================

static float* final_projection(Transformer* t) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    rmsnorm(
        s->x, s->x,
        w->rms_final_weight,
        p->dim
    );

    // 形式（fp32 / fp16 / int8）は -q で選ぶ
    classifier_matmul(t, s->logits, s->x);

    return s->logits;
}


// ============================================================
// Stage timing (compile with -DSTAGE_TIMING)
//
// forward() 内では配列に経過時間を記録するだけで何も出力しない。
// 生成終了後に stage_timing_dump() で従来と同じ形式でまとめて出力する。
// ============================================================

#ifdef STAGE_TIMING

enum {
    ST_EMBED, ST_ATT_NORM, ST_QKV, ST_ROPE, ST_ATTN, ST_ATT_OUT,
    ST_FFN_NORM, ST_FFN_UP, ST_SWIGLU, ST_FFN_DOWN, ST_FINAL,
    N_STAGES
};

static const char* stage_names[N_STAGES] = {
    "001:embedding-lookup", "002:attention-norm", "003:qkv-projection",
    "004:apply-rope", "005:compute-attention", "006:attention-output",
    "007:ffn-norm", "008:ffn-up-projection", "009:ffn-swiglu",
    "010:ffn-down-projection", "011:final-projection",
};

// [steps][n_layers][N_STAGES]。層に属さないEMBED/FINALはlayer=0にまとめる。
static int64_t* g_stage_ns = NULL;
static int g_steps = 0;
static int g_layers = 0;

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// 経過時間を記録し、現在時刻を返す（次のステージの開始時刻になる）
static inline int64_t stage_mark(int stage, int pos, int layer, int64_t t0) {
    int64_t t1 = now_ns();
    if (g_stage_ns != NULL && pos < g_steps) {
        g_stage_ns[((size_t)pos * g_layers + layer) * N_STAGES + stage] = t1 - t0;
    }
    return t1;
}

static void stage_timing_init(int steps, int n_layers) {
    g_steps = steps;
    g_layers = n_layers;
    size_t n = (size_t)steps * n_layers * N_STAGES;
    g_stage_ns = malloc(n * sizeof(int64_t));
    if (g_stage_ns == NULL) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    // 計測中にページフォールトが起きないよう、ここで全ページに触れておく
    memset(g_stage_ns, 0, n * sizeof(int64_t));
}

static void stage_timing_free(void) {
    free(g_stage_ns);
    g_stage_ns = NULL;
}

// 従来の measure_time() と同じ形式で出力する（bench_affinity.sh の集計で読める）
static void stage_timing_dump(FILE* f, int n_pos) {
    if (g_stage_ns == NULL) { return; }
    if (n_pos > g_steps) { n_pos = g_steps; }
    for (int pos = 0; pos < n_pos; pos++) {
        int64_t* tok = g_stage_ns + (size_t)pos * g_layers * N_STAGES;
        int64_t forward_ns = tok[ST_EMBED];
        fprintf(f, "[token:%d] %s: %.9f seconds\n",
                pos, stage_names[ST_EMBED], tok[ST_EMBED] / 1e9);
        for (int l = 0; l < g_layers; l++) {
            int64_t* lay = tok + (size_t)l * N_STAGES;
            int64_t layer_ns = 0;
            for (int st = ST_ATT_NORM; st <= ST_FFN_DOWN; st++) {
                fprintf(f, "[token:%d, layer:%d] %s: %.9f seconds\n",
                        pos, l, stage_names[st], lay[st] / 1e9);
                layer_ns += lay[st];
            }
            fprintf(f, "[token:%d, layer:%d] 102:layer-compute-total: %.9f seconds\n",
                    pos, l, layer_ns / 1e9);
            forward_ns += layer_ns;
        }
        fprintf(f, "[token:%d] %s: %.9f seconds\n",
                pos, stage_names[ST_FINAL], tok[ST_FINAL] / 1e9);
        forward_ns += tok[ST_FINAL];
        fprintf(f, "[token:%d] 103:forward-compute-total: %.9f seconds\n",
                pos, forward_ns / 1e9);
    }
}

#define T_START() int64_t _stage_t = now_ns()
#define T_MARK(st, l) (_stage_t = stage_mark((st), pos, (l), _stage_t))

#else

#define T_START()
#define T_MARK(st, l)

#endif

float* forward(Transformer* t, int token, int pos) {
    T_START();

    embedding_lookup(t, token);
    T_MARK(ST_EMBED, 0);

    for (int l = 0; l < t->config.n_layers; l++) {
        attention_norm(t, l);
        T_MARK(ST_ATT_NORM, l);

        qkv_projection(t, l, pos);
        T_MARK(ST_QKV, l);

        apply_rope(t, pos);
        T_MARK(ST_ROPE, l);

        compute_attention(t, l, pos);
        T_MARK(ST_ATTN, l);

        attention_output(t, l);
        T_MARK(ST_ATT_OUT, l);

        ffn_norm(t, l);
        T_MARK(ST_FFN_NORM, l);

        ffn_up_projection(t, l);
        T_MARK(ST_FFN_UP, l);

        ffn_swiglu(t);
        T_MARK(ST_SWIGLU, l);

        ffn_down_projection(t, l);
        T_MARK(ST_FFN_DOWN, l);
    }

    float* logits = final_projection(t);
    T_MARK(ST_FINAL, 0);

    return logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop


void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    #ifdef STAGE_TIMING
        stage_timing_dump(stderr, pos);
    #endif

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -q <string> output layer weights: fp32|fp16|int8|int8-exact, default: fp32\n");
    fprintf(stderr, "              int8-exact: int8 + fp32 rescoring, same output as fp32 (needs -t 0)\n");
    fprintf(stderr, "  -L <int> threads for layer matmuls/attention (0 = all), default: 0\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

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
    char *cls_name = "fp32";    // output layer weights: fp32|fp16|int8|int8-exact

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
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'q') { cls_name = argv[i + 1]; }
        else if (argv[i][1] == 'L') { g_layer_threads = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    if (g_layer_threads < 0) g_layer_threads = 0;
    ClsType cls_type = CLS_FP32;
    if (strcmp(cls_name, "fp32") == 0) { cls_type = CLS_FP32; }
    else if (strcmp(cls_name, "fp16") == 0) { cls_type = CLS_FP16; }
    else if (strcmp(cls_name, "int8") == 0) { cls_type = CLS_INT8; }
    else if (strcmp(cls_name, "int8-exact") == 0) { cls_type = CLS_INT8_EXACT; }
    else { fprintf(stderr, "unknown output layer weights: %s\n", cls_name); error_usage(); }
    // int8-exact は argmax の行以外の logits が近似値なので、サンプリングには使えない
    if (cls_type == CLS_INT8_EXACT && temperature != 0.0f) {
        fprintf(stderr, "-q int8-exact requires greedy decoding (-t 0)\n");
        error_usage();
    }

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    build_classifier(&transformer, cls_type);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    #ifdef STAGE_TIMING
        stage_timing_init(steps, transformer.config.n_layers);
    #endif

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }
    // 2段目で fp32 で計算し直した行の数（方針メモ 4.4 節: 2段目のコストの把握用）
    if (transformer.cls.type == CLS_INT8_EXACT && transformer.cls.n_calls > 0) {
        fprintf(stderr, "int8-exact candidates per token: mean %.1f, max %d (of %d)\n",
                (double)transformer.cls.n_cand / transformer.cls.n_calls,
                transformer.cls.max_cand, transformer.config.vocab_size);
    }

    // memory and file handles cleanup
    #ifdef STAGE_TIMING
        stage_timing_free();
    #endif
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
