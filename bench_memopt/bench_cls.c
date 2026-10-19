// 出力層（wcls: rows x dim の matvec）だけを取り出した単体ベンチマーク。
// 方針: プロジェクト/output-layer-memory-optimization.md 第6節 Step 2
//
// 変えられる条件:
//   --kernel  fp32 | fp32_r4 | fp16 | int8 | read | none
//   --page    file | 4k | 2m      （file は run.c と同じく model を mmap したまま読む）
//   --src     model | synth       （model: stories15M.bin の wcls、synth: 乱数）
//   --rows N / --dim N            （データ量。synth なら L3 内外のスイープに使う）
//   --split K                     （同じ行列を K 回の並列区間に分けて読む。呼び出しごとの固定コストの測定）
//   --weights w0,w1,...           （スレッドごとの行の配分比。P+E 分担の試作。省略時は均等）
//   --schedule range | static | dynamic:K | guided:K
//                                 （行の配り方。range は各スレッドに連続した範囲（--weights の比）、
//                                   それ以外は run.c の出力層と同じ 64 行のブロックを OpenMP の schedule で配る。
//                                   K はブロックの数。static は run.c の今の配り方）
//   --prefetch BYTES              （fp32_r4 のソフトウェアプリフェッチ距離。0 で無効）
//   --flush-mb N                  （各反復の間に N MB の別バッファを読み、キャッシュを追い出す）
// スレッド数と配置は OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES と taskset で外から指定する。
//
// 出力: CSV 1行（--header でヘッダ行だけ出す）。
//
// ビルド: gcc -Ofast -fopenmp -march=native -falign-functions=64 -falign-loops=64 bench_cls.c -lm -o bench_cls

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <omp.h>

#define HUGE_2M (2UL * 1024 * 1024)
#define Q_GROUP 32
#define MAX_THREADS 64

// ============================================================
// Options
// ============================================================

typedef struct {
    const char* kernel;
    const char* page;
    const char* src;
    const char* model;
    int rows;
    int dim;
    int split;
    int warmup;
    int min_iters;
    double min_time;
    int prefetch;
    int flush_mb;
    const char* weights;
    const char* schedule;
    unsigned seed;
} Options;

static const char* CSV_HEADER =
    "kernel,page,src,rows,dim,bytes,split,threads,cpus,weights,prefetch,flush_mb,"
    "iters,median_us,p10_us,p90_us,min_us,gbps_median,gbps_best,"
    "anon_huge_kb,max_abs_err,argmax_ok,schedule,thr_busy_us,thr_rows,imbalance_pct";

static void usage(void) {
    fprintf(stderr,
        "usage: bench_cls [options]\n"
        "  --kernel fp32|fp32_r4|fp16|int8|read|none (default fp32)\n"
        "  --page file|4k|2m      (default 2m; file requires --src model and fp32/fp32_r4/read)\n"
        "  --src model|synth      (default model)\n"
        "  --model PATH           (default stories15M.bin)\n"
        "  --rows N --dim N       (default: model's vocab x dim, synth: 32000 x 288)\n"
        "  --split K              (default 1)\n"
        "  --weights w0,w1,...    (per-thread row share, default equal)\n"
        "  --schedule S           range|static|dynamic:K|guided:K (default range; K in 64-row blocks)\n"
        "  --prefetch BYTES       (fp32_r4 only, default 0)\n"
        "  --flush-mb N           (default 0)\n"
        "  --warmup N --min-iters N --min-time SEC (default 10, 50, 0.5)\n"
        "  --seed N\n"
        "  --header               (print CSV header and exit)\n");
    exit(EXIT_FAILURE);
}

static void parse_args(int argc, char** argv, Options* o) {
    *o = (Options){
        .kernel = "fp32", .page = "2m", .src = "model", .model = "stories15M.bin",
        .rows = 0, .dim = 0, .split = 1, .warmup = 10, .min_iters = 50, .min_time = 0.5,
        .prefetch = 0, .flush_mb = 0, .weights = NULL, .schedule = "range", .seed = 42,
    };
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strcmp(a, "--header") == 0) { puts(CSV_HEADER); exit(EXIT_SUCCESS); }
        if (i + 1 >= argc) { usage(); }
        const char* v = argv[++i];
        if      (strcmp(a, "--kernel") == 0)    { o->kernel = v; }
        else if (strcmp(a, "--page") == 0)      { o->page = v; }
        else if (strcmp(a, "--src") == 0)       { o->src = v; }
        else if (strcmp(a, "--model") == 0)     { o->model = v; }
        else if (strcmp(a, "--rows") == 0)      { o->rows = atoi(v); }
        else if (strcmp(a, "--dim") == 0)       { o->dim = atoi(v); }
        else if (strcmp(a, "--split") == 0)     { o->split = atoi(v); }
        else if (strcmp(a, "--weights") == 0)   { o->weights = v; }
        else if (strcmp(a, "--schedule") == 0)  { o->schedule = v; }
        else if (strcmp(a, "--prefetch") == 0)  { o->prefetch = atoi(v); }
        else if (strcmp(a, "--flush-mb") == 0)  { o->flush_mb = atoi(v); }
        else if (strcmp(a, "--warmup") == 0)    { o->warmup = atoi(v); }
        else if (strcmp(a, "--min-iters") == 0) { o->min_iters = atoi(v); }
        else if (strcmp(a, "--min-time") == 0)  { o->min_time = atof(v); }
        else if (strcmp(a, "--seed") == 0)      { o->seed = (unsigned)atoi(v); }
        else { usage(); }
    }
}

// ============================================================
// Utilities
// ============================================================

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void die(const char* msg) {
    fprintf(stderr, "bench_cls: %s\n", msg);
    exit(EXIT_FAILURE);
}

static unsigned rng_state;
static float rand_uniform(void) {  // [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) * (1.0 / 8388608.0)) - 1.0f;
}

// 4KB ページ（THP 無効）か 2MB ページ（THP 有効）の匿名メモリを確保する
static void* alloc_pages(size_t bytes, const char* page) {
    if (strcmp(page, "2m") == 0) {
        size_t len = bytes + HUGE_2M;
        char* raw = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { die("mmap failed"); }
        char* p = (char*)(((uintptr_t)raw + HUGE_2M - 1) & ~(HUGE_2M - 1));
        if (madvise(p, bytes, MADV_HUGEPAGE) != 0) { die("madvise(MADV_HUGEPAGE) failed"); }
        return p;
    }
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { die("mmap failed"); }
    madvise(p, bytes, MADV_NOHUGEPAGE);
    return p;
}

// ptr を含むマッピングの AnonHugePages[kB] を /proc/self/smaps から読む
static long anon_huge_kb(const void* ptr) {
    FILE* f = fopen("/proc/self/smaps", "r");
    if (f == NULL) { return -1; }
    char line[512];
    int in_range = 0;
    long result = -1;
    uintptr_t addr = (uintptr_t)ptr;
    while (fgets(line, sizeof(line), f)) {
        unsigned long lo, hi;
        // マッピングの見出し行は "start-end perms ..." の形（16進数の直後が '-'）
        if (line[strspn(line, "0123456789abcdef")] == '-' && sscanf(line, "%lx-%lx ", &lo, &hi) == 2) {
            in_range = (addr >= lo && addr < hi);
            continue;
        }
        long kb;
        if (in_range && sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) {
            result = kb;
            break;
        }
    }
    fclose(f);
    return result;
}

// 各 OpenMP スレッドが実際に動いている CPU を、スレッド番号順に空白区切りで返す。
// （OMP_PROC_BIND で固定した後はメインスレッドの affinity が CPU 1つになるので、
//   sched_getaffinity ではなく各スレッドの sched_getcpu を使う）
static void thread_cpu_string(char* buf, size_t size) {
    int cpu[MAX_THREADS];
    int n = 0;
    #pragma omp parallel
    {
        cpu[omp_get_thread_num()] = sched_getcpu();
        #pragma omp single
        n = omp_get_num_threads();
    }
    size_t len = 0;
    buf[0] = '\0';
    for (int t = 0; t < n && len < size; t++) {
        len += snprintf(buf + len, size - len, t ? " %d" : "%d", cpu[t]);
    }
}

static int cmp_i64(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x > y) - (x < y);
}

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// ============================================================
// Data
// ============================================================

typedef struct {
    int rows, dim;
    const float* w32;      // fp32 / fp32_r4 / read
    const uint16_t* w16;   // fp16
    const int8_t* q8;      // int8 の値（rows x dim）
    const float* q8s;      // int8 のスケール（rows x dim/32）
    size_t bytes;          // 1回の matvec で読む重みのバイト数
    const void* base;      // AnonHugePages を調べる対象
} Weights;

static float* load_or_make_fp32(const Options* o, int* rows, int* dim, int* mapped) {
    *mapped = 0;
    if (strcmp(o->src, "synth") == 0) {
        if (*rows <= 0) { *rows = 32000; }
        if (*dim <= 0) { *dim = 288; }
        size_t n = (size_t)*rows * *dim;
        float* w = alloc_pages(n * sizeof(float), strcmp(o->page, "file") == 0 ? "4k" : o->page);
        for (size_t i = 0; i < n; i++) { w[i] = 0.05f * rand_uniform(); }
        return w;
    }

    int fd = open(o->model, O_RDONLY);
    if (fd < 0) { die("cannot open model"); }
    int header[7];
    if (read(fd, header, sizeof(header)) != sizeof(header)) { die("cannot read model header"); }
    int mdim = header[0], vocab = abs(header[5]);
    if (*dim <= 0) { *dim = mdim; }
    if (*rows <= 0) { *rows = vocab; }
    if (*dim != mdim || *rows > vocab) { die("--src model requires dim == model dim and rows <= vocab"); }

    struct stat st;
    fstat(fd, &st);
    char* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { die("mmap model failed"); }
    // run.c と同じく、ヘッダ（Config = int x 7）の直後が token_embedding_table（= wcls）
    float* src = (float*)(data + sizeof(header));
    if (strcmp(o->page, "file") == 0) {
        *mapped = 1;
        return src;
    }
    // A1: 匿名メモリにコピーする（2m なら THP に載る）
    size_t bytes = (size_t)*rows * *dim * sizeof(float);
    float* w = alloc_pages(bytes, o->page);
    memcpy(w, src, bytes);
    return w;
}

static Weights prepare_weights(const Options* o, float** ref_w32) {
    Weights W = {0};
    int rows = o->rows, dim = o->dim, mapped;
    float* w32 = load_or_make_fp32(o, &rows, &dim, &mapped);
    if (dim % Q_GROUP != 0) { die("dim must be a multiple of 32"); }
    W.rows = rows;
    W.dim = dim;
    *ref_w32 = w32;
    size_t n = (size_t)rows * dim;
    const char* k = o->kernel;

    if (strcmp(k, "fp16") == 0) {
        uint16_t* w16 = alloc_pages(n * sizeof(uint16_t), o->page);
        for (size_t i = 0; i < n; i += 8) {
            __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(w32 + i), _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i*)(w16 + i), h);
        }
        W.w16 = w16;
        W.bytes = n * sizeof(uint16_t);
        W.base = w16;
    } else if (strcmp(k, "int8") == 0) {
        size_t groups = n / Q_GROUP;
        // 値とスケールを1つの領域に置く（huge page の確認を1回で済ませる）
        size_t qbytes = (n + 63) & ~(size_t)63;
        char* buf = alloc_pages(qbytes + groups * sizeof(float), o->page);
        int8_t* q = (int8_t*)buf;
        float* s = (float*)(buf + qbytes);
        for (size_t g = 0; g < groups; g++) {
            const float* src = w32 + g * Q_GROUP;
            float amax = 0.0f;
            for (int j = 0; j < Q_GROUP; j++) { amax = fmaxf(amax, fabsf(src[j])); }
            float scale = amax / 127.0f;
            float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int j = 0; j < Q_GROUP; j++) { q[g * Q_GROUP + j] = (int8_t)lrintf(src[j] * inv); }
            s[g] = scale;
        }
        W.q8 = q;
        W.q8s = s;
        W.bytes = n + groups * sizeof(float);
        W.base = buf;
    } else {
        W.w32 = w32;
        W.bytes = (strcmp(k, "none") == 0) ? 0 : n * sizeof(float);
        W.base = mapped ? NULL : w32;
    }
    return W;
}

// ============================================================
// Kernels（いずれも行 [r0, r1) を計算する）
// ============================================================

// run.c の matmul と同じ素朴なループ（コンパイラの自動ベクトル化に任せる）
static void k_fp32(float* out, const float* x, const Weights* W, int r0, int r1) {
    int n = W->dim;
    for (int i = r0; i < r1; i++) {
        const float* row = W->w32 + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) { val += row[j] * x[j]; }
        out[i] = val;
    }
}

// A2: 4行を同時に計算する（＋任意でソフトウェアプリフェッチ）
static int g_prefetch = 0;
static void k_fp32_r4(float* out, const float* x, const Weights* W, int r0, int r1) {
    int n = W->dim;
    int i = r0;
    for (; i + 4 <= r1; i += 4) {
        const float* w0 = W->w32 + (size_t)i * n;
        if (g_prefetch > 0) {
            // 4行（連続領域）の先 g_prefetch バイトを 64B ごとに先読みする
            const char* pf = (const char*)w0 + g_prefetch;
            for (int b = 0; b < 4 * n * (int)sizeof(float); b += 64) {
                _mm_prefetch(pf + b, _MM_HINT_T0);
            }
        }
        const float* w1 = w0 + n;
        const float* w2 = w1 + n;
        const float* w3 = w2 + n;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int j = 0; j < n; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xv, a3);
        }
        out[i] = hsum256(a0);
        out[i + 1] = hsum256(a1);
        out[i + 2] = hsum256(a2);
        out[i + 3] = hsum256(a3);
    }
    if (i < r1) { k_fp32(out, x, W, i, r1); }
}

// B1: fp16 の重み（F16C で fp32 に変換して FMA）。4行同時
static void k_fp16(float* out, const float* x, const Weights* W, int r0, int r1) {
    int n = W->dim;
    for (int i = r0; i < r1; i += 4) {
        int nr = (r1 - i < 4) ? r1 - i : 4;
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int j = 0; j < n; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            for (int r = 0; r < nr; r++) {
                const uint16_t* row = W->w16 + (size_t)(i + r) * n;
                __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + j)));
                acc[r] = _mm256_fmadd_ps(wv, xv, acc[r]);
            }
        }
        for (int r = 0; r < nr; r++) { out[i + r] = hsum256(acc[r]); }
    }
}

// B2: int8（32要素ごとに fp32 スケール）。x も同じ形式に量子化してから内積を取る
static int8_t* g_xq;
static float* g_xs;
static void quantize_x(const float* x, int n) {
    for (int g = 0; g < n / Q_GROUP; g++) {
        float amax = 0.0f;
        for (int j = 0; j < Q_GROUP; j++) { amax = fmaxf(amax, fabsf(x[g * Q_GROUP + j])); }
        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < Q_GROUP; j++) { g_xq[g * Q_GROUP + j] = (int8_t)lrintf(x[g * Q_GROUP + j] * inv); }
        g_xs[g] = scale;
    }
}

static inline __m256i dot_i8x32(__m256i w, __m256i x) {
    // 符号付き x 符号付きの積和を、|w| x sign(x, w) の符号なし x 符号付きに直して計算する
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sx = _mm256_sign_epi8(x, w);
#ifdef __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sx);
#else
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
#endif
}

static void k_int8(float* out, const float* x, const Weights* W, int r0, int r1) {
    (void)x;
    int n = W->dim;
    int gpr = n / Q_GROUP;
    for (int i = r0; i < r1; i++) {
        const int8_t* row = W->q8 + (size_t)i * n;
        const float* rs = W->q8s + (size_t)i * gpr;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < gpr; g++) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(row + g * Q_GROUP));
            __m256i xv = _mm256_loadu_si256((const __m256i*)(g_xq + g * Q_GROUP));
            __m256 p = _mm256_cvtepi32_ps(dot_i8x32(wv, xv));
            acc = _mm256_fmadd_ps(p, _mm256_set1_ps(rs[g] * g_xs[g]), acc);
        }
        out[i] = hsum256(acc);
    }
}

// 帯域の上限の目安: 重みを読んで足すだけ
static float g_sink[MAX_THREADS * 16];
static void k_read(float* out, const float* x, const Weights* W, int r0, int r1) {
    (void)out; (void)x;
    const float* p = W->w32 + (size_t)r0 * W->dim;
    size_t n = (size_t)(r1 - r0) * W->dim;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + j));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + j + 8));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + j + 16));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + j + 24));
    }
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; j < n; j++) { s += p[j]; }
    g_sink[omp_get_thread_num() * 16] += s;
}

// 並列区間の起動と同期だけのコストを測る
static void k_none(float* out, const float* x, const Weights* W, int r0, int r1) {
    (void)out; (void)x; (void)W; (void)r0; (void)r1;
}

typedef void (*KernelFn)(float*, const float*, const Weights*, int, int);

static KernelFn select_kernel(const char* name) {
    if (strcmp(name, "fp32") == 0)    { return k_fp32; }
    if (strcmp(name, "fp32_r4") == 0) { return k_fp32_r4; }
    if (strcmp(name, "fp16") == 0)    { return k_fp16; }
    if (strcmp(name, "int8") == 0)    { return k_int8; }
    if (strcmp(name, "read") == 0)    { return k_read; }
    if (strcmp(name, "none") == 0)    { return k_none; }
    die("unknown kernel");
    return NULL;
}

// ============================================================
// Partition and run
// ============================================================

static int g_nthreads;
static double g_share[MAX_THREADS + 1];  // 累積の配分比（0..1）

static void setup_partition(const char* weights) {
    g_nthreads = omp_get_max_threads();
    if (g_nthreads > MAX_THREADS) { die("too many threads"); }
    double w[MAX_THREADS];
    int nw = 0;
    if (weights != NULL) {
        char* buf = strdup(weights);
        for (char* tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
            if (nw >= MAX_THREADS) { die("too many weights"); }
            w[nw++] = atof(tok);
        }
        free(buf);
        if (nw != g_nthreads) { die("--weights count must equal OMP_NUM_THREADS"); }
    } else {
        for (int t = 0; t < g_nthreads; t++) { w[t] = 1.0; }
        nw = g_nthreads;
    }
    double total = 0.0;
    for (int t = 0; t < nw; t++) { total += w[t]; }
    g_share[0] = 0.0;
    for (int t = 0; t < nw; t++) { g_share[t + 1] = g_share[t] + w[t] / total; }
    g_share[nw] = 1.0;
}

// [r0, r1) をスレッド t の担当範囲に分ける。境界は 4 行単位に揃える（fp32_r4 / fp16 用）
static inline void thread_range(int t, int r0, int r1, int* a, int* b) {
    int n = r1 - r0;
    *a = r0 + ((int)(g_share[t] * n) & ~3);
    *b = (t == g_nthreads - 1) ? r1 : r0 + ((int)(g_share[t + 1] * n) & ~3);
}

// 行の配り方。SCHED_RANGE は各スレッドに連続した範囲（thread_range）、
// それ以外は CLS_BLOCK 行のブロックを omp for schedule(runtime) で配る（run.c の classifier_matmul と同じ単位）
#define CLS_BLOCK 64
enum { SCHED_RANGE, SCHED_BLOCKS };
static int g_sched = SCHED_RANGE;

static void setup_schedule(const char* spec, int weighted) {
    if (strcmp(spec, "range") == 0) { g_sched = SCHED_RANGE; return; }
    if (weighted) { die("--weights is only for --schedule range"); }
    g_sched = SCHED_BLOCKS;
    int chunk = 0;
    const char* colon = strchr(spec, ':');
    if (colon) { chunk = atoi(colon + 1); }
    if (strncmp(spec, "static", 6) == 0)       { omp_set_schedule(omp_sched_static, chunk); }
    else if (strncmp(spec, "dynamic", 7) == 0) { omp_set_schedule(omp_sched_dynamic, chunk > 0 ? chunk : 1); }
    else if (strncmp(spec, "guided", 6) == 0)  { omp_set_schedule(omp_sched_guided, chunk > 0 ? chunk : 1); }
    else { die("unknown --schedule"); }
}

// スレッドごとの所要時間（並列区間に入ってから自分の担当を終えるまで）と、計算した行数の合計。
// 計測中だけ記録する。所要時間の差が、遅いスレッドを待っている時間になる
static int g_record = 0;
static int64_t g_busy_ns[MAX_THREADS * 8];
static int64_t g_rows[MAX_THREADS * 8];
static int64_t g_iter_imbalance_ns, g_iter_max_ns;  // 反復ごとの (最大 - 最小) と 最大 の合計

static void run_once(KernelFn kernel, float* out, const float* x, const Weights* W, int split) {
    if (kernel == k_int8) { quantize_x(x, W->dim); }
    int64_t busy[MAX_THREADS];
    for (int k = 0; k < split; k++) {
        int r0 = (int)((int64_t)W->rows * k / split);
        int r1 = (int)((int64_t)W->rows * (k + 1) / split);
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            int64_t t0 = g_record ? now_ns() : 0;
            int rows = 0;
            if (g_sched == SCHED_RANGE) {
                int a, b;
                thread_range(t, r0, r1, &a, &b);
                kernel(out, x, W, a, b);
                rows = b - a;
            } else {
                int n_blocks = (r1 - r0 + CLS_BLOCK - 1) / CLS_BLOCK;
                #pragma omp for schedule(runtime) nowait
                for (int blk = 0; blk < n_blocks; blk++) {
                    int a = r0 + blk * CLS_BLOCK;
                    int b = a + CLS_BLOCK < r1 ? a + CLS_BLOCK : r1;
                    kernel(out, x, W, a, b);
                    rows += b - a;
                }
            }
            if (g_record) {
                busy[t] = now_ns() - t0;
                g_busy_ns[t * 8] += busy[t];
                g_rows[t * 8] += rows;
            }
        }
        if (g_record) {
            int64_t lo = busy[0], hi = busy[0];
            for (int t = 1; t < g_nthreads; t++) {
                if (busy[t] < lo) { lo = busy[t]; }
                if (busy[t] > hi) { hi = busy[t]; }
            }
            g_iter_imbalance_ns += hi - lo;
            g_iter_max_ns += hi;
        }
    }
}

static float flush_cache(const float* buf, size_t n) {
    float s = 0.0f;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < n; i += 16) { s += buf[i]; }
    return s;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    Options o;
    parse_args(argc, argv, &o);
    rng_state = o.seed;
    g_prefetch = o.prefetch;

    if (strcmp(o.page, "file") == 0 && strcmp(o.src, "model") != 0) { die("--page file requires --src model"); }
    if (strcmp(o.page, "file") == 0 &&
        (strcmp(o.kernel, "fp16") == 0 || strcmp(o.kernel, "int8") == 0)) {
        die("--page file is only for fp32 / fp32_r4 / read / none");
    }
    if (o.split < 1) { die("--split must be >= 1"); }

    KernelFn kernel = select_kernel(o.kernel);
    setup_partition(o.weights);
    setup_schedule(o.schedule, o.weights != NULL);

    float* ref_w32;
    Weights W = prepare_weights(&o, &ref_w32);
    int rows = W.rows, dim = W.dim;

    float* x = aligned_alloc(64, (size_t)dim * sizeof(float));
    float* out = aligned_alloc(64, (((size_t)rows * sizeof(float)) + 63) & ~(size_t)63);
    g_xq = aligned_alloc(64, (size_t)dim);
    g_xs = aligned_alloc(64, (size_t)(dim / Q_GROUP) * sizeof(float));
    for (int j = 0; j < dim; j++) { x[j] = rand_uniform(); }

    float* flush = NULL;
    size_t flush_n = (size_t)o.flush_mb * 1024 * 1024 / sizeof(float);
    if (flush_n > 0) {
        flush = alloc_pages(flush_n * sizeof(float), "2m");
        for (size_t i = 0; i < flush_n; i++) { flush[i] = 1.0f; }
    }

    // 正しさの確認: fp32 の重みを double で計算した値と比べる（read / none は対象外）
    double max_err = -1.0;
    int argmax_ok = -1;
    if (kernel != k_read && kernel != k_none) {
        run_once(kernel, out, x, &W, o.split);
        int am_ref = 0, am_out = 0;
        double best_ref = -INFINITY, best_out = -INFINITY;
        max_err = 0.0;
        for (int i = 0; i < rows; i++) {
            const float* row = ref_w32 + (size_t)i * dim;
            double r = 0.0;
            for (int j = 0; j < dim; j++) { r += (double)row[j] * x[j]; }
            max_err = fmax(max_err, fabs(r - out[i]));
            if (r > best_ref) { best_ref = r; am_ref = i; }
            if (out[i] > best_out) { best_out = out[i]; am_out = i; }
        }
        argmax_ok = (am_ref == am_out);
    }

    // 計測
    for (int i = 0; i < o.warmup; i++) {
        if (flush) { g_sink[0] += flush_cache(flush, flush_n); }
        run_once(kernel, out, x, &W, o.split);
    }
    g_record = 1;
    size_t cap = 1 << 16, iters = 0;
    int64_t* t = malloc(cap * sizeof(int64_t));
    int64_t t_begin = now_ns();
    while (iters < (size_t)o.min_iters || (now_ns() - t_begin) < (int64_t)(o.min_time * 1e9)) {
        if (iters == cap) { cap *= 2; t = realloc(t, cap * sizeof(int64_t)); }
        if (flush) { g_sink[0] += flush_cache(flush, flush_n); }
        int64_t t0 = now_ns();
        run_once(kernel, out, x, &W, o.split);
        t[iters++] = now_ns() - t0;
    }
    g_record = 0;
    qsort(t, iters, sizeof(int64_t), cmp_i64);
    double med = t[iters / 2] / 1e3, p10 = t[iters / 10] / 1e3, p90 = t[iters * 9 / 10] / 1e3;
    double best = t[0] / 1e3;

    char cpus[256];
    thread_cpu_string(cpus, sizeof(cpus));
    // CSV に入れるので、配分比のカンマは ':' に置き換える
    char weights[256] = "";
    if (o.weights) {
        snprintf(weights, sizeof(weights), "%s", o.weights);
        for (char* c = weights; *c; c++) { if (*c == ',') { *c = ':'; } }
    }
    // スレッドごとの平均の所要時間 [us] と平均の行数（スレッド番号順、':' 区切り）
    char thr_busy[1024] = "", thr_rows[1024] = "";
    size_t lb = 0, lr = 0;
    size_t n_regions = iters * (size_t)o.split;
    for (int i = 0; i < g_nthreads; i++) {
        lb += snprintf(thr_busy + lb, sizeof(thr_busy) - lb, i ? ":%.1f" : "%.1f",
                       g_busy_ns[i * 8] / 1e3 / n_regions);
        lr += snprintf(thr_rows + lr, sizeof(thr_rows) - lr, i ? ":%.0f" : "%.0f",
                       (double)g_rows[i * 8] / n_regions);
    }
    // 一番遅いスレッドを待つ時間の割合: 平均 (最大 - 最小) / 平均 最大
    double imbalance = g_iter_max_ns ? 100.0 * g_iter_imbalance_ns / g_iter_max_ns : 0.0;
    long huge_kb = W.base ? anon_huge_kb(W.base) : -1;
    double gb_med = W.bytes ? W.bytes / (med * 1e-6) / 1e9 : 0.0;
    double gb_best = W.bytes ? W.bytes / (best * 1e-6) / 1e9 : 0.0;

    printf("%s,%s,%s,%d,%d,%zu,%d,%d,%s,%s,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%ld,%.3g,%d,%s,%s,%s,%.1f\n",
           o.kernel, o.page, o.src, rows, dim, W.bytes, o.split, g_nthreads, cpus,
           weights, o.prefetch, o.flush_mb,
           iters, med, p10, p90, best, gb_med, gb_best, huge_kb, max_err, argmax_ok,
           o.schedule, thr_busy, thr_rows, imbalance);

    // 最適化で計算が消されないように結果を使う
    volatile float keep = out[0] + g_sink[0];
    (void)keep;
    return 0;
}
