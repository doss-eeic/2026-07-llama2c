/* dev/sakurada d7e1a399: int8 screening, then fp32 recomputation of argmax candidates. */
#include <float.h>

static inline float dot_f32(const float* w, const float* x, int n) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) { val += w[j] * x[j]; }
    return val;
}

static struct {
    int enabled, gs;
    int8_t* q;
    float* s;
    float* lo;
    void* buf;
    size_t buf_size;
} cls;

static void setup_classifier(float* wcls, int dim, int vocab, int gs, int huge_pages) {
#if defined __linux__
    cls.gs = gs;
    const size_t huge = 2 * 1024 * 1024;
    int ng = dim / gs;
    size_t qbytes = (size_t)vocab * dim, sbytes = (size_t)vocab * ng * sizeof(float);
    size_t sofs = (qbytes + 63) / 64 * 64;
    cls.buf_size = sofs + sbytes + huge;
    char* base = mmap(NULL, cls.buf_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    if (madvise(base, cls.buf_size, huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
        perror("madvise"); exit(EXIT_FAILURE);
    }
    char* start = (char*)(((uintptr_t)base + huge - 1) & ~(uintptr_t)(huge - 1));
    cls.buf = base;
    cls.q = (int8_t*)start;
    cls.s = (float*)(start + sofs);
    cls.lo = malloc(vocab * sizeof(float));
    if (!cls.lo) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < vocab; i++) {
        for (int g = 0; g < ng; g++) {
            const float* w = wcls + (size_t)i * dim + g * gs;
            float wmax = 0.0f;
            for (int j = 0; j < gs; j++) { float a = fabsf(w[j]); if (a > wmax) wmax = a; }
            float scale = wmax / 127.0f;
            cls.s[(size_t)i * ng + g] = scale;
            for (int j = 0; j < gs; j++) {
                cls.q[(size_t)i * dim + g * gs + j] = scale > 0 ? (int8_t)roundf(w[j] / scale) : 0;
            }
        }
    }
    cls.enabled = 1;
#else
    fprintf(stderr, "-a 1 requires Linux\n"); exit(EXIT_FAILURE);
#endif
}

static void classifier_int8(float* logits, float* x, float* wcls, int n, int d) {
#if defined __linux__
    int gs = cls.gs, ng = n / gs;
    float A[ng];
    for (int g = 0; g < ng; g++) {
        float a = 0.0f;
        for (int j = 0; j < gs; j++) { a += fabsf(x[g * gs + j]); }
        A[g] = a;
    }
    // Finite sentinel: -Ofast permits optimizing away comparisons with infinities.
    float L = -FLT_MAX;
    #pragma omp parallel for schedule(runtime) reduction(max:L) num_threads(section_num_threads())
    for (int i = 0; i < d; i++) {
        const int8_t* q = cls.q + (size_t)i * n;
        const float* sc = cls.s + (size_t)i * ng;
        float approx = 0.0f, bound = 0.0f;
        for (int g = 0; g < ng; g++) {
            float dot = 0.0f;
            for (int j = 0; j < gs; j++) { dot += (float)q[g * gs + j] * x[g * gs + j]; }
            approx += dot * sc[g];
            bound += sc[g] * A[g];
        }
        // Same rounding margin as Sakurada's implementation.
        float err = 0.51f * bound + 1e-6f;
        logits[i] = approx + err;
        cls.lo[i] = approx - err;
        if (cls.lo[i] > L) { L = cls.lo[i]; }
    }
    for (int i = 0; i < d; i++) {
        logits[i] = logits[i] >= L ? dot_f32(wcls + (size_t)i * n, x, n) : -FLT_MAX;
    }
#else
    fprintf(stderr, "-a 1 requires Linux\n"); exit(EXIT_FAILURE);
#endif
}

static void free_classifier(void) {
#if defined __linux__
    if (cls.buf) { munmap(cls.buf, cls.buf_size); }
#endif
    free(cls.lo);
    cls.enabled = 0;
    cls.buf = NULL;
    cls.lo = NULL;
}
