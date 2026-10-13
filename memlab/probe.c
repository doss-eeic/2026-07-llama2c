// probe: mmap existing files (their page cache pages stay where they are) and measure, interleaved:
//   seq    : pure streaming read (4 AVX accumulators)
//   matvec : llama2.c's matmul (n=288) over the whole file, i.e. the real access pattern
//   walk   : one load per 4KB page, random order, data in cache => translation cost
// usage: probe <rounds> <file>...
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define PAGE 4096
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
static double med(double* v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }
static size_t N, npages; static unsigned* perm; volatile float sink;

static double seq(const float* p) {
    double t[9];
    for (int r = 0; r < 9; r++) {
        double t0 = now(); __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t i = 0; i + 32 <= N / 4; i += 32) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i)); a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16)); a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
        }
        t[r] = now() - t0; float o[8]; _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))); sink += o[0];
    }
    return N / med(t, 9) / 1e9;
}
static void matmul(float* xout, const float* x, const float* w, int n, int d) { // same as run.c
    for (int i = 0; i < d; i++) { float val = 0.0f; for (int j = 0; j < n; j++) val += w[i * n + j] * x[j]; xout[i] = val; }
}
static double matvec(const float* p) {
    const int n = 288; int d = (int)(N / 4 / n);
    static float x[288], out[60000]; for (int i = 0; i < n; i++) x[i] = 0.01f * i;
    double t[9];
    for (int r = 0; r < 9; r++) { double t0 = now(); matmul(out, x, p + 7, n, d); t[r] = now() - t0; sink += out[d / 2]; }
    return (double)d * n * 4 / med(t, 9) / 1e9;
}
static double walk(const float* p) {
    double t[9];
    for (int r = 0; r < 12; r++) {
        double t0 = now(); float s = 0;
        for (size_t i = 0; i < npages; i++) { size_t pg = perm[i]; s += p[pg * (PAGE / 4) + (pg % 64) * 16]; }
        double dt = now() - t0; sink += s; if (r >= 3) t[r - 3] = dt;
    }
    return med(t, 9) / npages * 1e9;
}
int main(int argc, char** argv) {
    int rounds = atoi(argv[1]), nf = argc - 2;
    struct stat st; stat(argv[2], &st); N = (st.st_size - 64) / PAGE * PAGE; npages = N / PAGE;
    perm = malloc(npages * sizeof *perm); srand(1);
    for (size_t i = 0; i < npages; i++) perm[i] = i;
    for (size_t i = npages - 1; i > 0; i--) { size_t j = rand() % (i + 1); unsigned t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    const float* m[16]; double S[16][64], M[16][64], W[16][64];
    for (int f = 0; f < nf; f++) {
        int fd = open(argv[2 + f], O_RDONLY);
        m[f] = mmap(NULL, N + 64, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        for (size_t i = 0; i < N; i += PAGE) sink += m[f][i / 4];
    }
    for (int r = 0; r < rounds; r++)
        for (int k = 0; k < nf; k++) { int f = (k + r) % nf; S[f][r] = seq(m[f]); M[f][r] = matvec(m[f]); W[f][r] = walk(m[f]); }
    printf("%-60s %9s %11s %8s\n", "file", "seq GB/s", "matvec GB/s", "walk ns");
    for (int f = 0; f < nf; f++)
        printf("%-60s %9.1f %11.1f %8.2f\n", argv[2 + f], med(S[f], rounds), med(M[f], rounds), med(W[f], rounds));
    return 0;
}
