// chunkshuffle: does the *order* of physical pages matter, with the set of pages held fixed?
// A memfd (shmem, 4KB pages; their physical location is fixed once populated) is mapped
// several times into virtual memory: once in file order ("identity") and once per chunk size
// with the chunks in random order. Every view reads exactly the same physical pages and uses
// 4KB PTEs, so the only difference is how physically scattered consecutive virtual pages are.
// The chunk size at which bandwidth recovers hints at the mechanism:
//   ~8-16KB      DRAM row buffer (row open/close)
//   64KB-2MB     prefetcher / channel interleaving / cache coloring
//   no change    which pages (not their order) matters
// usage: chunkshuffle <rounds> [nofill]
//   by default ~200MB of anonymous memory is touched first to use up the fragmented free
//   memory, so the memfd is populated from large free blocks (likely physically contiguous).
//   "nofill" populates the memfd directly, i.e. from the fragments.
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096
#define POOL (64u << 20) // about the size of stories15M.bin

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
static double med(double* v, int n) { double c[64]; memcpy(c, v, n * sizeof(double)); qsort(c, n, sizeof(double), cmpd); return c[n / 2]; }
volatile float sink;

static double seq(const float* p) {
    double t[7];
    for (int r = 0; r < 7; r++) {
        double t0 = now(); __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t i = 0; i + 32 <= POOL / 4; i += 32) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i)); a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16)); a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
        }
        t[r] = now() - t0; float o[8]; _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))); sink += o[0];
    }
    return POOL / med(t, 7) / 1e9;
}

static void matmul(float* xout, const float* x, const float* w, int n, int d) { // same as run.c
    for (int i = 0; i < d; i++) { float val = 0.0f; for (int j = 0; j < n; j++) val += w[i * n + j] * x[j]; xout[i] = val; }
}

static double matvec(const float* p) {
    const int n = 288; const int d = POOL / 4 / n;
    static float x[288], out[POOL / 4 / 288]; for (int i = 0; i < n; i++) x[i] = 0.01f * i;
    double t[7];
    for (int r = 0; r < 7; r++) { double t0 = now(); matmul(out, x, p, n, d); t[r] = now() - t0; sink += out[d / 2]; }
    return (double)d * n * 4 / med(t, 7) / 1e9;
}

static const float* make_view(int fd, size_t chunk, unsigned seed) {
    char* base = mmap(NULL, POOL, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("reserve"); exit(1); }
    size_t n = POOL / chunk;
    size_t* perm = malloc(n * sizeof *perm);
    for (size_t i = 0; i < n; i++) perm[i] = i;
    if (seed) { srand(seed); for (size_t i = n - 1; i > 0; i--) { size_t j = rand() % (i + 1), t = perm[i]; perm[i] = perm[j]; perm[j] = t; } }
    for (size_t i = 0; i < n; i++) {
        void* p = mmap(base + i * chunk, chunk, PROT_READ, MAP_SHARED | MAP_FIXED | MAP_POPULATE, fd, perm[i] * chunk);
        if (p == MAP_FAILED) { perror("mmap chunk"); exit(1); }
    }
    free(perm);
    return (const float*)base;
}

int main(int argc, char** argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 10;
    int fill = !(argc > 2 && !strcmp(argv[2], "nofill"));
    char* filler = NULL; size_t fsz = 200u << 20;
    if (fill) { // eat the fragmented free memory first
        filler = mmap(NULL, fsz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(filler, fsz, MADV_NOHUGEPAGE);
        memset(filler, 1, fsz);
    }
    int fd = memfd_create("pool", 0);
    if (ftruncate(fd, POOL)) { perror("ftruncate"); return 1; }
    float* init = mmap(NULL, POOL, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    for (size_t i = 0; i < POOL / 4; i++) init[i] = 1e-3f * (float)(i % 1000); // allocates the pages in file order
    munmap(init, POOL);
    if (filler) munmap(filler, fsz);

    size_t chunks[] = {0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1u << 20, 2u << 20};
    const int nv = sizeof chunks / sizeof chunks[0];
    const float* view[16]; double S[16][64], M[16][64];
    for (int v = 0; v < nv; v++) view[v] = chunks[v] ? make_view(fd, chunks[v], 1000 + v) : make_view(fd, PAGE, 0);

    printf("pool=%uMB populated %s, rounds=%d\n", POOL >> 20, fill ? "after using up fragments (filler)" : "directly (nofill)", rounds);
    int order[16];
    for (int r = 0; r < rounds; r++) {
        for (int v = 0; v < nv; v++) order[v] = v;
        srand(r + 1); for (int i = nv - 1; i > 0; i--) { int j = rand() % (i + 1), t = order[i]; order[i] = order[j]; order[j] = t; }
        for (int k = 0; k < nv; k++) { int v = order[k]; S[v][r] = seq(view[v]); M[v][r] = matvec(view[v]); }
    }
    double s0 = med(S[0], rounds), m0 = med(M[0], rounds);
    printf("%-18s %9s %8s %12s %8s\n", "view", "seq GB/s", "vs id", "matvec GB/s", "vs id");
    for (int v = 0; v < nv; v++) {
        char name[32];
        if (!chunks[v]) snprintf(name, sizeof name, "identity");
        else if (chunks[v] < (1u << 20)) snprintf(name, sizeof name, "shuffle %zuKB", chunks[v] >> 10);
        else snprintf(name, sizeof name, "shuffle %zuMB", chunks[v] >> 20);
        double s = med(S[v], rounds), m = med(M[v], rounds);
        printf("%-18s %9.2f %7.3fx %12.2f %7.3fx\n", name, s, s / s0, m, m / m0);
    }
    return 0;
}
