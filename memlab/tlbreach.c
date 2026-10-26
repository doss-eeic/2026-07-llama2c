// tlbreach: does a 2MB guest huge page really give one 2MB TLB entry inside WSL2?
// One load per 4KB page (at a hashed cache line) over the first K pages in random order; the touched lines (K*64B)
// stay in cache, so time/access ~ translation cost. Natively, THP should stay flat (~cache
// latency) for all K up to 60MB, while 4KB pages jump once K exceeds the TLB (~2k entries).
// If THP also jumps, the effective TLB entry is still 4KB: the host (2nd stage) maps the
// guest memory with 4KB pages, and a TLB entry can be no larger than the smaller of the two.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#define PAGE 4096
#define HUGE (2u << 20)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
volatile float sink;
static char* anon(size_t n, int advice) {
    char* b = mmap(NULL, n + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char* a = (char*)(((uintptr_t)b + HUGE - 1) & ~(uintptr_t)(HUGE - 1));
    madvise(a, n, advice); memset(a, 1, n); return a;
}
// Which cache line of the page to touch. Must not be a simple function of the page number
// (e.g. pg % 64): on physically contiguous memory (a real 2MB page) that maps all touched lines
// to only 64 L2 sets, and the resulting cache misses look like TLB misses.
static size_t line_of(size_t pg) {
    uint64_t z = pg * 0x9E3779B97F4A7C15ull;
    z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 32;
    return z % 64;
}
static double walk(const char* p, const uint32_t* perm, size_t k) {
    double t[15];
    for (int r = 0; r < 18; r++) {
        double t0 = now(); float s = 0;
        for (size_t i = 0; i < k; i++) { size_t pg = perm[i]; s += ((const float*)(p + pg * PAGE))[line_of(pg) * 16]; }
        double dt = now() - t0; sink += s; if (r >= 3) t[r - 3] = dt;
    }
    qsort(t, 15, sizeof(double), cmpd); return t[7] / k * 1e9;
}
int main(void) {
    const size_t npages = 14848; // 58MB, the size of the model
    char* small = anon(npages * PAGE, MADV_NOHUGEPAGE);
    char* huge = anon(npages * PAGE, MADV_HUGEPAGE);
    size_t ks[] = {64, 256, 512, 1024, 2048, 4096, 8192, 14848};
    printf("%8s %8s %12s %12s\n", "pages", "MB", "4KB ns", "THP ns");
    for (int j = 0; j < 8; j++) {
        size_t k = ks[j];
        uint32_t* perm = malloc(k * sizeof *perm);
        for (size_t i = 0; i < k; i++) perm[i] = i;
        srand(7); for (size_t i = k - 1; i > 0; i--) { size_t x = rand() % (i + 1); uint32_t t = perm[i]; perm[i] = perm[x]; perm[x] = t; }
        double a = walk(small, perm, k), b = walk(huge, perm, k), a2 = walk(small, perm, k), b2 = walk(huge, perm, k);
        printf("%8zu %8.1f %12.2f %12.2f\n", k, k * 4.0 / 1024, (a + a2) / 2, (b + b2) / 2);
        free(perm);
    }
    FILE* f = fopen("/proc/self/smaps_rollup", "r"); char l[256];
    while (fgets(l, sizeof l, f)) if (!strncmp(l, "AnonHugePages", 13)) printf("%s", l);
    return 0;
}
