// memlab: why is a file mmap faster than an anonymous 4KB copy of the same bytes under WSL2?
// Places the same 60MB (the model file) in several kinds of memory and measures, single thread:
//   seq  : sequential read bandwidth (what llama2.c's matmul does)
//   walk : one load per 4KB page in random page order, with the touched cache lines (~1MB)
//          already in L2. Data comes from cache, but ~15k pages >> TLB, so the time is
//          dominated by address translation (page walks). Same guest page size => same guest
//          page-table work, so a difference points at the 2nd (host/EPT) translation stage.
// usage: memlab base <model> <scratch_dir> <rounds>
//        memlab recache <model> <rounds>
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>

#define PAGE 4096
#define HUGE (2u << 20)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
static double med(double* v, int n) { qsort(v, n, sizeof(double), cmpd); return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2; }

static size_t N;          // bytes measured (page multiple)
static uint32_t* perm;    // random page order, shared by all regions
static size_t npages;
volatile float sink;

static double seq_gbs(const float* p) {
    double t[9];
    for (int r = 0; r < 9; r++) {
        // 4 independent AVX accumulators so the loop is limited by memory, not by add latency
        double t0 = now();
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t i = 0; i + 32 <= N / 4; i += 32) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16));
            a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
        }
        t[r] = now() - t0;
        float o[8]; _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        sink += o[0];
    }
    return N / med(t, 9) / 1e9;
}

static double walk_ns(const float* p) {
    double t[9];
    for (int r = 0; r < 12; r++) {
        double t0 = now(); float s = 0;
        for (size_t i = 0; i < npages; i++) {
            size_t pg = perm[i];
            s += p[pg * (PAGE / 4) + (pg % 64) * 16]; // different cache line per page, avoids set conflicts
        }
        double dt = now() - t0; sink += s;
        if (r >= 3) t[r - 3] = dt; // first passes bring the lines into cache
    }
    return med(t, 9) / npages * 1e9;
}

static char* anon(size_t n, int advice) {
    char* b = mmap(NULL, n + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) { perror("mmap"); exit(1); }
    char* a = (char*)(((uintptr_t)b + HUGE - 1) & ~(uintptr_t)(HUGE - 1));
    if (madvise(a, n, advice)) { perror("madvise"); exit(1); }
    return a;
}

static char* map_file(const char* path, int prot, int flags) {
    int fd = open(path, prot & PROT_WRITE ? O_RDWR : O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    char* p = mmap(NULL, N, prot, flags, fd, 0);
    if (p == MAP_FAILED) { perror("mmap file"); exit(1); }
    close(fd);
    return p;
}

static double resident_frac(const char* path) {
    int fd = open(path, O_RDONLY);
    char* p = mmap(NULL, N, PROT_READ, MAP_SHARED, fd, 0);
    unsigned char* v = malloc(npages);
    mincore(p, N, v);
    size_t r = 0; for (size_t i = 0; i < npages; i++) r += v[i] & 1;
    munmap(p, N); close(fd); free(v);
    return (double)r / npages;
}

static void drop_cache(const char* path) {
    int fd = open(path, O_RDONLY);
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

static void read_all(const char* path) { // repopulates the page cache with freshly allocated pages
    int fd = open(path, O_RDONLY); char* buf = malloc(1 << 20);
    while (read(fd, buf, 1 << 20) > 0) {}
    free(buf); close(fd);
}

static long huge_kb(void) {
    FILE* f = fopen("/proc/self/smaps_rollup", "r"); char l[256]; long kb = -1;
    while (f && fgets(l, sizeof l, f)) if (sscanf(l, "AnonHugePages: %ld", &kb) == 1) break;
    if (f) fclose(f);
    return kb;
}

typedef struct { const char* name; const float* p; double seq[64], walk[64]; } Region;

static void measure(Region* R, int nr, int rounds) {
    for (int r = 0; r < rounds; r++) {
        int order[16];
        for (int i = 0; i < nr; i++) order[i] = i;
        for (int i = nr - 1; i > 0; i--) { int j = rand() % (i + 1), t = order[i]; order[i] = order[j]; order[j] = t; }
        for (int k = 0; k < nr; k++) { Region* g = &R[order[k]]; g->seq[r] = seq_gbs(g->p); g->walk[r] = walk_ns(g->p); }
    }
    printf("%-28s %10s %10s %12s\n", "region", "seq GB/s", "walk ns", "walk min-max");
    for (int i = 0; i < nr; i++) {
        double w[64]; memcpy(w, R[i].walk, sizeof w);
        double s = med(R[i].seq, rounds), m = med(w, rounds);
        printf("%-28s %10.1f %10.2f %6.2f-%-6.2f\n", R[i].name, s, m, w[0], w[rounds - 1]);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see header\n"); return 1; }
    const char* model = argv[2];
    struct stat st; stat(model, &st);
    N = st.st_size / PAGE * PAGE; npages = N / PAGE;
    perm = malloc(npages * sizeof *perm);
    srand(12345);
    for (size_t i = 0; i < npages; i++) perm[i] = i;
    for (size_t i = npages - 1; i > 0; i--) { size_t j = rand() % (i + 1); uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }

    if (!strcmp(argv[1], "base")) {
        int rounds = atoi(argv[4]);
        char copy_path[4096]; snprintf(copy_path, sizeof copy_path, "%s/model_copy.bin", argv[3]);
        printf("model page cache resident before start: %.0f%%\n", 100 * resident_frac(model));
        char* fold = map_file(model, PROT_READ, MAP_PRIVATE);
        for (size_t i = 0; i < N; i += PAGE) sink += fold[i];                 // fault it in (page cache already warm)
        char* a4k = anon(N, MADV_NOHUGEPAGE); memcpy(a4k, fold, N);
        char* thp = anon(N, MADV_HUGEPAGE); memcpy(thp, fold, N);
        int fd = open(copy_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);        // brand new page cache pages
        if (write(fd, fold, N) != (ssize_t)N) { perror("write"); return 1; }
        close(fd);
        char* fnew = map_file(copy_path, PROT_READ, MAP_PRIVATE);
        int mfd = memfd_create("w", 0);                                        // shmem (tmpfs) pages, 4KB
        if (write(mfd, fold, N) != (ssize_t)N) { perror("write memfd"); return 1; }
        char* shm = mmap(NULL, N, PROT_READ, MAP_SHARED, mfd, 0);
        char* cow = map_file(model, PROT_READ | PROT_WRITE, MAP_PRIVATE);       // file mapping, but CoW anon pages
        for (size_t i = 0; i < N; i += PAGE) cow[i] = cow[i];
        for (size_t i = 0; i < N; i += PAGE) sink += fnew[i] + shm[i];
        printf("AnonHugePages: %ld kB (expect ~%zu for the THP region)\n\n", huge_kb(), N / 1024 / 2048 * 2048);
        Region R[] = {
            {"file mmap (model, cached)", (float*)fold}, {"file mmap (new copy)", (float*)fnew},
            {"shmem memfd 4KB", (float*)shm}, {"anon 4KB copy", (float*)a4k},
            {"anon CoW of file mapping", (float*)cow}, {"anon THP 2MB copy", (float*)thp}};
        measure(R, 6, rounds);
        unlink(copy_path);
    } else if (!strcmp(argv[1], "recache")) {
        // same file, same mmap code: only the physical pages under the page cache change
        int rounds = atoi(argv[3]);
        char* f = map_file(model, PROT_READ, MAP_PRIVATE);
        for (size_t i = 0; i < N; i += PAGE) sink += f[i];
        Region before = {"model mmap: current cache", (const float*)f};
        measure(&before, 1, rounds);
        munmap(f, N);
        drop_cache(model);
        printf("  after FADV_DONTNEED resident: %.0f%%\n", 100 * resident_frac(model));
        read_all(model);
        printf("  after re-read resident: %.0f%%\n", 100 * resident_frac(model));
        f = map_file(model, PROT_READ, MAP_PRIVATE);
        for (size_t i = 0; i < N; i += PAGE) sink += f[i];
        Region after = {"model mmap: fresh cache", (const float*)f};
        measure(&after, 1, rounds);
        // reuse: free the cache again and immediately take anon pages (likely the same just-freed pages,
        // before free page reporting hands them back to the host)
        char* src = anon(N, MADV_HUGEPAGE); memcpy(src, f, N);
        munmap(f, N);
        drop_cache(model);
        char* reuse = anon(N, MADV_NOHUGEPAGE); memcpy(reuse, src, N);
        Region r2 = {"anon 4KB on just-freed pages", (const float*)reuse};
        measure(&r2, 1, rounds);
        read_all(model); // leave the page cache populated
    }
    return 0;
}
