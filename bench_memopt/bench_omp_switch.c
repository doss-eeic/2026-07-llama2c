// 層と出力層でスレッドの構成を切り替えたときの、並列区間の起動・同期のコストを測る。
// 方針: プロジェクト/output-layer-memory-optimization.md 3.2 節
//
// 中身の計算はせず（各スレッドが 1 回書き込むだけ）、オーバーヘッドだけを測る。
// 1 "token" = 同期 49 回（run.c の 49 parallel regions/token と同じ回数）。うち 48 回が層、1 回が出力層。
//
//   regions L C : 層は num_threads(L) の並列区間、出力層は num_threads(C) の並列区間を開く
//                 （L = C ならスレッド数は変わらない。L < C なら毎トークン 2 回チームの大きさが変わる）
//   idle L      : 毎回プール全体（OMP_NUM_THREADS）で並列区間を開き、層ではスレッド L 以上は何もしない
//                 （区間の開き方は全体で同じ。オーバーヘッドの測定上は regions N N と同じ）
//   barriers    : 1 トークンを 1 つの並列区間にし、区間の中でバリアを 49 回置く
//
// ビルド: gcc -O2 -fopenmp bench_memopt/bench_omp_switch.c -o bench_omp_switch
// 例（基準機、層は P コア 4 スレッド、出力層は P + E の 8 スレッド）:
//   export OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=threads OMP_NUM_THREADS=8
//   taskset -c 0-3,4,5,8,9 ./bench_omp_switch regions 4 8

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <omp.h>

#define TOKENS 20000
#define WARMUP 1000
#define LAYER_SYNCS 48

static volatile int sink[64 * 16];

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static void region(int nth) {
    #pragma omp parallel num_threads(nth)
    {
        sink[omp_get_thread_num() * 16]++;
    }
}

static void region_idle(int active) {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        if (t < active) { sink[t * 16]++; }
    }
}

static void token_regions(int l, int c) {
    for (int r = 0; r < LAYER_SYNCS; r++) { region(l); }
    region(c);
}

static void token_idle(int l) {
    for (int r = 0; r < LAYER_SYNCS; r++) { region_idle(l); }
    region_idle(omp_get_max_threads());
}

static void token_barriers(void) {
    #pragma omp parallel
    {
        for (int r = 0; r < LAYER_SYNCS + 1; r++) {
            sink[omp_get_thread_num() * 16]++;
            #pragma omp barrier
        }
    }
}

static void show_team(int nth) {
    int cpu[64];
    int n = 0;
    #pragma omp parallel num_threads(nth)
    {
        cpu[omp_get_thread_num()] = sched_getcpu();
        #pragma omp single
        n = omp_get_num_threads();
    }
    printf("team of %d -> cpus:", n);
    for (int i = 0; i < n; i++) { printf(" %d", cpu[i]); }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_omp_switch regions L C | idle L | barriers\n");
        return 1;
    }
    const char* mode = argv[1];
    int l = argc > 2 ? atoi(argv[2]) : omp_get_max_threads();
    int c = argc > 3 ? atoi(argv[3]) : omp_get_max_threads();
    int kind = strcmp(mode, "regions") == 0 ? 0 : strcmp(mode, "idle") == 0 ? 1 :
               strcmp(mode, "barriers") == 0 ? 2 : -1;
    if (kind < 0) { fprintf(stderr, "unknown mode: %s\n", mode); return 1; }

    if (kind == 0) { show_team(l); show_team(c); } else { show_team(omp_get_max_threads()); }

    double* v = malloc(TOKENS * sizeof(double));
    for (int t = -WARMUP; t < TOKENS; t++) {
        double t0 = now_us();
        if (kind == 0) { token_regions(l, c); }
        else if (kind == 1) { token_idle(l); }
        else { token_barriers(); }
        if (t >= 0) { v[t] = now_us() - t0; }
    }
    qsort(v, TOKENS, sizeof(double), cmp_double);
    printf("%s: per token (%d syncs): median %.1f us, p90 %.1f us\n",
           mode, LAYER_SYNCS + 1, v[TOKENS / 2], v[TOKENS * 9 / 10]);
    free(v);
    return 0;
}
