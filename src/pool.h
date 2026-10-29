/* 常駐ワーカー（-x pool）: OpenMP の parallel 領域の代わりに使う。
 *
 * 起動時の CPU（taskset で指定したもの）それぞれに 1 本ずつスレッドを作って固定し、以後は作り直しも移動もしない。
 * メインスレッドは先頭の CPU に固定し、自分の分も計算する（rank 0）。
 * parallel 領域 1 つは、各ワーカーの受け取り口（別々のキャッシュライン）に仕事を書いて渡し、
 * 全員が終わるまで待つ。待つときはスピンする（pause 命令）。
 * OpenMP と違い、領域ごとにチームの情報を作り直したり、集中型のバリアで待ち合わせたりしないので、
 * 小さな parallel 領域がたくさんある forward() では同期が軽くなる。
 * スレッド数は OMP_NUM_THREADS ではなく、起動時の CPU の数で決まる。
 * OMP_PROC_BIND が有効だと、libgomp が main() より前にメインスレッドを 1 つの CPU に固定してしまい、
 * 起動時の CPU がわからなくなるので、OMP_PROC_BIND=false（または未指定）で使う。
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#define POOL_MAX_CPUS 256

typedef void (*PoolFn)(void* ctx, int k0, int k1, int rank); // [k0, k1) を計算する。rank は 0 ～ pool_n - 1

static int pool_on = 0; // -x pool なら 1
static int pool_n = 1;  // スレッド数（メインスレッドを含む）

#if defined __linux__
#include <pthread.h>
#include <sched.h>

// ワーカー 1 本分の受け取り口。メインスレッドが fn ～ k1 を書いてから gen を増やす
typedef struct {
    _Alignas(64) unsigned gen; // 仕事を渡すたびに 1 増える
    PoolFn fn;
    void* ctx;
    int k0, k1, rank;
} PoolWorker;

static PoolWorker pool_workers[POOL_MAX_CPUS];
static struct { _Alignas(64) int n; } pool_done; // 今の仕事を終えたワーカーの数

static inline void pool_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static void* pool_worker_main(void* arg) {
    PoolWorker* wk = arg;
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        while ((g = __atomic_load_n(&wk->gen, __ATOMIC_ACQUIRE)) == seen) { pool_relax(); }
        seen = g;
        wk->fn(wk->ctx, wk->k0, wk->k1, wk->rank);
        __atomic_fetch_add(&pool_done.n, 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

// メインスレッドを固定し、ワーカーを作る（プロセスが終わるまでスピンし続ける）
static void pool_start(void) {
#ifdef _OPENMP
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        fprintf(stderr, "-x pool needs OMP_PROC_BIND=false\n"); exit(EXIT_FAILURE);
    }
#endif
    cpu_set_t start, set;
    if (sched_getaffinity(0, sizeof(start), &start) != 0) { perror("sched_getaffinity"); exit(EXIT_FAILURE); }
    int n = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &start)) { continue; }
        if (n >= POOL_MAX_CPUS) { fprintf(stderr, "-x pool supports up to %d CPUs\n", POOL_MAX_CPUS); exit(EXIT_FAILURE); }
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (n == 0) {
            if (sched_setaffinity(0, sizeof(set), &set) != 0) { perror("sched_setaffinity"); exit(EXIT_FAILURE); }
        } else {
            // 最初からその CPU で動き出すように、作る前に固定する
            PoolWorker* wk = &pool_workers[n];
            wk->rank = n;
            pthread_attr_t attr;
            pthread_t thread;
            pthread_attr_init(&attr);
            pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
            if (pthread_create(&thread, &attr, pool_worker_main, wk) != 0) {
                fprintf(stderr, "failed to start a worker on CPU %d\n", cpu); exit(EXIT_FAILURE);
            }
            pthread_attr_destroy(&attr);
        }
        n++;
    }
    pool_n = n;
    pool_on = 1;
}

// [0, n) を全スレッドで均等に分けて fn を実行し、全部終わるまで待つ
static void pool_run(PoolFn fn, void* ctx, int n) {
    // 次の gen の書き込み（release）より前なので、ワーカーが足すのはこの 0 より後になる
    __atomic_store_n(&pool_done.n, 0, __ATOMIC_RELAXED);
    for (int r = 1; r < pool_n; r++) {
        PoolWorker* wk = &pool_workers[r];
        wk->fn = fn;
        wk->ctx = ctx;
        wk->k0 = (int)((long)n * r / pool_n);
        wk->k1 = (int)((long)n * (r + 1) / pool_n);
        __atomic_store_n(&wk->gen, __atomic_load_n(&wk->gen, __ATOMIC_RELAXED) + 1, __ATOMIC_RELEASE);
    }
    fn(ctx, 0, (int)((long)n / pool_n), 0);
    while (__atomic_load_n(&pool_done.n, __ATOMIC_ACQUIRE) < pool_n - 1) { pool_relax(); }
}
#else
static void pool_start(void) { fprintf(stderr, "-x pool requires Linux\n"); exit(EXIT_FAILURE); }
static void pool_run(PoolFn fn, void* ctx, int n) { fn(ctx, 0, n, 0); }
#endif
