/* 常駐ワーカー（-x pool）と、出力層とそれ以外の CPU の指定（-b / -c）。
 *
 * -x pool: OpenMP の parallel 領域の代わりに使う。
 *   使う CPU それぞれに 1 本ずつスレッドを作って固定し、以後は作り直しも移動もしない。
 *   parallel 領域 1 つは、今のチームのワーカーの受け取り口（別々のキャッシュライン）にだけ仕事を書いて渡し、
 *   そのワーカーが終わるまで待つ。待つときはスピンする（pause 命令）。
 *   OpenMP と違い、領域ごとにチームの情報を作り直したり、集中型のバリアで全員を待ち合わせたりしないので、
 *   小さな parallel 領域がたくさんある forward() では同期が軽くなる。
 *   スレッド数は OMP_NUM_THREADS ではなく、使う CPU の数で決まる。
 *
 * -b / -c: 出力層以外（body）と出力層（cls）で、使う CPU とスレッド数を分ける（両方を一緒に指定する）。
 *   pool: -b と -c の CPU すべてにワーカーを作り、区間ごとにその区間の CPU のワーカーだけをチームにする。
 *         メインスレッドは -b の先頭の CPU に固定し、自分の CPU が区間に入っていればその分を計算する。
 *   omp : 区間に入るたびに OpenMP の各スレッドを sched_setaffinity で区間の CPU に固定し直し、
 *         区間の中の parallel 領域は num_threads(区間の CPU の数) で動かす。
 *   -b / -c がないときは、起動時の CPU（taskset で指定したもの）を両方の区間に使う。
 *
 * どちらの場合も OMP_PROC_BIND=false（または未指定）で使う。
 * OMP_PROC_BIND が有効だと、libgomp が main() より前にメインスレッドを 1 つの CPU に固定して起動時の CPU がわからなくなり、
 * また OpenMP のスレッドの並びも変わって、固定し直した CPU がずれる。
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#define POOL_MAX_CPUS 256

typedef void (*PoolFn)(void* ctx, int k0, int k1, int rank); // [k0, k1) を計算する。rank は 0 ～ pool_threads() - 1

typedef enum { SEC_BODY = 0, SEC_CLS = 1 } SectionId;

typedef struct {
    int cpus[POOL_MAX_CPUS]; // スレッド i（rank i）を cpus[i] に固定する
    int n;
} CpuList;

static int pool_on = 0;               // -x pool なら 1
static int section_on = 0;            // -b / -c を指定したら 1
static SectionId section_cur = SEC_BODY;
static CpuList sections[2];           // [SEC_BODY], [SEC_CLS]

#ifdef _OPENMP
// OpenMP の parallel 領域のスレッド数。-b / -c のときは今の区間の CPU の数
static inline int section_num_threads(void) {
    return section_on ? sections[section_cur].n : omp_get_max_threads();
}
#endif

// pool_run() で今の区間を動かすスレッドの数（メインスレッドを含む）
static inline int pool_threads(void) { return sections[section_cur].n; }

#if defined __linux__
#include <pthread.h>
#include <sched.h>

// ワーカー 1 本分の受け取り口。メインスレッドが fn ～ rank を書いてから gen を増やす
typedef struct {
    _Alignas(64) unsigned gen; // 仕事を渡すたびに 1 増える
    PoolFn fn;
    void* ctx;
    int k0, k1, rank;
    int cpu;
} PoolWorker;

static PoolWorker pool_workers[2 * POOL_MAX_CPUS];
static int pool_n_workers = 0;
static PoolWorker* pool_team[2][POOL_MAX_CPUS];  // [区間][i] は sections[区間].cpus[i] のワーカー。NULL はメインスレッド
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

static void pin_set(cpu_set_t* set, int cpu) {
    CPU_ZERO(set);
    CPU_SET(cpu, set);
}

// "0,2,4-7" の形の CPU リストを読む
static void parse_cpu_list(const char* s, CpuList* list) {
    list->n = 0;
    const char* p = s;
    while (*p) {
        char* end;
        long a = strtol(p, &end, 10), b = a;
        if (end == p) { goto bad; }
        if (*end == '-') {
            p = end + 1;
            b = strtol(p, &end, 10);
            if (end == p || b < a) { goto bad; }
        }
        for (long c = a; c <= b; c++) {
            if (c < 0 || c >= CPU_SETSIZE || list->n >= POOL_MAX_CPUS) { goto bad; }
            for (int i = 0; i < list->n; i++) { if (list->cpus[i] == c) { goto bad; } }
            list->cpus[list->n++] = (int)c;
        }
        if (*end == ',') { end++; } else if (*end != '\0') { goto bad; }
        p = end;
    }
    if (list->n > 0) { return; }
bad:
    fprintf(stderr, "invalid CPU list (or a CPU appears twice): %s\n", s);
    exit(EXIT_FAILURE);
}

// メインスレッドを固定し、2 つの区間の CPU にワーカーを作る（プロセスが終わるまでスピンし続ける）
static void pool_start(void) {
    cpu_set_t set;
    int main_cpu = sections[SEC_BODY].cpus[0];
    pin_set(&set, main_cpu);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) { perror("sched_setaffinity"); exit(EXIT_FAILURE); }
    for (int id = 0; id < 2; id++) {
        for (int i = 0; i < sections[id].n; i++) {
            int cpu = sections[id].cpus[i];
            PoolWorker* wk = NULL;
            if (cpu != main_cpu) {
                for (int w = 0; w < pool_n_workers; w++) {
                    if (pool_workers[w].cpu == cpu) { wk = &pool_workers[w]; }
                }
                if (wk == NULL) {
                    // 最初からその CPU で動き出すように、作る前に固定する
                    wk = &pool_workers[pool_n_workers++];
                    wk->cpu = cpu;
                    pthread_attr_t attr;
                    pthread_t thread;
                    pin_set(&set, cpu);
                    pthread_attr_init(&attr);
                    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
                    if (pthread_create(&thread, &attr, pool_worker_main, wk) != 0) {
                        fprintf(stderr, "failed to start a worker on CPU %d\n", cpu); exit(EXIT_FAILURE);
                    }
                    pthread_attr_destroy(&attr);
                }
            }
            pool_team[id][i] = wk;
        }
    }
    pool_on = 1;
}

// workers は "omp" か "pool"。body_cpus / cls_cpus は -b / -c（どちらも NULL か、どちらも指定）
static void setup_workers(const char* workers, const char* body_cpus, const char* cls_cpus) {
    int pool = strcmp(workers, "pool") == 0;
    if ((body_cpus == NULL) != (cls_cpus == NULL)) {
        fprintf(stderr, "-b and -c must be given together\n"); exit(EXIT_FAILURE);
    }
    section_on = body_cpus != NULL;
    if (!section_on && !pool) { return; } // これまでと同じ OpenMP
#ifdef _OPENMP
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        fprintf(stderr, "-x pool and -b / -c need OMP_PROC_BIND=false\n"); exit(EXIT_FAILURE);
    }
    omp_set_dynamic(0); // スレッド数を勝手に減らされないようにする
#endif
    if (section_on) {
        parse_cpu_list(body_cpus, &sections[SEC_BODY]);
        parse_cpu_list(cls_cpus, &sections[SEC_CLS]);
    } else {
        cpu_set_t start;
        if (sched_getaffinity(0, sizeof(start), &start) != 0) { perror("sched_getaffinity"); exit(EXIT_FAILURE); }
        CpuList* all = &sections[SEC_BODY];
        all->n = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE && all->n < POOL_MAX_CPUS; cpu++) {
            if (CPU_ISSET(cpu, &start)) { all->cpus[all->n++] = cpu; }
        }
        sections[SEC_CLS] = *all;
    }
    if (pool) { pool_start(); }
}

static int section_pinned = -1; // omp で -b / -c のとき、今どの区間の CPU に固定しているか

// 区間に入る。omp で -b / -c のときは、前と違う区間なら各スレッドをその区間の CPU に固定し直す
static void enter_section(SectionId id) {
    section_cur = id;
    if (!section_on || pool_on || section_pinned == (int)id) { return; }
    section_pinned = id;
#ifdef _OPENMP
    const CpuList* sec = &sections[id];
    #pragma omp parallel num_threads(sec->n)
    {
        cpu_set_t set;
        pin_set(&set, sec->cpus[omp_get_thread_num()]);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
}

// 今の区間の CPU で [0, n) を均等に分けて fn を実行し、全部終わるまで待つ
static void pool_run(PoolFn fn, void* ctx, int n) {
    int nt = sections[section_cur].n, k0_main = 0, k1_main = 0, main_rank = -1, sent = 0;
    // 次の gen の書き込み（release）より前なので、ワーカーが足すのはこの 0 より後になる
    __atomic_store_n(&pool_done.n, 0, __ATOMIC_RELAXED);
    for (int i = 0; i < nt; i++) {
        int k0 = (int)((long)n * i / nt), k1 = (int)((long)n * (i + 1) / nt);
        PoolWorker* wk = pool_team[section_cur][i];
        if (wk == NULL) { k0_main = k0; k1_main = k1; main_rank = i; continue; }
        wk->fn = fn;
        wk->ctx = ctx;
        wk->k0 = k0;
        wk->k1 = k1;
        wk->rank = i;
        __atomic_store_n(&wk->gen, __atomic_load_n(&wk->gen, __ATOMIC_RELAXED) + 1, __ATOMIC_RELEASE);
        sent++;
    }
    // メインスレッドの CPU が今の区間にないときは、待つだけ
    if (main_rank >= 0) { fn(ctx, k0_main, k1_main, main_rank); }
    while (__atomic_load_n(&pool_done.n, __ATOMIC_ACQUIRE) < sent) { pool_relax(); }
}
#else
static void setup_workers(const char* workers, const char* body_cpus, const char* cls_cpus) {
    if (strcmp(workers, "pool") == 0 || body_cpus != NULL || cls_cpus != NULL) {
        fprintf(stderr, "-x pool and -b / -c require Linux\n"); exit(EXIT_FAILURE);
    }
}
static void enter_section(SectionId id) { (void)id; }
static void pool_run(PoolFn fn, void* ctx, int n) { fn(ctx, 0, n, 0); }
#endif
