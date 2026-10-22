/* Persistent, CPU-pinned workers. Included only by the LLAMA_STAY build.
 * STAY_CPUS lists layer CPUs first, then output-only CPUs.
 * One coordinator calls forward at a time; model state is not reentrant.
 */
#if defined(_OPENMP)
#error "Build stay with -pthread, without -fopenmp"
#endif
#include <pthread.h>
#include <errno.h>
#ifdef LLAMA_EXPERIMENT
#include <stdatomic.h>
static atomic_int stay_row_cursor;
#endif
#ifdef __linux__
#include <sched.h>
#elif !defined(STAY_TEST_UNPINNED)
#error "Stay CPU pinning requires Linux"
#endif

#define STAY_MAX_WORKERS 128
typedef struct {
    pthread_t thread;
    pthread_cond_t wake;
    int cpu, rank, ready;
} StayWorker;
static StayWorker stay_workers[STAY_MAX_WORKERS];
static int stay_initialized, stay_count, stay_layer_count, stay_p_weight;
static int stay_output_mode, stay_pending, stay_stopping, stay_started;
static _Thread_local int stay_rank;
static void (*stay_function)(void *);
static void *stay_context;
static pthread_mutex_t stay_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t stay_done = PTHREAD_COND_INITIALIZER;
#ifdef __linux__
static pthread_barrier_t stay_barrier;
#else
/* Portable barrier for correctness tests on macOS; production uses Linux. */
static pthread_mutex_t stay_barrier_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t stay_barrier_cond = PTHREAD_COND_INITIALIZER;
static unsigned stay_arrived, stay_generation;
#endif

static void stay_fail(const char *message) {
    fprintf(stderr, "stay: %s\n", message);
    exit(EXIT_FAILURE);
}
static void stay_check(int error) {
    if (error) { errno = error; perror("stay pthread"); exit(EXIT_FAILURE); }
}
static void stay_sync(void) {
    if (stay_output_mode) return;
#ifdef __linux__
    int error = pthread_barrier_wait(&stay_barrier);
    if (error != PTHREAD_BARRIER_SERIAL_THREAD) stay_check(error);
#else
    stay_check(pthread_mutex_lock(&stay_barrier_mutex));
    unsigned generation = stay_generation;
    if (++stay_arrived == (unsigned)stay_layer_count) {
        stay_arrived = 0;
        stay_generation++;
        stay_check(pthread_cond_broadcast(&stay_barrier_cond));
    } else {
        while (generation == stay_generation)
            stay_check(pthread_cond_wait(&stay_barrier_cond, &stay_barrier_mutex));
    }
    stay_check(pthread_mutex_unlock(&stay_barrier_mutex));
#endif
}
static int stay_boundary(int rows, int rank) {
    int total = stay_layer_count, prefix = rank;
    if (stay_output_mode) {
        total = stay_layer_count * stay_p_weight + stay_count - stay_layer_count;
        prefix = rank <= stay_layer_count ? rank * stay_p_weight
                 : stay_layer_count * stay_p_weight + rank - stay_layer_count;
    }
    return (int)((long long)rows * prefix / total);
}
#define STAY_SERIAL if (stay_rank == 0)
#define STAY_SYNC() stay_sync()
#define STAY_BEGIN(rows) stay_boundary((rows), stay_rank)
#define STAY_END(rows) stay_boundary((rows), stay_rank + 1)

static void *stay_worker(void *argument) {
    StayWorker *worker = argument;
    stay_rank = worker->rank;
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(worker->cpu, &mask);
    stay_check(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask));
    if (sched_getcpu() != worker->cpu) stay_fail("CPU pin verification failed");
#endif
    stay_check(pthread_mutex_lock(&stay_mutex));
    stay_started++;
    stay_check(pthread_cond_signal(&stay_done));
    for (;;) {
        while (!worker->ready && !stay_stopping)
            stay_check(pthread_cond_wait(&worker->wake, &stay_mutex));
        if (stay_stopping) break;
        worker->ready = 0;
        stay_check(pthread_mutex_unlock(&stay_mutex));
        stay_function(stay_context);
        stay_check(pthread_mutex_lock(&stay_mutex));
        if (--stay_pending == 0) stay_check(pthread_cond_signal(&stay_done));
    }
    stay_check(pthread_mutex_unlock(&stay_mutex));
    return NULL;
}
static void stay_dispatch(void (*function)(void *), void *context, int output) {
    stay_check(pthread_mutex_lock(&stay_mutex));
    stay_output_mode = output;
#ifdef LLAMA_EXPERIMENT
    atomic_store_explicit(&stay_row_cursor, 0, memory_order_relaxed);
#endif
    stay_function = function;
    stay_context = context;
    stay_pending = output ? stay_count : stay_layer_count;
    for (int i = 0; i < (output ? stay_count : stay_layer_count); i++) {
        stay_workers[i].ready = 1;
        stay_check(pthread_cond_signal(&stay_workers[i].wake));
    }
    while (stay_pending) stay_check(pthread_cond_wait(&stay_done, &stay_mutex));
    stay_check(pthread_mutex_unlock(&stay_mutex));
}
static void stay_shutdown(void) {
    stay_check(pthread_mutex_lock(&stay_mutex));
    stay_stopping = 1;
    for (int i = 0; i < stay_count; i++)
        stay_check(pthread_cond_signal(&stay_workers[i].wake));
    stay_check(pthread_mutex_unlock(&stay_mutex));
    for (int i = 0; i < stay_count; i++) {
        stay_check(pthread_join(stay_workers[i].thread, NULL));
        stay_check(pthread_cond_destroy(&stay_workers[i].wake));
    }
#ifdef __linux__
    stay_check(pthread_barrier_destroy(&stay_barrier));
#endif
}
static int stay_setting(const char *name, int fallback, int maximum) {
    const char *value = getenv(name);
    if (!value) return fallback;
    char *end;
    errno = 0;
    long number = strtol(value, &end, 10);
    if (errno || end == value || *end || number < 1 || number > maximum)
        stay_fail(name);
    return (int)number;
}
static void stay_init(void) {
    const char *cursor = getenv("STAY_CPUS");
    if (!cursor || !*cursor) stay_fail("set STAY_CPUS to an ordered CPU list");
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) stay_fail("sched_getaffinity failed");
#endif
    for (;;) {
        char *end;
        errno = 0;
        long cpu = strtol(cursor, &end, 10);
        if (errno || end == cursor || cpu < 0 || cpu >= 1024 ||
            (*end && *end != ',') || stay_count == STAY_MAX_WORKERS)
            stay_fail("invalid STAY_CPUS");
#ifdef __linux__
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed))
            stay_fail("STAY_CPUS contains an unavailable CPU");
#endif
        for (int i = 0; i < stay_count; i++)
            if (stay_workers[i].cpu == cpu) stay_fail("duplicate STAY_CPUS entry");
        stay_workers[stay_count].cpu = (int)cpu;
        stay_workers[stay_count].rank = stay_count;
        stay_count++;
        if (!*end) break;
        cursor = end + 1;
    }
    stay_layer_count = stay_setting("STAY_LAYER_THREADS", stay_count, stay_count);
    stay_p_weight = stay_setting("STAY_P_WEIGHT", 2, 16);
#ifdef __linux__
    stay_check(pthread_barrier_init(&stay_barrier, NULL, stay_layer_count));
#endif
    for (int i = 0; i < stay_count; i++) {
        stay_check(pthread_cond_init(&stay_workers[i].wake, NULL));
        stay_check(pthread_create(&stay_workers[i].thread, NULL, stay_worker, &stay_workers[i]));
    }
    stay_check(pthread_mutex_lock(&stay_mutex));
    while (stay_started < stay_count) stay_check(pthread_cond_wait(&stay_done, &stay_mutex));
    stay_check(pthread_mutex_unlock(&stay_mutex));
    if (atexit(stay_shutdown)) stay_fail("atexit failed");
    stay_initialized = 1;
}
