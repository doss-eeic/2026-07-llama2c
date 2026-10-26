/* One OpenMP team for the whole generation/chat loop. No pthread workers. */
#include <errno.h>
#include <limits.h>

typedef void (*OmpRows)(int begin, int end, void* context);
static int omp_pool_enabled, omp_pool_running, omp_pool_output_stage;
static int omp_pool_layer_threads = 1, omp_pool_output_threads = 1, omp_pool_total = 1;

static int omp_pool_setting(const char* name, int fallback, int minimum) {
    const char* value = getenv(name);
    if (!value) return fallback;
    char* end;
    errno = 0;
    long n = strtol(value, &end, 10);
    if (errno || end == value || *end || n < minimum || n > INT_MAX) {
        fprintf(stderr, "Invalid %s\n", name); exit(EXIT_FAILURE);
    }
    return (int)n;
}

#if defined(_OPENMP) && !defined(_WIN32)
#include <omp.h>
static int omp_pool_schedule, omp_pool_chunk;
static struct {
    OmpRows rows;
    void* context;
    int count, active, next, stop;
} omp_job;

static void omp_pool_work(int tid) {
    int rows = omp_job.count, active = omp_job.active;
    if (omp_pool_schedule == omp_sched_static) {
        if (omp_pool_chunk == 0) {
            omp_job.rows(rows * tid / active, rows * (tid + 1) / active, omp_job.context);
        } else {
            for (int begin = tid * omp_pool_chunk; begin < rows; begin += active * omp_pool_chunk) {
                int end = begin + omp_pool_chunk;
                omp_job.rows(begin, end < rows ? end : rows, omp_job.context);
            }
        }
    } else {
        for (;;) {
            int begin, end;
            if (omp_pool_schedule == omp_sched_dynamic) {
                #pragma omp atomic capture
                { begin = omp_job.next; omp_job.next += omp_pool_chunk; }
                end = begin + omp_pool_chunk;
            } else {
                // Reserve a decreasing chunk without serializing workers on a lock.
                // A failed CAS updates begin to the current queue position.
                begin = __atomic_load_n(&omp_job.next, __ATOMIC_RELAXED);
                do {
                    int chunk = (rows - begin) / active;
                    if (chunk < omp_pool_chunk) chunk = omp_pool_chunk;
                    end = begin + chunk;
                } while (begin < rows && !__atomic_compare_exchange_n(
                    &omp_job.next, &begin, end, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            }
            if (begin >= rows) break;
            omp_job.rows(begin, end < rows ? end : rows, omp_job.context);
        }
    }
}

/* Coordinator and workers encounter these exact same two barrier constructs. */
static int omp_pool_exchange(void) {
    #pragma omp barrier
    int stop = omp_job.stop;
    int tid = omp_get_thread_num();
    if (!stop && tid < omp_job.active) omp_pool_work(tid);
    #pragma omp barrier
    return stop;  // captured before the coordinator can publish the next job
}

static void omp_pool_rows(int count, OmpRows rows, void* context) {
    omp_job.rows = rows;
    omp_job.context = context;
    omp_job.count = count;
    omp_job.active = omp_pool_output_stage ? omp_pool_output_threads : omp_pool_layer_threads;
    omp_job.next = omp_job.stop = 0;
    omp_pool_exchange();
}

static int omp_pool_rank(void) { return omp_get_thread_num(); }
#else
static void omp_pool_rows(int count, OmpRows rows, void* context) { rows(0, count, context); }
static int omp_pool_rank(void) { return 0; }
#endif

static void omp_pool_init(void) {
    omp_pool_enabled = omp_pool_setting("PERSISTENT_OPENMP", 0, 0);
    if (omp_pool_enabled > 1) {
        fprintf(stderr, "PERSISTENT_OPENMP must be 0 or 1\n"); exit(EXIT_FAILURE);
    }
    if (!omp_pool_enabled) return;
#if defined(_OPENMP) && !defined(_WIN32)
    if (omp_get_dynamic()) {
        fprintf(stderr, "Persistent OpenMP requires OMP_DYNAMIC=FALSE\n"); exit(EXIT_FAILURE);
    }
    omp_pool_layer_threads = omp_get_max_threads();
    omp_pool_output_threads = omp_pool_setting("OUTPUT_THREADS", omp_pool_layer_threads, 1);
    omp_pool_total = omp_pool_layer_threads > omp_pool_output_threads ? omp_pool_layer_threads : omp_pool_output_threads;
    omp_sched_t schedule;
    omp_get_schedule(&schedule, &omp_pool_chunk);
    omp_pool_schedule = schedule & ~omp_sched_monotonic;
    if (omp_pool_schedule != omp_sched_static && omp_pool_schedule != omp_sched_dynamic
            && omp_pool_schedule != omp_sched_guided) {
        fprintf(stderr, "Persistent OpenMP supports static/dynamic/guided schedules\n"); exit(EXIT_FAILURE);
    }
    if (omp_pool_schedule != omp_sched_static && omp_pool_chunk < 1) omp_pool_chunk = 1;
#else
    fprintf(stderr, "PERSISTENT_OPENMP requires an OpenMP build on Linux/macOS\n"); exit(EXIT_FAILURE);
#endif
}

static void omp_pool_run(void (*run)(void*), void* context) {
    if (!omp_pool_enabled) { run(context); return; }
#if defined(_OPENMP) && !defined(_WIN32)
    #pragma omp parallel num_threads(omp_pool_total)
    {
        if (omp_get_num_threads() != omp_pool_total) {
            fprintf(stderr, "Could not create the requested persistent OpenMP team\n"); exit(EXIT_FAILURE);
        }
        if (omp_get_thread_num() == 0) {
            omp_pool_running = 1;
            fprintf(stderr, "persistent OpenMP: team=%d layer=%d output=%d\n",
                    omp_pool_total, omp_pool_layer_threads, omp_pool_output_threads);
            run(context);
            omp_job.stop = 1;
            omp_pool_exchange();
        } else {
            while (!omp_pool_exchange()) {}
        }
    }
    omp_pool_running = omp_pool_output_stage = 0;
#endif
}
