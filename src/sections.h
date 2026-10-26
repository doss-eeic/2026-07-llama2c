/* 出力層とそれ以外の CPU の指定（-b / -c）。
 *
 * 出力層以外（body）と出力層（cls）で、使う CPU とスレッド数を分ける（両方を一緒に指定する）。
 * taskset や OMP_PLACES はプロセス全体に効くので、区間ごとには変えられない。
 * そこで区間に入るときに OpenMP の各スレッドを sched_setaffinity で区間の CPU に固定し直し、
 * 区間の中の parallel 領域は num_threads(区間の CPU の数) で動かす。固定し直すのは区間が変わるときだけ（1 トークンに 2 回）。
 * -b / -c がないときは何もしない（これまでと同じ動き）。
 *
 * スレッド i が毎回同じ OS スレッドになる前提なので、OMP_PROC_BIND=false（または未指定）で使う。
 * OMP_PROC_BIND が有効だと、libgomp がスレッドを並べ替えて、固定し直した CPU がずれる。
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#define SECTION_MAX_CPUS 256

typedef enum { SEC_BODY = 0, SEC_CLS = 1 } SectionId;

typedef struct {
    int cpus[SECTION_MAX_CPUS]; // スレッド i を cpus[i] に固定する
    int n;
} CpuList;

#ifdef _OPENMP
static int section_on = 0;            // -b / -c を指定したら 1
static SectionId section_cur = SEC_BODY;
static CpuList sections[2];           // [SEC_BODY], [SEC_CLS]

// OpenMP の parallel 領域のスレッド数。-b / -c のときは今の区間の CPU の数
static inline int section_num_threads(void) {
    return section_on ? sections[section_cur].n : omp_get_max_threads();
}
#endif

#if defined __linux__ && defined _OPENMP
#include <sched.h>

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
            if (c < 0 || c >= CPU_SETSIZE || list->n >= SECTION_MAX_CPUS) { goto bad; }
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

// body_cpus / cls_cpus は -b / -c（どちらも NULL か、どちらも指定）
static void setup_sections(const char* body_cpus, const char* cls_cpus) {
    if ((body_cpus == NULL) != (cls_cpus == NULL)) {
        fprintf(stderr, "-b and -c must be given together\n"); exit(EXIT_FAILURE);
    }
    if (body_cpus == NULL) { return; } // これまでと同じ OpenMP
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        fprintf(stderr, "-b / -c need OMP_PROC_BIND=false\n"); exit(EXIT_FAILURE);
    }
    omp_set_dynamic(0); // スレッド数を勝手に減らされないようにする
    parse_cpu_list(body_cpus, &sections[SEC_BODY]);
    parse_cpu_list(cls_cpus, &sections[SEC_CLS]);
    section_on = 1;
}

static int section_pinned = -1; // 今どの区間の CPU に固定しているか

// 区間に入る。-b / -c のときは、前と違う区間なら各スレッドをその区間の CPU に固定し直す
static void enter_section(SectionId id) {
    section_cur = id;
    if (!section_on || section_pinned == (int)id) { return; }
    section_pinned = id;
    const CpuList* sec = &sections[id];
    #pragma omp parallel num_threads(sec->n)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(sec->cpus[omp_get_thread_num()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}
#else
static void setup_sections(const char* body_cpus, const char* cls_cpus) {
    if (body_cpus != NULL || cls_cpus != NULL) {
        fprintf(stderr, "-b / -c require Linux and OpenMP\n"); exit(EXIT_FAILURE);
    }
}
static void enter_section(SectionId id) { (void)id; }
#endif
