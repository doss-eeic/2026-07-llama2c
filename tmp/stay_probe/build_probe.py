"""Generate a temporary diagnostic variant; leave the normal kernels unchanged."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
s = (ROOT / 'run_stay2.c').read_text()
s = s.replace('    int execution_cpu;', '''    int execution_cpu;
    unsigned long long trace_dispatch;
    int trace_stage, trace_count;
    struct { unsigned long long dispatch, start, finish, cpu_ns; int stage, rows; } trace[1024];''')
s = s.replace('static int stay_diagnostics;', 'static int stay_diagnostics;\nstatic int probe_trace, probe_output_only;')
s = s.replace('    stay_diagnostics = getenv', '''    probe_trace = getenv("LLAMA_PROBE_TRACE") != NULL;
    probe_output_only = getenv("LLAMA_PROBE_OUTPUT_ONLY") != NULL;
    stay_diagnostics = getenv''')
s = s.replace('        for (int i = worker->begin; i < worker->end; i++)', '''        unsigned long long probe_start = 0, probe_cpu_start = 0;
        struct timespec probe_cpu;
        if (probe_trace) {
            probe_start = profile_now();
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &probe_cpu);
            probe_cpu_start = (unsigned long long)probe_cpu.tv_sec * 1000000000ULL + probe_cpu.tv_nsec;
        }
        for (int i = worker->begin; i < worker->end; i++)''')
s = s.replace('        stay_check(pthread_mutex_lock(&stay_mutex));\n        if (--stay_pending', '''        if (probe_trace && worker->trace_count < 1024) {
            unsigned long long finish = profile_now();
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &probe_cpu);
            int n = worker->trace_count++;
            worker->trace[n].dispatch = worker->trace_dispatch;
            worker->trace[n].start = probe_start;
            worker->trace[n].finish = finish;
            worker->trace[n].cpu_ns = (unsigned long long)probe_cpu.tv_sec * 1000000000ULL + probe_cpu.tv_nsec - probe_cpu_start;
            worker->trace[n].stage = worker->trace_stage;
            worker->trace[n].rows = worker->end - worker->begin;
        }
        stay_check(pthread_mutex_lock(&stay_mutex));
        if (--stay_pending''')
s = s.replace('    stay_pending = rule->count;', '    stay_pending = rule->count;\n    unsigned long long probe_dispatch = probe_trace ? profile_now() : 0;')
s = s.replace('        worker->ready = 1;', '        worker->trace_dispatch = probe_dispatch;\n        worker->trace_stage = stage_id;\n        worker->ready = 1;')
s = s.replace('        stay_check(pthread_cond_destroy(&stay_workers[i].wake));', '''        for (int n = 0; n < stay_workers[i].trace_count; n++) {
            StayWorker *w = &stay_workers[i];
            fprintf(stderr, "probe,%d,%s,%d,%llu,%llu,%llu,%llu\\n", w->cpu,
                    stay_stages[w->trace[n].stage], w->trace[n].rows,
                    w->trace[n].dispatch, w->trace[n].start,
                    w->trace[n].finish, w->trace[n].cpu_ns);
        }
        stay_check(pthread_cond_destroy(&stay_workers[i].wake));''')
s = s.replace('    stay_dispatch("default", -2, stay_layers_count, stay_layers, &job);', '''    if (!probe_output_only) stay_dispatch("default", -2, stay_layers_count, stay_layers, &job);
    else memcpy(transformer->state.x, transformer->weights.token_embedding_table + token * transformer->config.dim,
                transformer->config.dim * sizeof(float));''')
# Optional intervention: give SMT P workers a larger share of output rows.
s = s.replace('    int trace_stage, trace_count;', '    int trace_stage, trace_count;')
s = s.replace('    stay_pending = rule->count;', '\n'.join([
    '    int probe_weight = getenv("LLAMA_PROBE_P_WEIGHT") ? atoi(getenv("LLAMA_PROBE_P_WEIGHT")) : 1;',
    '    if (probe_weight < 1 || probe_weight > 8) stay_fail("invalid probe P weight");',
    '    int probe_weights[STAY_MAX_CPUS], probe_total = 0, probe_prefix = 0;',
    '    const char *probe_p_cpus = getenv("LLAMA_PROBE_P_CPUS");',
    '    for (int i = 0; i < rule->count; i++) {',
    '        char needle[32]; snprintf(needle, sizeof(needle), ",%d,", rule->cpus[i]);',
    '        int is_p = probe_p_cpus && strstr(probe_p_cpus, needle);',
    '        probe_weights[i] = stage_id == 8 && is_p ? probe_weight : 1;',
    '        probe_total += probe_weights[i];',
    '    }',
    '    stay_pending = rule->count;']))
s = s.replace('        worker->begin = (int)((long long)rows * rank / rule->count);',
              '        worker->begin = (int)((long long)rows * probe_prefix / probe_total);')
s = s.replace('        worker->end = (int)((long long)rows * (rank + 1) / rule->count);',
              '        probe_prefix += probe_weights[rank];\n        worker->end = (int)((long long)rows * probe_prefix / probe_total);')

p = ROOT / 'tmp/stay_probe/probe.c'
p.write_text(s)
subprocess.run(['gcc', '-Ofast', '-march=native', '-pthread', '-DLLAMA_PROFILE', '-I', str(ROOT), str(p), '-lm', '-o', str(p.with_suffix(''))], check=True)
