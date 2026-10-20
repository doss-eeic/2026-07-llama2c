/* Profiling format restored from dev/wataru-2 (66870ea:run_detail.c).
 * Only stay rank zero records layer stages; coordinator records dispatch totals. */
// Flat JSONL records, buffered until shutdown. No I/O inside forward().
#ifdef LLAMA_PROFILE
typedef struct {
    const char *stage;
    int forward_id, token, layer, input_token;
    unsigned long long duration_ns;
} ProfileRecord;
typedef struct {
    int forward_id, pos, input_token, next_token;
} ProfileToken;
static ProfileRecord *profile_records;
static size_t profile_count, profile_capacity;
static ProfileToken *profile_tokens;
static size_t profile_token_count, profile_token_capacity;
static int profile_forward_id = -1, profile_input_token;
static char profile_model_json[512];

static unsigned long long profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void profile_reserve(int n_layers, int input_token) {
    size_t needed = profile_count + (size_t)n_layers * 16 + 8;
    if (needed > profile_capacity) {
        size_t capacity = needed * 2;
        ProfileRecord *records = realloc(profile_records, capacity * sizeof(*records));
        if (!records) { fprintf(stderr, "profile allocation failed!\n"); exit(EXIT_FAILURE); }
        profile_records = records;
        profile_capacity = capacity;
    }
    profile_forward_id++;
    profile_input_token = input_token;
}

static void profile_record(const char *stage, int token, int layer, unsigned long long start) {
    unsigned long long elapsed = profile_now() - start;
    profile_records[profile_count++] = (ProfileRecord){
        stage, profile_forward_id, token, layer, profile_input_token, elapsed};
}

static void profile_token(int pos, int token, int next_token) {
    if (profile_token_count == profile_token_capacity) {
        size_t capacity = (profile_token_capacity + 1) * 2;
        ProfileToken *tokens = realloc(profile_tokens, capacity * sizeof(*tokens));
        if (!tokens) { fprintf(stderr, "token log allocation failed!\n"); exit(EXIT_FAILURE); }
        profile_tokens = tokens;
        profile_token_capacity = capacity;
    }
    profile_tokens[profile_token_count++] = (ProfileToken){profile_forward_id, pos, token, next_token};
}

static void profile_dump(void) {
    const char *path = getenv("LLAMA_PROFILE_PATH");
    FILE *log = path ? fopen(path, "w") : stderr;
    if (!log) { perror("profile log"); exit(EXIT_FAILURE); }
    fprintf(log, "{\"event\":\"meta\",\"schema\":1,\"clock\":\"monotonic\",%s}\n",
            profile_model_json);
    for (size_t i = 0; i < profile_count; i++) {
        ProfileRecord *r = &profile_records[i];
        fprintf(log, "{\"event\":\"timing\",\"forward_id\":%d,\"pos\":%d,"
                     "\"input_token\":%d,\"layer\":%d,\"stage\":\"%s\",\"duration_ns\":%llu}\n",
                     r->forward_id, r->token, r->input_token, r->layer, r->stage, r->duration_ns);
    }
    for (size_t i = 0; i < profile_token_count; i++) {
        ProfileToken *r = &profile_tokens[i];
        fprintf(log, "{\"event\":\"token\",\"forward_id\":%d,\"pos\":%d,"
                     "\"input_token\":%d,\"next_token\":%d}\n",
                     r->forward_id, r->pos, r->input_token, r->next_token);
    }
    fprintf(log, "{\"event\":\"end\",\"forward_calls\":%d,\"timing_records\":%zu,"
                 "\"token_records\":%zu}\n", profile_forward_id + 1, profile_count, profile_token_count);
    // fclose flushes file output; stderr stays open and needs an explicit flush.
    int write_failed = ferror(log);
    int finish_result = path ? fclose(log) : fflush(log);
    if (write_failed || finish_result != 0) { perror("write profile log"); exit(EXIT_FAILURE); }
    free(profile_records);
    free(profile_tokens);
    profile_records = NULL;
    profile_tokens = NULL;
    profile_count = profile_capacity = profile_token_count = profile_token_capacity = 0;
    profile_forward_id = -1;
}
#define PROFILE_START(name) unsigned long long name = 0; STAY_SERIAL name = profile_now()
#define PROFILE_END(stage, layer, start) STAY_SERIAL profile_record(stage, pos, layer, start)
#else
#define profile_reserve(n_layers, token) ((void)0)
#define profile_dump() ((void)0)
#define profile_token(pos, token, next) ((void)0)
#define PROFILE_START(name) ((void)0)
#define PROFILE_END(stage, layer, start) ((void)0)
#endif
