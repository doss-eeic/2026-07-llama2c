# AI向けの最新引き継ぎ（2026-10-22・全14CPU探索後）

最新は [1〜14論理CPUの探索と5アプローチ比較](AI向け_20261022_14CPU探索.md)。
固定269型を全測定し、入れ子mask＋kernelの118,800候補を5手法×2seedで探索した。
ビーム成長も深さ14まで完走。全14人の16条件も独立確認し、計3,551測定を保存した。
今回の確認最速平均は前回と同じpthread層8→output12人（798.24 tok/s）。
人数モデルが比較的良く、L2特徴追加の予測上の利点は小予算では明確でなかった。
CPU代表交換の差・外れ値も残しており、同値化を性能上の完全一致とは扱わない。

前段は [480候補の構成探索と追加ベンチマーク](AI向け_20261022_構成探索.md)。
単体・複合比較、トポロジー予測探索、48候補の実測オンライン探索2通りと独立確認を行った。
最速平均はpthread P4E4の層8→output12人で824.03 tok/s。qkv-ffnブランチ比の小差は未確定。
探索方式は48/480候補のreplayで平均latency差1.73%、randomは4.58%。
16セル間隔の基準補正が順位を歪めた失敗も含めて記録した。

以下の2026-10-20要因比較も引き続き参照する。

[最適化の要因比較](AI向け_20261020_最適化要因比較.md)では、
共通カーネルでpthread/OpenMP、static/dynamic、QKV融合、FFN融合、1/4行処理を交差させ、
速度測定700回と詳細計測18回を行った。前段の「融合pthreadは未比較」は過去時点の説明。

- QKV/FFNの改善は両方式に有効。併用の相乗・重複は条件依存。
- 同条件の8→14切替はOpenMPで悪化、pthreadでは大きな悪化なく微改善傾向だった。
- 全14固定は遅いが、outputだけ14人を使う構成まで否定しない。
- 時間経過による速度低下があったため、フェーズを跨いだ絶対速度の比較を避ける。
- PythonのBenchmarkConfigで設定し、benchmark_factorial.pyから比較する。
- 前段の経緯は[CPU並列化メモ](AI向け_20261020_CPU並列化.md)に保存している。
- この機械は14論理CPU・12物理コア。CPU番号の意味は他の機械で再確認する。

以下は過去のコード例・記録。現在の実装はrun.cとsrc/、最新の測定は上記リンクを優先する。

---

```c
// ============================================================
// Token Embedding
// ============================================================

static void embedding_lookup(Transformer* t, int token) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    float* row = w->token_embedding_table + token * p->dim;
    memcpy(s->x, row, p->dim * sizeof(float));
}


// ============================================================
// Attention: RMSNorm
// ============================================================

static void attention_norm(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    rmsnorm(
        s->xb,
        s->x,
        w->rms_att_weight + l * dim,
        dim
    );
}


// ============================================================
// Attention: Q, K, V Projection
// ============================================================

static void qkv_projection(Transformer* t, int l, int pos) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int kv_dim = dim * p->n_kv_heads / p->n_heads;

    // K, V の出力先は KV Cache 内の現在位置
    int loff = l * p->seq_len * kv_dim;

    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // Q = Wq * xb
    matmul(
        s->q, s->xb,
        w->wq + l * dim * dim,
        dim, dim
    );

    // K = Wk * xb（KV Cacheに直接書き込む）
    matmul(
        s->k, s->xb,
        w->wk + l * dim * kv_dim,
        dim, kv_dim
    );

    // V = Wv * xb（KV Cacheに直接書き込む）
    matmul(
        s->v, s->xb,
        w->wv + l * dim * kv_dim,
        dim, kv_dim
    );
}


// ============================================================
// Attention: RoPE
// ============================================================

static void apply_rope(Transformer* t, int pos) {
    Config* p = &t->config;
    RunState* s = &t->state;

    int dim = p->dim;
    int head_size = dim / p->n_heads;
    int kv_dim = dim * p->n_kv_heads / p->n_heads;

    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;

        float freq = 1.0f / powf(
            10000.0f,
            head_dim / (float)head_size
        );

        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        // Query と Key に適用（Valueには適用しない）
        int rotn = i < kv_dim ? 2 : 1;

        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k;

            float v0 = vec[i];
            float v1 = vec[i + 1];

            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}


// ============================================================
// Attention: Score, Softmax, Weighted Sum
// ============================================================

static void compute_attention(Transformer* t, int l, int pos) {
    Config* p = &t->config;
    RunState* s = &t->state;

    int head_size = p->dim / p->n_heads;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int loff = l * p->seq_len * kv_dim;

    int h;

    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {

        float* q = s->q + h * head_size;
        float* att = s->att + h * p->seq_len;

        // 1. Attention Score
        for (int pos2 = 0; pos2 <= pos; pos2++) {
            float* k = s->key_cache
                     + loff
                     + pos2 * kv_dim
                     + (h / kv_mul) * head_size;

            float score = 0.0f;

            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }

            score /= sqrtf(head_size);
            att[pos2] = score;
        }

        // 2. Softmax
        softmax(att, pos + 1);

        // 3. Weighted Sum
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (int pos2 = 0; pos2 <= pos; pos2++) {
            float* v = s->value_cache
                     + loff
                     + pos2 * kv_dim
                     + (h / kv_mul) * head_size;

            float a = att[pos2];

            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}


// ============================================================
// Attention: Output Projection + Residual
// ============================================================

static void attention_output(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    // Wo
    matmul(
        s->xb2, s->xb,
        w->wo + l * dim * dim,
        dim, dim
    );

    // Residual
    for (int i = 0; i < dim; i++) {
        s->x[i] += s->xb2[i];
    }
}


// ============================================================
// FFN: RMSNorm
// ============================================================

static void ffn_norm(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;

    rmsnorm(
        s->xb,
        s->x,
        w->rms_ffn_weight + l * dim,
        dim
    );
}


// ============================================================
// FFN: W1, W3 Projection
// ============================================================

static void ffn_up_projection(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    // W1
    matmul(
        s->hb, s->xb,
        w->w1 + l * dim * hidden_dim,
        dim, hidden_dim
    );

    // W3
    matmul(
        s->hb2, s->xb,
        w->w3 + l * dim * hidden_dim,
        dim, hidden_dim
    );
}


// ============================================================
// FFN: SwiGLU
// ============================================================

static void ffn_swiglu(Transformer* t) {
    Config* p = &t->config;
    RunState* s = &t->state;

    for (int i = 0; i < p->hidden_dim; i++) {
        float val = s->hb[i];

        // SiLU
        val *= 1.0f / (1.0f + expf(-val));

        // Elementwise multiplication
        val *= s->hb2[i];

        s->hb[i] = val;
    }
}


// ============================================================
// FFN: W2 Projection + Residual
// ============================================================

static void ffn_down_projection(Transformer* t, int l) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    // W2
    matmul(
        s->xb, s->hb,
        w->w2 + l * dim * hidden_dim,
        hidden_dim, dim
    );

    // Residual
    for (int i = 0; i < dim; i++) {
        s->x[i] += s->xb[i];
    }
}


// ============================================================
// Final RMSNorm + Logits
// ============================================================

static float* final_projection(Transformer* t) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    rmsnorm(
        s->x, s->x,
        w->rms_final_weight,
        p->dim
    );

    matmul(
        s->logits, s->x,
        w->wcls,
        p->dim, p->vocab_size
    );

    return s->logits;
}


// Capture the end before printing so stage timings exclude log output.
static double measure_time(struct timespec start_time, const char* message,
                           int pos, int layer) {
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    if (layer >= 0) {
        fprintf(stderr, "[token:%d, layer:%d] %s: %.9f seconds\n",
                pos, layer, message, elapsed_time);
    } else {
        fprintf(stderr, "[token:%d] %s: %.9f seconds\n",
                pos, message, elapsed_time);
    }
    return elapsed_time;
}

float* forward(Transformer* t, int token, int pos) {
    struct timespec stage_start;
    double forward_elapsed = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    embedding_lookup(t, token);
    forward_elapsed += measure_time(stage_start, "001:embedding-lookup", pos, -1);

    for (int l = 0; l < t->config.n_layers; l++) {
        double layer_elapsed = 0.0;

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        attention_norm(t, l);
        layer_elapsed += measure_time(stage_start, "002:attention-norm", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        qkv_projection(t, l, pos);
        layer_elapsed += measure_time(stage_start, "003:qkv-projection", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        apply_rope(t, pos);
        layer_elapsed += measure_time(stage_start, "004:apply-rope", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        compute_attention(t, l, pos);
        layer_elapsed += measure_time(stage_start, "005:compute-attention", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        attention_output(t, l);
        layer_elapsed += measure_time(stage_start, "006:attention-output", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        ffn_norm(t, l);
        layer_elapsed += measure_time(stage_start, "007:ffn-norm", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        ffn_up_projection(t, l);
        layer_elapsed += measure_time(stage_start, "008:ffn-up-projection", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        ffn_swiglu(t);
        layer_elapsed += measure_time(stage_start, "009:ffn-swiglu", pos, l);

        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        ffn_down_projection(t, l);
        layer_elapsed += measure_time(stage_start, "010:ffn-down-projection", pos, l);

        // Sum compute durations rather than including time spent printing logs.
        forward_elapsed += layer_elapsed;
        fprintf(stderr, "[token:%d, layer:%d] 102:layer-compute-total: %.9f seconds\n",
                pos, l, layer_elapsed);
    }

    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    float* logits = final_projection(t);
    forward_elapsed += measure_time(stage_start, "011:final-projection", pos, -1);
    fprintf(stderr, "[token:%d] 103:forward-compute-total: %.9f seconds\n",
            pos, forward_elapsed);
    return logits;
}
```
llama2.cのforwardの部分を関数に分割して，各時間計測の関数を突っ込んだ。




## 学科PCの環境計測

```shell
denjo@EE00044:~/2026-07-llama2c$ uv run analyze/analyze.py
threads=1, run=1, layers=1332, tokens=222, n_layers=6
threads=1, run=2, layers=1332, tokens=222, n_layers=6
threads=1, run=3, layers=1332, tokens=222, n_layers=6
threads=1, run=4, layers=1332, tokens=222, n_layers=6
threads=1, run=5, layers=1332, tokens=222, n_layers=6
threads=2, run=1, layers=1332, tokens=222, n_layers=6
threads=2, run=2, layers=1332, tokens=222, n_layers=6
threads=2, run=3, layers=1332, tokens=222, n_layers=6
threads=2, run=4, layers=1332, tokens=222, n_layers=6
threads=2, run=5, layers=1332, tokens=222, n_layers=6
threads=4, run=1, layers=1332, tokens=222, n_layers=6
threads=4, run=2, layers=1332, tokens=222, n_layers=6
threads=4, run=3, layers=1332, tokens=222, n_layers=6
threads=4, run=4, layers=1332, tokens=222, n_layers=6
threads=4, run=5, layers=1332, tokens=222, n_layers=6
threads=8, run=1, layers=1332, tokens=222, n_layers=6
threads=8, run=2, layers=1332, tokens=222, n_layers=6
threads=8, run=3, layers=1332, tokens=222, n_layers=6
threads=8, run=4, layers=1332, tokens=222, n_layers=6
threads=8, run=5, layers=1332, tokens=222, n_layers=6

=== Median time per call (µs) ===
Stage                                  1 threads           2 threads           4 threads           8 threads
embedding-lookup                   0.577       0.462       0.708       0.706
attention-norm                     0.230       0.229       0.459       0.535
qkv-projection                   101.012      58.418     104.802     164.610
apply-rope                         2.354       2.316       5.761       6.770
compute-attention                 35.753      13.819      40.353      76.925
attention-output                  36.368      20.920      31.196      50.997
ffn-norm                           0.314       0.286       0.510       0.572
ffn-up-projection                147.292      81.400     126.516     166.076
ffn-swiglu                         0.612       0.557       1.401       1.710
ffn-down-projection               81.822      41.268      64.021      83.512
final-projection                2760.275    1344.282    2158.233    2293.962
layer-compute-total              405.978     219.220     372.102     550.877
forward-compute-total           5196.734    2660.061    4397.081    5611.854

=== Time per token (ms) ===
Stage                                  1 threads           2 threads           4 threads           8 threads
embedding-lookup                  0.0006      0.0005      0.0007      0.0007
attention-norm                    0.0014      0.0014      0.0028      0.0032
qkv-projection                    0.6061      0.3505      0.6288      0.9877
apply-rope                        0.0141      0.0139      0.0346      0.0406
compute-attention                 0.2145      0.0829      0.2421      0.4615
attention-output                  0.2182      0.1255      0.1872      0.3060
ffn-norm                          0.0019      0.0017      0.0031      0.0034
ffn-up-projection                 0.8837      0.4884      0.7591      0.9965
ffn-swiglu                        0.0037      0.0033      0.0084      0.0103
ffn-down-projection               0.4909      0.2476      0.3841      0.5011
final-projection                  2.7603      1.3443      2.1582      2.2940
layer-compute-total               2.4359      1.3153      2.2326      3.3053
forward-compute-total             5.1967      2.6601      4.3971      5.6119

=== Forward time breakdown (%) ===
Stage                                  1 threads           2 threads           4 threads           8 threads
embedding-lookup                    0.01        0.02        0.02        0.01
attention-norm                      0.03        0.05        0.06        0.06
qkv-projection                     11.66       13.18       14.30       17.60
apply-rope                          0.27        0.52        0.79        0.72
compute-attention                   4.13        3.12        5.51        8.22
attention-output                    4.20        4.72        4.26        5.45
ffn-norm                            0.04        0.06        0.07        0.06
ffn-up-projection                  17.01       18.36       17.26       17.76
ffn-swiglu                          0.07        0.13        0.19        0.18
ffn-down-projection                 9.45        9.31        8.74        8.93
final-projection                   53.12       50.54       49.08       40.88

=== Major component breakdown ===

Threads: 1
  Forward:   5.1967 ms (100%)
  Layers     2.4359 ms (46.87%)
  Embedding  0.0006 ms (0.01%)
  Final      2.7603 ms (53.12%)
  Difference: 0.000013 ms

Threads: 2
  Forward:   2.6601 ms (100%)
  Layers     1.3153 ms (49.45%)
  Embedding  0.0005 ms (0.02%)
  Final      1.3443 ms (50.54%)
  Difference: 0.000000 ms

Threads: 4
  Forward:   4.3971 ms (100%)
  Layers     2.2326 ms (50.77%)
  Embedding  0.0007 ms (0.02%)
  Final      2.1582 ms (49.08%)
  Difference: 0.005530 ms

Threads: 8
  Forward:   5.6119 ms (100%)
  Layers     3.3053 ms (58.90%)
  Embedding  0.0007 ms (0.01%)
  Final      2.2940 ms (40.88%)
  Difference: 0.011924 ms

=== Speedup ===
Threads        Layer speedup     Forward speedup
1                      1.000x              1.000x
2                      1.852x              1.954x
4                      1.091x              1.182x
8                      0.737x              0.926x
```


```shell
denjo@EE00044:~/2026-07-llama2c$ lscpu | grep -E 'Model name|CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core|Socket\(s\)'
CPU(s):                                  14
On-line CPU(s) list:                     0-13
Model name:                              Intel(R) Core(TM) Ultra 5 225U
Thread(s) per core:                      2
Core(s) per socket:                      12
Socket(s):                               1
CPU(s) scaling MHz:                      25%
NUMA node0 CPU(s):                       0-13
denjo@EE00044:~/2026-07-llama2c$ nproc
14
denjo@EE00044:~/2026-07-llama2c$ taskset -pc $$
pid 8429's current affinity list: 0-13
denjo@EE00044:~/2026-07-llama2c$ cat /sys/fs/cgroup/cpu.max
cat: /sys/fs/cgroup/cpu.max: No such file or directory
denjo@EE00044:~/2026-07-llama2c$ cat /sys/fs/cgroup/cpuset.cpus.effective
0-13
denjo@EE00044:~/2026-07-llama2c$ 
```

```shell
denjo@EE00044:~/2026-07-llama2c$ lscpu -e=CPU,CORE,MAXMHZ,MINMHZ,ONLINE
CPU CORE    MAXMHZ   MINMHZ ONLINE
  0    0 4800.0000 400.0000    yes
  1    0 4800.0000 400.0000    yes
  2    1 4800.0000 400.0000    yes
  3    1 4800.0000 400.0000    yes
  4    2 3800.0000 400.0000    yes
  5    3 3800.0000 400.0000    yes
  6    4 3800.0000 400.0000    yes
  7    5 3800.0000 400.0000    yes
  8    6 3800.0000 400.0000    yes
  9    7 3800.0000 400.0000    yes
 10    8 3800.0000 400.0000    yes
 11    9 3800.0000 400.0000    yes
 12   10 2400.0000 400.0000    yes
```

## llama2.c CPU並列化・コア配置実験 サマリー
```shell
denjo@EE00044:~/2026-07-llama2c$ uv run analyze/analyze_affinity.py
Benchmark directory: /home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533

auto_1     run=1  tokens=222  layers=1332  forward=4.9350 ms
auto_1     run=2  tokens=222  layers=1332  forward=4.7955 ms
auto_1     run=3  tokens=222  layers=1332  forward=4.7962 ms
auto_1     run=4  tokens=222  layers=1332  forward=4.8179 ms
auto_1     run=5  tokens=222  layers=1332  forward=4.8216 ms
auto_2     run=1  tokens=222  layers=1332  forward=2.3522 ms
auto_2     run=2  tokens=222  layers=1332  forward=2.9431 ms
auto_2     run=3  tokens=222  layers=1332  forward=2.7030 ms
auto_2     run=4  tokens=222  layers=1332  forward=2.9180 ms
auto_2     run=5  tokens=222  layers=1332  forward=3.4311 ms
auto_4     run=1  tokens=222  layers=1332  forward=1.7922 ms
auto_4     run=2  tokens=222  layers=1332  forward=1.7633 ms
auto_4     run=3  tokens=222  layers=1332  forward=1.7020 ms
auto_4     run=4  tokens=222  layers=1332  forward=1.7559 ms
auto_4     run=5  tokens=222  layers=1332  forward=1.7936 ms
auto_8     run=1  tokens=222  layers=1332  forward=4.9489 ms
auto_8     run=2  tokens=222  layers=1332  forward=4.6942 ms
auto_8     run=3  tokens=222  layers=1332  forward=4.9248 ms
auto_8     run=4  tokens=222  layers=1332  forward=4.6502 ms
auto_8     run=5  tokens=222  layers=1332  forward=4.9681 ms
P1         run=1  tokens=222  layers=1332  forward=4.6974 ms
P1         run=2  tokens=222  layers=1332  forward=4.8078 ms
P1         run=3  tokens=222  layers=1332  forward=4.8562 ms
P1         run=4  tokens=222  layers=1332  forward=4.8323 ms
P1         run=5  tokens=222  layers=1332  forward=4.8500 ms
P2         run=1  tokens=222  layers=1332  forward=2.8303 ms
P2         run=2  tokens=222  layers=1332  forward=2.0566 ms
P2         run=3  tokens=222  layers=1332  forward=2.8691 ms
P2         run=4  tokens=222  layers=1332  forward=2.2210 ms
P2         run=5  tokens=222  layers=1332  forward=2.3659 ms
P4_HT      run=1  tokens=222  layers=1332  forward=1.6283 ms
P4_HT      run=2  tokens=222  layers=1332  forward=1.6346 ms
P4_HT      run=3  tokens=222  layers=1332  forward=1.6160 ms
P4_HT      run=4  tokens=222  layers=1332  forward=1.6195 ms
P4_HT      run=5  tokens=222  layers=1332  forward=1.6276 ms
E2         run=1  tokens=222  layers=1332  forward=6.3293 ms
E2         run=2  tokens=222  layers=1332  forward=6.3395 ms
E2         run=3  tokens=222  layers=1332  forward=6.3066 ms
E2         run=4  tokens=222  layers=1332  forward=6.3346 ms
E2         run=5  tokens=222  layers=1332  forward=6.3286 ms
E4         run=1  tokens=222  layers=1332  forward=9.4365 ms
E4         run=2  tokens=222  layers=1332  forward=8.8080 ms
E4         run=3  tokens=222  layers=1332  forward=9.4516 ms
E4         run=4  tokens=222  layers=1332  forward=9.3593 ms
E4         run=5  tokens=222  layers=1332  forward=9.4278 ms
E8         run=1  tokens=222  layers=1332  forward=5.3710 ms
E8         run=2  tokens=222  layers=1332  forward=5.4787 ms
E8         run=3  tokens=222  layers=1332  forward=5.5543 ms
E8         run=4  tokens=222  layers=1332  forward=5.4629 ms
E8         run=5  tokens=222  layers=1332  forward=5.5448 ms
P2E2       run=1  tokens=222  layers=1332  forward=5.0928 ms
P2E2       run=2  tokens=222  layers=1332  forward=5.0482 ms
P2E2       run=3  tokens=222  layers=1332  forward=4.9594 ms
P2E2       run=4  tokens=222  layers=1332  forward=5.0256 ms
P2E2       run=5  tokens=222  layers=1332  forward=5.0986 ms

=====================================================================================
LLAMA2.C CPU AFFINITY BENCHMARK
=====================================================================================
Directory: /home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533

=== Measurement validation ===

auto_1     runs=5  tokens=222-222 layers=1332-1332
auto_2     runs=5  tokens=222-222 layers=1332-1332
auto_4     runs=5  tokens=222-222 layers=1332-1332
auto_8     runs=5  tokens=222-222 layers=1332-1332
P1         runs=5  tokens=222-222 layers=1332-1332
P2         runs=5  tokens=222-222 layers=1332-1332
P4_HT      runs=5  tokens=222-222 layers=1332-1332
E2         runs=5  tokens=222-222 layers=1332-1332
E4         runs=5  tokens=222-222 layers=1332-1332
E8         runs=5  tokens=222-222 layers=1332-1332
P2E2       runs=5  tokens=222-222 layers=1332-1332

=== Overall performance ===

Case         Threads   Layers(ms)   Final(ms)   Forward(ms)    Speedup
auto_1             1       2.2168      2.6007        4.8179     1.000x
auto_2             2       1.3787      1.5390        2.9180     1.651x
auto_4             4       0.9032      0.8597        1.7633     2.732x
auto_8             8       2.8820      2.0420        4.9248     0.978x
P1                 1       2.2051      2.6267        4.8323     0.997x
P2                 2       1.1655      1.2000        2.3659     2.036x
P4_HT              4       0.8324      0.7925        1.6276     2.960x
E2                 2       3.0146      3.3182        6.3293     0.761x
E4                 4       4.7527      4.6727        9.4278     0.511x
E8                 8       3.3606      2.1014        5.4787     0.879x
P2E2               4       2.6359      2.4115        5.0482     0.954x

=== Fastest configuration ===

Case: P4_HT
Threads: 4
CPU affinity: 0-3
Forward: 1.6276 ms/token
Speedup: 2.960x

=== Run-to-run variation ===

Case             Min(ms)    Median(ms)     Max(ms)    Range(%)
auto_1            4.7955        4.8179      4.9350        2.90
auto_2            2.3522        2.9180      3.4311       36.97
auto_4            1.7020        1.7633      1.7936        5.20
auto_8            4.6502        4.9248      4.9681        6.45
P1                4.6974        4.8323      4.8562        3.29
P2                2.0566        2.3659      2.8691       34.34
P4_HT             1.6160        1.6276      1.6346        1.14
E2                6.3066        6.3293      6.3395        0.52
E4                8.8080        9.4278      9.4516        6.83
E8                5.3710        5.4787      5.5543        3.35
P2E2              4.9594        5.0482      5.0986        2.76

=== Projection bottleneck ===

Case          Projection(ms)   Projection(%)    Final(%)   Layers(%)
auto_1                4.6100           95.68       54.01       45.98
auto_2                2.7825           95.36       52.75       47.24
auto_4                1.6732           94.89       48.75       51.23
auto_8                4.5149           91.66       41.46       58.52
P1                    4.6247           95.70       54.34       45.65
P2                    2.2739           96.02       50.67       49.31
P4_HT                 1.5390           94.63       48.74       51.24
E2                    5.9816           94.51       52.38       47.61
E4                    8.7805           93.12       49.58       50.41
E8                    4.8630           88.72       38.47       61.52
P2E2                  4.7000           93.10       47.62       52.36

=== Stage breakdown: P4_HT ===

Stage                     Per call (us)    Per token (ms)    Forward (%)
embedding                         0.306            0.0003           0.02
attention_norm                    0.179            0.0011           0.07
qkv                              37.600            0.2256          13.88
rope                              1.921            0.0115           0.71
compute_attention                11.517            0.0691           4.27
attention_output                 12.737            0.0764           4.71
ffn_norm                          0.218            0.0013           0.08
ffn_up                           49.333            0.2960          18.21
swiglu                            0.630            0.0038           0.23
ffn_down                         24.617            0.1477           9.07
final                           792.515            0.7925          48.74
layers                          138.736            0.8324          51.24
forward                        1627.623            1.6276         100.00

=== CPU affinity comparisons ===

      P2 -> E2       speedup=0.374x (2.3659 -> 6.3293 ms)
      P2 -> E4       speedup=0.251x (2.3659 -> 9.4278 ms)
      P2 -> E8       speedup=0.432x (2.3659 -> 5.4787 ms)
      P2 -> P4_HT    speedup=1.454x (2.3659 -> 1.6276 ms)
      P2 -> P2E2     speedup=0.469x (2.3659 -> 5.0482 ms)
      E2 -> E4       speedup=0.671x (6.3293 -> 9.4278 ms)
      E4 -> E8       speedup=1.721x (9.4278 -> 5.4787 ms)
  auto_2 -> P2       speedup=1.233x (2.9180 -> 2.3659 ms)
  auto_4 -> P2E2     speedup=0.349x (1.7633 -> 5.0482 ms)

=== Interpretation guide ===

P2 vs E2:
  Compare P-core and E-core performance at 2 threads.
E2 vs E4 vs E8:
  Evaluate scaling using homogeneous E-cores.
P2 vs P4_HT:
  Evaluate Hyper-Threading on the same physical P-cores.
P2 vs P2E2:
  Evaluate mixed P/E-core scheduling.
auto_2 vs P2:
  Evaluate explicit CPU affinity.

NOTE:
  These measurements exclude most logging overhead.
  Slowdowns alone do not establish DRAM bandwidth saturation.
  CPU frequency, contention and OpenMP overhead also matter.
  Medians are calculated independently per metric.

=== Saved files ===
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533/analysis/cases.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533/analysis/stages.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533/analysis/runs.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_175533/analysis/report.txt
```

### 1. 実験概要
- **目的**：LLM推論のボトルネック特定と、OpenMPのスレッド数・CPU配置による高速化の検証
- **CPU**：Intel Core Ultra 5 225U（P-core 2個、E-core 8個、LP E-core 2個）
- **モデル**：`stories15M.bin`
- **実行条件**：222トークン、6レイヤー、各構成5回測定
- **評価指標**：Forward計算時間（ms/token）の中央値

### 2. 実験結果

| CPU構成 | スレッド数 | 時間（ms/token） | 高速化率 |
|---|---:|---:|---:|
| P-core ×1 | 1 | 4.832 | 1.00倍 |
| P-core ×2 | 2 | 2.366 | 2.04倍 |
| **P-core ×2（HTあり）** | **4** | **1.628** | **2.96倍** |
| 自動配置 | 4 | 1.763 | 2.73倍 |
| E-core ×2 | 2 | 6.329 | 0.76倍 |
| E-core ×4 | 4 | 9.428 | 0.51倍 |
| E-core ×8 | 8 | 5.479 | 0.88倍 |
| P-core ×2 + E-core ×2 | 4 | 5.048 | 0.95倍 |

※高速化率は自動配置・1スレッド（4.818 ms/token）を基準とする。

**最速はP-core 2個の4論理スレッド（HT利用）で、約2.96倍の高速化を達成した。**

### 3. ボトルネック分析

最速構成でのForward実行時間の内訳：

| 処理 | 割合 |
|---|---:|
| Final Projection | 48.7% |
| FFN Up Projection | 18.2% |
| QKV Projection | 13.9% |
| FFN Down Projection | 9.1% |
| Attention Output Projection | 4.7% |
| その他 | 5.4% |

**行列・ベクトル積（Projection）が推論時間の約94.6%を占める。**

したがって、今後は `matmul()` の最適化が最も有効と考えられる。

### 4. L2キャッシュ構造の確認

今回の追加調査で、E-coreが2つのクラスターに分かれていることを確認した。

| CPU番号 | L2キャッシュ | 共有コア |
|---|---|---|
| 4〜7 | 2MB | E-core 4個 |
| 8〜11 | 2MB | E-core 4個 |

これにより、以下の結果を説明できる可能性がある。

- **E2 → E4：** 6.329 → 9.428 ms（約49%遅延）
- **E4 → E8：** 9.428 → 5.479 ms（約42%短縮）

E4では4コアが同じL2キャッシュを共有する一方、E8では2つのL2キャッシュを利用できる。

このため、**L2キャッシュの共有による競合がE4の性能低下に関係している可能性**がある。

ただし、キャッシュミスや帯域を直接測定したわけではないため、まだ仮説の段階である。

### 5. 考察と今後の方針

今回の実験から、以下が明らかになった。

1. **単純にスレッド数を増やしても高速化しない。** コアの種類と配置が重要である。
2. **P-core上のHyper-Threadingは有効。** P2からP4_HTで約1.45倍高速化した。
3. **P-coreとE-coreの混在は今回の均等分割では不利。** 異種コア間の処理速度差や同期コストが原因として考えられる。
4. **E-core間のL2キャッシュ共有も性能に影響している可能性がある。**
5. **全体の約95%をProjectionが占めるため、`matmul()` の最適化が最優先。**

**次の実験：** E-coreを同一L2クラスターに配置した場合（4,5,6,7）と、異なるL2クラスターに分散配置した場合（4,5,8,9）を比較する。

これにより、E-coreの性能低下がL2キャッシュ共有に起因するかを検証する。

---

**現時点の結論：** 今回のCPUでは、スレッド数よりもコア配置が推論性能に大きく影響する。今後は、キャッシュ構造を考慮したスレッド配置と、異種コア間の負荷分散を最適化する方向が有望である。

# L2キャッシュ周りの実測。
```shell
denjo@EE00044:~/2026-07-llama2c$ for cpu in {4..11}; do
    printf "CPU %2d: " "$cpu"
    for cache in /sys/devices/system/cpu/cpu${cpu}/cache/index*; do
    for cache in /sys/devices/system/cpu/cpu${cpu}/cache/index*; do
        if [[ $(cat "$cache/level") == "2" ]]; then
            printf "L2=%s, shared=%s" \\
                "$(cat "$cache/size")" \u_list")"
                "$(cat "$cache/shared_cpu_list")"
        fi
    done
    echo
    echo
done
CPU  4: L2=2048K, shared=4-7
CPU  5: L2=2048K, shared=4-7
CPU  6: L2=2048K, shared=4-7
CPU  7: L2=2048K, shared=4-7
CPU  8: L2=2048K, shared=8-11
CPU  9: L2=2048K, shared=8-11
CPU 10: L2=2048K, shared=8-11
CPU 11: L2=2048K, shared=8-11
```
## 実行
```shell
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="./stories15M.bin"
REPEATS=5

STEPS=256
TEMPERATURE=0
SEED=42
PROMPT="Once upon a time"

BENCH_ROOT="benchmark_affinity_stories15M"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${BENCH_ROOT}/${TIMESTAMP}_l2"

mkdir -p "$OUT_DIR"

CASES=(
    "P1|1|0"

    "E2|2|4,5"
    "E2_B|2|8,9"
    "E2_cross|2|4,8"

    "E4|4|4-7"
    "E4_B|4|8-11"
    "E4_cross|4|4,5,8,9"

    "E8|8|4-11"
)

# OpenMP版をビルド
make runomp CC=gcc

if ! ldd ./run | grep -Eq 'libgomp|libomp'; then
    echo "ERROR: OpenMP is not enabled"
    exit 1
fi

# 環境を保存
{
    date
    lscpu
    echo
    echo "Model: $MODEL"
    echo "Steps: $STEPS"
    echo "Seed: $SEED"
    echo "Repeats: $REPEATS"
} > "$OUT_DIR/environment.txt"

for ((r=1; r<=REPEATS; r++)); do
    for spec in "${CASES[@]}"; do

        IFS='|' read -r name threads cpus <<< "$spec"

        echo "Run $r/$REPEATS: $name"

        taskset -c "$cpus" \
            env \
                OMP_NUM_THREADS="$threads" \
                OMP_DYNAMIC=FALSE \
                OMP_PROC_BIND=TRUE \
                OMP_PLACES=threads \
            ./run "$MODEL" \
                -n "$STEPS" \
                -t "$TEMPERATURE" \
                -s "$SEED" \
                -i "$PROMPT" \
            > "$OUT_DIR/${name}_r${r}.out" \
            2> "$OUT_DIR/${name}_r${r}.log"

    done
done

echo "Benchmark completed: $OUT_DIR"

uv run analyze/analyze_affinity.py --dir "$OUT_DIR"
```
## 結果
```shell
P1         run=1  tokens=222  layers=1332  forward=5.1692 ms
P1         run=2  tokens=222  layers=1332  forward=4.9283 ms
P1         run=3  tokens=222  layers=1332  forward=5.1127 ms
P1         run=4  tokens=222  layers=1332  forward=5.0569 ms
P1         run=5  tokens=222  layers=1332  forward=5.0875 ms
E2         run=1  tokens=222  layers=1332  forward=6.6206 ms
E2         run=2  tokens=222  layers=1332  forward=6.5091 ms
E2         run=3  tokens=222  layers=1332  forward=6.2695 ms
E2         run=4  tokens=222  layers=1332  forward=6.1980 ms
E2         run=5  tokens=222  layers=1332  forward=6.2735 ms
E4         run=1  tokens=222  layers=1332  forward=5.9329 ms
E4         run=2  tokens=222  layers=1332  forward=6.2794 ms
E4         run=3  tokens=222  layers=1332  forward=7.3355 ms
E4         run=4  tokens=222  layers=1332  forward=7.3693 ms
E4         run=5  tokens=222  layers=1332  forward=6.8492 ms
E8         run=1  tokens=222  layers=1332  forward=6.0724 ms
E8         run=2  tokens=222  layers=1332  forward=5.5342 ms
E8         run=3  tokens=222  layers=1332  forward=5.7722 ms
E8         run=4  tokens=222  layers=1332  forward=5.4257 ms
E8         run=5  tokens=222  layers=1332  forward=5.8049 ms
E2_B       run=1  tokens=222  layers=1332  forward=6.1642 ms
E2_B       run=2  tokens=222  layers=1332  forward=6.5359 ms
E2_B       run=3  tokens=222  layers=1332  forward=6.3113 ms
E2_B       run=4  tokens=222  layers=1332  forward=6.3672 ms
E2_B       run=5  tokens=222  layers=1332  forward=6.4621 ms
E2_cross   run=1  tokens=222  layers=1332  forward=6.5402 ms
E2_cross   run=2  tokens=222  layers=1332  forward=6.5835 ms
E2_cross   run=3  tokens=222  layers=1332  forward=6.5024 ms
E2_cross   run=4  tokens=222  layers=1332  forward=6.6864 ms
E2_cross   run=5  tokens=222  layers=1332  forward=6.7736 ms
E4_B       run=1  tokens=222  layers=1332  forward=5.9520 ms
E4_B       run=2  tokens=222  layers=1332  forward=5.5680 ms
E4_B       run=3  tokens=222  layers=1332  forward=6.0754 ms
E4_B       run=4  tokens=222  layers=1332  forward=6.0042 ms
E4_B       run=5  tokens=222  layers=1332  forward=5.9944 ms
E4_cross   run=1  tokens=222  layers=1332  forward=4.6982 ms
E4_cross   run=2  tokens=222  layers=1332  forward=4.5678 ms
E4_cross   run=3  tokens=222  layers=1332  forward=4.7819 ms
E4_cross   run=4  tokens=222  layers=1332  forward=4.3804 ms
E4_cross   run=5  tokens=222  layers=1332  forward=4.5313 ms

=====================================================================================
LLAMA2.C CPU AFFINITY BENCHMARK
=====================================================================================
Directory: /home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_183251_l2

=== Measurement validation ===

P1         runs=5  tokens=222-222 layers=1332-1332
E2         runs=5  tokens=222-222 layers=1332-1332
E4         runs=5  tokens=222-222 layers=1332-1332
E8         runs=5  tokens=222-222 layers=1332-1332
E2_B       runs=5  tokens=222-222 layers=1332-1332
E2_cross   runs=5  tokens=222-222 layers=1332-1332
E4_B       runs=5  tokens=222-222 layers=1332-1332
E4_cross   runs=5  tokens=222-222 layers=1332-1332

=== Overall performance ===

Case         Threads   Layers(ms)   Final(ms)   Forward(ms)    Speedup
P1                 1       2.3644      2.7225        5.0875     1.000x
E2                 2       3.0441      3.2337        6.2735     0.811x
E4                 4       3.5701      3.2782        6.8492     0.743x
E8                 8       3.6615      2.0750        5.7722     0.881x
E2_B               2       3.1162      3.2191        6.3672     0.799x
E2_cross           2       3.1737      3.3715        6.5835     0.773x
E4_B               4       3.0935      2.8944        5.9944     0.849x
E4_cross           4       2.5393      2.0205        4.5678     1.114x

=== Fastest configuration ===

Case: E4_cross
Threads: 4
CPU affinity: 4,5,8,9
Forward: 4.5678 ms/token
Speedup: 1.114x

=== Run-to-run variation ===

Case             Min(ms)    Median(ms)     Max(ms)    Range(%)
P1                4.9283        5.0875      5.1692        4.73
E2                6.1980        6.2735      6.6206        6.74
E4                5.9329        6.8492      7.3693       20.97
E8                5.4257        5.7722      6.0724       11.20
E2_B              6.1642        6.3672      6.5359        5.84
E2_cross          6.5024        6.5835      6.7736        4.12
E4_B              5.5680        5.9944      6.0754        8.46
E4_cross          4.3804        4.5678      4.7819        8.79

=== Projection bottleneck ===

Case          Projection(ms)   Projection(%)    Final(%)   Layers(%)
P1                    4.8511           95.41       53.49       46.50
E2                    5.9062           94.12       51.46       48.52
E4                    6.3536           92.53       47.86       52.12
E8                    5.1456           88.85       36.91       63.08
E2_B                  6.0027           94.28       51.04       48.94
E2_cross              6.2435           94.55       51.55       48.43
E4_B                  5.5667           92.87       47.64       52.34
E4_cross              4.2040           92.04       44.39       55.59

=== Stage breakdown: E4_cross ===

Stage                     Per call (us)    Per token (ms)    Forward (%)
embedding                         0.839            0.0008           0.02
attention_norm                    0.592            0.0036           0.08
qkv                             116.964            0.7018          15.36
rope                              7.842            0.0471           1.05
compute_attention                48.002            0.2880           6.31
attention_output                 38.419            0.2305           5.06
ffn_norm                          0.673            0.0040           0.09
ffn_up                          134.501            0.8070          17.67
swiglu                            3.195            0.0192           0.41
ffn_down                         72.868            0.4372           9.59
final                          2020.479            2.0205          44.39
layers                          423.222            2.5393          55.59
forward                        4567.765            4.5678         100.00

=== CPU affinity comparisons ===

      E2 -> E4       speedup=0.916x (6.2735 -> 6.8492 ms)
      E4 -> E8       speedup=1.187x (6.8492 -> 5.7722 ms)
      E2 -> E2_B     speedup=0.985x (6.2735 -> 6.3672 ms)
      E2 -> E2_cross speedup=0.953x (6.2735 -> 6.5835 ms)
      E4 -> E4_B     speedup=1.143x (6.8492 -> 5.9944 ms)
      E4 -> E4_cross speedup=1.499x (6.8492 -> 4.5678 ms)
    E4_B -> E4_cross speedup=1.312x (5.9944 -> 4.5678 ms)
      E2 -> E4       speedup=0.916x (6.2735 -> 6.8492 ms)
      E4 -> E8       speedup=1.187x (6.8492 -> 5.7722 ms)
E4_cross -> E8       speedup=0.791x (4.5678 -> 5.7722 ms)

=== Interpretation guide ===

P2 vs E2:
  Compare P-core and E-core performance at 2 threads.
E2 vs E4 vs E8:
  Evaluate scaling using homogeneous E-cores.
P2 vs P4_HT:
  Evaluate Hyper-Threading on the same physical P-cores.
P2 vs P2E2:
  Evaluate mixed P/E-core scheduling.
auto_2 vs P2:
  Evaluate explicit CPU affinity.

NOTE:
  These measurements exclude most logging overhead.
  Slowdowns alone do not establish DRAM bandwidth saturation.
  CPU frequency, contention and OpenMP overhead also matter.
  Medians are calculated independently per metric.

=== Saved files ===
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_183251_l2/analysis/cases.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_183251_l2/analysis/stages.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_183251_l2/analysis/runs.csv
/home/denjo/2026-07-llama2c/benchmark_affinity_stories15M/20261012_183251_l2/analysis/report.txt
```

# 1. 今回の実験結果まとめ

## 実験：L2キャッシュ共有とCPU affinityによるllama2.c推論性能の変化

**実験日：2026年10月12日**

### 目的

llama2.cのCPU推論において、E-coreのL2キャッシュ共有関係とスレッド配置が性能に与える影響を調査する。

特に、以下の2つを比較する。

- 同じL2キャッシュを共有するE-coreにスレッドを集中させる配置
- 異なるL2キャッシュクラスタにスレッドを分散させる配置

### 実験環境・条件

| 項目 | 内容 |
|---|---|
| プログラム | llama2.c（OpenMP並列化版） |
| モデル | stories15M.bin |
| CPU | P-core / E-core混合構成 |
| E-coreクラスタA | CPU 4–7、L2 2MB共有 |
| E-coreクラスタB | CPU 8–11、L2 2MB共有 |
| 入力 | Once upon a time |
| 測定回数 | 各構成5回 |
| 測定対象 | forward計算時間 |
| 実行トークン数 | 222 token / run |
| Transformer layer呼出回数 | 1332回 / run |

### 測定結果

各構成の5回の測定値の中央値を使用する。

| 構成 | 使用CPU | スレッド | Forward (ms/token) | P1比 |
|---|---|---:|---:|---:|
| P1 | 0 | 1 | 5.0875 | 1.000 |
| E2 | 4,5 | 2 | 6.2735 | 0.811 |
| E4 | 4,5,6,7 | 4 | 6.8492 | 0.743 |
| E8 | 4–11 | 8 | 5.7722 | 0.881 |
| E2_B | 8,9 | 2 | 6.3672 | 0.799 |
| E2_cross | 4,8 | 2 | 6.5835 | 0.773 |
| E4_B | 8,9,10,11 | 4 | 5.9944 | 0.849 |
| **E4_cross** | **4,5,8,9** | **4** | **4.5678** | **1.114** |

CPU affinity別の推論時間

Forward計算時間の中央値。小さいほど高速。

Case	Ms
P1	5.0875
E2	6.2735
E4	6.8492
E8	5.7722
E2_B	6.3672
E2_cross	6.5835
E4_B	5.9944
E4_cross	4.5678

### 主な発見

**① 同一L2クラスタ内では、スレッド数の増加によって性能が悪化した。**

E2 → E4では、forward計算時間が6.2735msから6.8492msへ増加した。スループットは約8.4%低下している。

**② 同じ4スレッドでも、L2クラスタへの配置によって大きく性能が変化した。**

E4とE4_crossを比較すると、E4_crossは約1.50倍高速だった。

別クラスタのE4_Bとの比較でも約1.31倍高速だった。

**③ E4_crossはP-core単独よりも高速だった。**

P1の5.0875msに対して、E4_crossは4.5678msだった。

ただし、演算ステージごとに比較すると次のようになる。

| 処理 | P1 | E4_cross | 高速な構成 |
|---|---:|---:|---|
| Transformer Layers | 2.3644ms | 2.5393ms | P1 |
| Final Projection | 2.7225ms | 2.0205ms | E4_cross |
| Forward全体 | 5.0875ms | 4.5678ms | E4_cross |

この結果は、**演算ステージによって最適なCPU配置が異なる可能性**を示している。

**④ 性能変動が比較的大きい構成も存在した。**

特にE4の計算時間は5.9329〜7.3693msと大きく変動した。

CPU周波数、温度、バックグラウンド負荷などが測定結果に影響している可能性がある。

### 現時点の結論

今回の結果は、CPU affinityとL2キャッシュクラスタの構成がllama2.cの推論性能に大きく影響することを示している。

特に、E-core 4スレッドを2つのL2クラスタに分散した構成では、単一クラスタに集中した構成より高い性能が得られた。

ただし、この性能差がL2キャッシュ競合そのものに起因するかは未確認である。共有実行資源、メモリ帯域、CPU周波数、OpenMPの同期コストなども考慮する必要がある。

したがって、次の実験では、**スレッド数とクラスタへの分配比率を体系的に変化させ、性能差が生じる条件を特定する。**

---
# E-coreのL2クラスタ内・クラスタ間の配置を比較
```shell
#!/usr/bin/env bash
# L2 topology scaling experiment for instrumented llama2.c (OpenMP).
# Run this script from anywhere after copying it into ~/2026-07-llama2c/.
set -Eeuo pipefail
export LC_ALL=C

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
MODEL="${MODEL:-./stories15M.bin}"
RUN_BIN="${RUN_BIN:-./run}"
OUT_ROOT="${OUT_ROOT:-./benchmark_affinity_stories15M}"
REPEATS="${REPEATS:-7}"
WARMUPS="${WARMUPS:-1}"
STEPS="${STEPS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Once upon a time}"
ORDER_SEED="${ORDER_SEED:-20261012}"
REBUILD="${REBUILD:-0}"
PERF="${PERF:-0}"
PERF_REPEATS="${PERF_REPEATS:-2}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-references,cache-misses}"
ANALYZER="${ANALYZER:-./analyze/analyze_l2_scaling.py}"

for integer in REPEATS WARMUPS STEPS ORDER_SEED PERF_REPEATS; do
  if ! [[ "${!integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $integer must be a nonnegative integer." >&2; exit 1
  fi
done
if (( REPEATS < 1 || STEPS < 1 )); then echo "ERROR: REPEATS and STEPS must be >= 1" >&2; exit 1; fi
for tool in python3 taskset sha256sum; do
  command -v "$tool" >/dev/null || { echo "ERROR: missing $tool" >&2; exit 1; }
done
if [[ "$REBUILD" == 1 ]]; then
  echo 'Rebuilding OpenMP executable: make runomp CC=gcc'
  make runomp CC=gcc
fi
[[ -x "$RUN_BIN" ]] || { echo "ERROR: executable not found: $RUN_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -f "$ANALYZER" ]] || { echo "ERROR: analyzer missing: $ANALYZER" >&2; exit 1; }

# Output folder is intentionally newly created: never overwrite prior measurements.
mkdir -p "$OUT_ROOT"
OUT_DIR="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_l2_scaling"
mkdir "$OUT_DIR"
mkdir "$OUT_DIR/logs" "$OUT_DIR/stdout" "$OUT_DIR/perf"

# cluster_a = CPUs 4-7 (shared 2 MiB L2); cluster_b = CPUs 8-11 (shared 2 MiB L2).
# The A/B symmetry and different splits at fixed thread count help separate
# cluster balance from a specific CPU cluster's performance.
cat > "$OUT_DIR/config.tsv" <<'CASES'
case	threads	cpus	cluster_a	cluster_b
P1	1	0	0	0
A1	1	4	1	0
B1	1	8	0	1
A2	2	4,5	2	0
B2	2	8,9	0	2
X2_1_1	2	4,8	1	1
A3	3	4,5,6	3	0
B3	3	8,9,10	0	3
X3_2_1	3	4,5,8	2	1
X3_1_2	3	4,8,9	1	2
A4	4	4,5,6,7	4	0
B4	4	8,9,10,11	0	4
X4_3_1	4	4,5,6,8	3	1
X4_2_2	4	4,5,8,9	2	2
X4_1_3	4	4,8,9,10	1	3
X5_4_1	5	4,5,6,7,8	4	1
X5_3_2	5	4,5,6,8,9	3	2
X5_2_3	5	4,5,8,9,10	2	3
X5_1_4	5	4,8,9,10,11	1	4
X6_4_2	6	4,5,6,7,8,9	4	2
X6_3_3	6	4,5,6,8,9,10	3	3
X6_2_4	6	4,5,8,9,10,11	2	4
X7_4_3	7	4,5,6,7,8,9,10	4	3
X7_3_4	7	4,5,6,8,9,10,11	3	4
X8_4_4	8	4,5,6,7,8,9,10,11	4	4
CASES

# Check the actual L2 topology; abort rather than silently comparing
# arrangements that don't have the assumed cache-sharing relationship.
for spec in '4:4-7' '5:4-7' '6:4-7' '7:4-7' '8:8-11' '9:8-11' '10:8-11' '11:8-11'; do
  cpu="${spec%%:*}"; expected="${spec#*:}"; found=0
  for cache in /sys/devices/system/cpu/cpu"$cpu"/cache/index*; do
    [[ -f "$cache/level" ]] || continue
    if [[ "$(< "$cache/level")" == 2 ]]; then
      actual="$(< "$cache/shared_cpu_list")"
      if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: CPU $cpu L2 shared with '$actual', expected '$expected'" >&2
        exit 1
      fi
      found=1
      break
    fi
  done
  (( found == 1 )) || { echo "ERROR: CPU $cpu L2 topology unavailable" >&2; exit 1; }
done

# Validate that all requested affinity masks are permitted for this user.
while IFS=$'\t' read -r name threads cpus a b; do
  [[ "$name" == case ]] && continue
  if ! taskset -c "$cpus" /bin/true >/dev/null 2>&1; then
    echo "ERROR: CPU affinity '$cpus' ($name) not permitted here" >&2
    exit 1
  fi
done < "$OUT_DIR/config.tsv"

cat > "$OUT_DIR/environment.txt" <<META
start_local=$(date -Is)
project_dir=$PROJECT_DIR
run_bin=$RUN_BIN
model=$MODEL
repeats=$REPEATS
warmups=$WARMUPS
steps=$STEPS
temperature=$TEMPERATURE
seed=$SEED
prompt=$PROMPT
order_seed=$ORDER_SEED
perf=$PERF
perf_repeats=$PERF_REPEATS
perf_events=$PERF_EVENTS
uname=$(uname -a)
allowed_cpus=$(taskset -pc $$ 2>&1)
META
if command -v git >/dev/null && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  { git rev-parse HEAD; git status --short; } > "$OUT_DIR/git_revision_and_status.txt"
fi
printf 'case\trun\tthreads\tcpus\tcluster_a\tcluster_b\tlog\tstdout\tsha256\n' > "$OUT_DIR/manifest.tsv"

# A different reproducible random permutation of all 25 cases per block.
# Interleaving all cases per block limits the influence of drift in clock,
# temperature and background contention compared with grouped runs.
random_cases() {
  python3 - "$OUT_DIR/config.tsv" "$ORDER_SEED" "$1" <<'PY'
import csv, random, sys
with open(sys.argv[1], newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
random.Random(int(sys.argv[2]) + int(sys.argv[3])).shuffle(rows)
for row in rows:
    print('\t'.join(row[key] for key in ('case','threads','cpus','cluster_a','cluster_b')))
PY
}

run_in_affinity() {
  local threads="$1" cpus="$2" stdout="$3" stderr="$4"
  OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE \
    OMP_PLACES=threads taskset -c "$cpus" \
    "$RUN_BIN" "$MODEL" -n "$STEPS" -t "$TEMPERATURE" -s "$SEED" -i "$PROMPT" \
    >"$stdout" 2>"$stderr"
}

printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Conditions: %s cases x %s measured runs (+ %s warm-up rounds)\n' \
  "$(($(wc -l < "$OUT_DIR/config.tsv") - 1))" "$REPEATS" "$WARMUPS"

for ((round=1; round<=WARMUPS; round++)); do
  echo "Warm-up block $round/$WARMUPS"
  while IFS=$'\t' read -r name threads cpus a b; do
    echo "  warm-up $name ($cpus)"
    run_in_affinity "$threads" "$cpus" /dev/null /dev/null
  done < <(random_cases "$(( -round ))")
done

for ((round=1; round<=REPEATS; round++)); do
  echo "Measured block $round/$REPEATS"
  while IFS=$'\t' read -r name threads cpus a b; do
    stem="${name}_r$(printf '%02d' "$round")"
    out="stdout/$stem.out"; log="logs/$stem.log"
    run_in_affinity "$threads" "$cpus" "$OUT_DIR/$out" "$OUT_DIR/$log"
    if ! grep -Eq '103:[[:alnum:]_-]+:{1,2}[[:space:]]*[0-9.]+' "$OUT_DIR/$log"; then
      echo "ERROR: no stage-103 timing found in $OUT_DIR/$log" >&2
      echo 'The run executable must contain the existing stage-timing instrumentation.' >&2
      exit 1
    fi
    sha="$(sha256sum "$OUT_DIR/$out" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$round" "$threads" "$cpus" "$a" "$b" "$log" "$out" "$sha" \
      >> "$OUT_DIR/manifest.tsv"
    printf '  %-10s CPUs=%-22s run=%d\n' "$name" "$cpus" "$round"
  done < <(random_cases "$round")
done

# Optional second, separate pass: perf overhead is NOT mixed into timing trials.
# For hybrid Intel cores, generic counters may not count identically across
# PMUs, and cache-misses is not the same as L2 misses.
if [[ "$PERF" == 1 ]]; then
  if ! command -v perf >/dev/null; then
    echo 'WARNING: perf unavailable; skipping hardware counters.' >&2
  elif ! perf stat -x, -e "$PERF_EVENTS" -- /bin/true >/dev/null 2>&1; then
    echo 'WARNING: perf permission denied or event unsupported; skipping hardware counters.' >&2
  else
    for ((round=1; round<=PERF_REPEATS; round++)); do
      echo "Perf block $round/$PERF_REPEATS"
      while IFS=$'\t' read -r name threads cpus a b; do
        stem="${name}_p$(printf '%02d' "$round")"
        if ! perf stat -x, -o "$OUT_DIR/perf/$stem.csv" -e "$PERF_EVENTS" -- \
          env OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=threads \
          taskset -c "$cpus" "$RUN_BIN" "$MODEL" -n "$STEPS" -t "$TEMPERATURE" \
          -s "$SEED" -i "$PROMPT" > "$OUT_DIR/perf/$stem.out" 2> "$OUT_DIR/perf/$stem.log"; then
          echo "WARNING: perf failed for $stem (see $OUT_DIR/perf/$stem.log)" >&2
          rm -f "$OUT_DIR/perf/$stem.csv"
        fi
      done < <(random_cases "$((ORDER_SEED + round + REPEATS))")
    done
  fi
fi

echo "end_local=$(date -Is)" >> "$OUT_DIR/environment.txt"
python3 "$ANALYZER" --dir "$OUT_DIR"
echo "Saved: $OUT_DIR"
```
# llama2.c 第2実験：L2キャッシュクラスタとスレッド配置の性能評価

**実験日：2026年10月12日**  
**対象：stories15M.bin / OpenMP並列化版 llama2.c**

## 1. 結論・要約

今回の実験では、25種類のCPU affinity構成について各7回、合計175回のベンチマークを実施した。

最も重要な結果は、**CPUスレッド数だけでなく、L2キャッシュクラスタ間のスレッド分配が推論性能に大きく影響する**ことが確認された点である。

主な結果は以下のとおり。

| 項目 | 結果 |
|---|---|
| 最速構成 | X6_3_3（E-core 6スレッド、3+3配置） |
| 最速時間 | 4.1965 ms/token |
| P-core単独比 | 1.222倍 |
| 4スレッドの最速構成 | X4_2_2（2+2配置） |
| 4スレッド均等分散の効果 | 同一クラスタ集中比1.38〜1.44倍 |
| 4→6スレッドの改善 | 約4.0% |
| 主なボトルネック | Projection演算（全体の約90〜96%） |

今回提供された試行別データも確認したところ、175回すべてでトークン数222、Layer呼出回数1332、標準出力のSHA-256ハッシュが一致していた。したがって、出力内容や処理回数の違いによる測定結果の不一致は確認されなかった。貼り付けたテキスト（1）

ただし、**E-coreのL2キャッシュ競合が性能差の原因であることまでは確定していない。** 今回の結果はその仮説と整合的だが、OpenMPのスレッド配置やクラスタ間の共有資源も影響している可能性がある。

---

## 2. スレッド数と推論性能の関係

### 2.1 E-coreのスケーリング

各スレッド数について、最も高速だったE-core構成を比較した。

| Threads | 最速構成 | Forward (ms/token) | Throughput (tokens/s) |
|---:|---|---:|---:|
| 1 | B1 | 11.3688 | 88.0 |
| 2 | B2 | 6.2275 | 160.6 |
| 3 | A3 | 4.8916 | 204.4 |
| 4 | X4_2_2 | 4.3650 | 229.1 |
| 5 | X5_3_2 | 4.2181 | 237.1 |
| **6** | **X6_3_3** | **4.1965** | **238.3** |
| 7 | X7_4_3 | 4.6375 | 215.6 |
| 8 | X8_4_4 | 5.0441 | 198.2 |

E-coreスレッド数と推論時間

各スレッド数の最速CPU affinity構成。小さいほど高速。25構成・各7回の測定中央値。

Threads	Forward
1	11.3688
2	6.2275
3	4.8916
4	4.365
5	4.2181
6	4.1965
7	4.6375
8	5.0441

スレッド数を1から4へ増やすと大幅に高速化した。一方、4〜6スレッドでは改善が小さくなり、7〜8スレッドでは性能が悪化した。

特に、4スレッドから6スレッドへの増加では、スレッド数が50%増えているにもかかわらず、スループット向上は約4.0%にとどまった。

したがって、今回のモデル・CPU・実装では、**4〜6スレッド付近が性能上の有力な最適領域**と考えられる。

### 2.2 同一クラスタ内でのスケーリング

同一L2クラスタだけを使用した場合、以下の結果となった。

| Threads | Cluster A | Cluster B |
|---:|---:|---:|
| 1 | 11.4397 | 11.3688 |
| 2 | 6.3040 | 6.2275 |
| 3 | 4.8916 | 5.1231 |
| 4 | 6.2669 | 6.0392 |

単一クラスタでは、3スレッドから4スレッドに増やすと性能が悪化した。

- Cluster A：実行時間が28.1%増加
- Cluster B：実行時間が17.9%増加

この結果は、4つのE-coreが同じ2MBのL2キャッシュを共有する構成において、何らかの共有資源の競合や並列化オーバーヘッドが顕在化している可能性を示唆する。

ただし、L2キャッシュ容量の不足だけでなく、キャッシュ帯域やOpenMPの同期なども候補になる。

---

## 3. L2クラスタ間の分散配置の効果

### 3.1 4スレッドでの比較

今回、最も明確な傾向が観察されたのは4スレッド構成だった。

| 構成 | 配置 | Forward (ms/token) |
|---|---|---:|
| A4 | 4+0 | 6.2669 |
| B4 | 0+4 | 6.0392 |
| X4_1_3 | 1+3 | 4.8827 |
| X4_3_1 | 3+1 | 4.5929 |
| **X4_2_2** | **2+2** | **4.3650** |

X4_2_2は、A4に対して約1.44倍、B4に対して約1.38倍のスループットを示した。

試行別データから、同じ反復番号での性能を比較したところ、**X4_2_2はA4、B4の両方に対して7回すべてで高速**だった。

さらに、前回の実験でもE4_cross（2+2）がE4（4+0）に対して約1.50倍高速だった。

今回の1.44倍という結果と整合しており、4スレッドでの均等分散配置の優位性には再現性が認められる。

### 3.2 2スレッドでは分散配置が逆効果

一方、2スレッドでは以下のようになった。

- A2：6.3040ms
- B2：6.2275ms
- X2_1_1：6.5169ms

この場合、クラスタを分散させた方が遅い。

したがって、クラスタ分散は常に有効というわけではない。

スレッド数によって、共有資源の競合を回避する利益と、分散配置によるコストの大小関係が変化する可能性がある。

**最適なCPU配置は、スレッド数とクラスタへの分配数の両方に依存する**と考えられる。

---

## 4. Cluster A/B間の非対称性

今回の実験では、特に5〜7スレッドで大きな非対称性が観察された。

| Threads | 配置1 | Forward | 配置2 | Forward | 性能比 |
|---:|---|---:|---|---:|---:|
| 5 | 1+4 | 8.1912 | 4+1 | 5.3825 | 1.52倍 |
| 6 | 2+4 | 7.1891 | 4+2 | 4.9106 | 1.46倍 |
| 7 | 3+4 | 6.1673 | 4+3 | 4.6375 | 1.33倍 |

どの組み合わせでも、Cluster Bに4スレッドを配置した構成の方が遅い。

また、この差は単一クラスタの性能差だけでは説明しにくい。

実際、A4単独とB4単独の時間はそれぞれ6.2669ms、6.0392msであり、B4の方がわずかに高速だった。

ところが、別クラスタのスレッドと組み合わせると、B側に4スレッドを配置した構成では顕著な性能低下が発生している。

この結果は、単純なL2キャッシュ容量のモデルだけでは説明できない。

考えられる原因には、OpenMPのスレッド割当、CPU間のスレッド移動、クラスタ固有の実行資源、周波数制御などがある。

特に、`taskset`によるCPU集合の制限が、個々のOpenMPスレッドを固定しているとは限らない点に注意する必要がある。

---

## 5. ステージ別の性能分析

今回の追加データから、**X6_3_3が最速になった理由を、演算ステージ別にある程度説明できる。**

### 5.1 主要構成の比較

単位はms/token。各ステージの独立した中央値を示す。

| Stage | P1 | X4_2_2 | X5_3_2 | X6_3_3 |
|---|---:|---:|---:|---:|
| QKV Projection | **0.591** | 0.640 | 0.644 | 0.712 |
| Attention計算 | **0.211** | 0.261 | 0.269 | 0.326 |
| FFN Up | 0.865 | 0.759 | 0.754 | **0.735** |
| FFN Down | 0.476 | **0.411** | 0.418 | 0.414 |
| Final Projection | 2.756 | 2.011 | 1.782 | **1.689** |
| Layers全体 | 2.373 | **2.359** | 2.396 | 2.531 |
| **Forward全体** | 5.130 | 4.365 | 4.218 | **4.196** |

出典：ステージ別計測値。貼り付けたテキスト（2） 貼り付けたテキスト（2） 貼り付けたテキスト（2）

ここから3つの重要な点が分かる。

**第一に、Transformer Layers全体ではX4_2_2が最速である。**

X6_3_3はX4_2_2に比べてLayers処理が約7.3%遅い。

**第二に、Final ProjectionではX6_3_3が最速である。**

X4_2_2に対して約16.0%、P1に対して約38.7%の時間短縮が得られている。

**第三に、X6_3_3の全体性能の優位性は、主にFinal Projectionの高速化によって実現されている。**

つまり、スレッド数を6に増やすとFinal Projectionは高速化する一方、Transformer Layers内部の処理は遅くなる。

これは、以前検討していた**演算ステージごとのスレッド数最適化**を支持する結果である。

### 5.2 X6_3_3のボトルネック

X6_3_3における主要な処理時間の内訳は以下のとおり。

| Stage | Forwardに占める割合 |
|---|---:|
| Final Projection | 40.24% |
| FFN Up | 17.61% |
| QKV Projection | 16.89% |
| FFN Down | 9.86% |
| Attention Output | 5.70% |
| Compute Attention | 7.81% |

5種類のprojection系演算の合計は約90.31%となる。

したがって、今回のモデルでは依然としてprojection演算が主なボトルネックになっている。

ただし、projectionが実行時間の大半を占めることだけでは、メモリ帯域律速と断定することはできない。

### 5.3 X6_2_4が遅い原因

X6_2_4とX6_3_3を比較すると、

| Stage | X6_2_4 | X6_3_3 |
|---|---:|---:|
| QKV | 1.108 | 0.712 |
| FFN Up | 1.285 | 0.735 |
| FFN Down | 0.636 | 0.414 |
| Final | 3.107 | 1.689 |
| Forward | 7.189 | 4.196 |

となる。

特定の演算だけでなく、複数の主要演算で時間が増加している。貼り付けたテキスト（2）

したがって、X6_2_4の性能低下はFinal Projection固有の問題というより、**複数の演算に共通するスレッド実行環境の問題**である可能性がある。

これも、スレッドの実際の配置やCPU周波数を調べる必要性を示している。

---

## 6. 測定の安定性と最適構成の判定

各構成7回の測定について、中央値とMADを確認した。

| 構成 | Median (ms) | MAD (ms) | Min–Max (ms) |
|---|---:|---:|---:|
| X4_2_2 | 4.3650 | 0.1496 | 3.6955–4.6580 |
| X5_3_2 | 4.2181 | 0.2717 | 3.0918–4.4898 |
| X6_3_3 | 4.1965 | 0.1133 | 3.1148–4.3388 |

X6_3_3が中央値では最速だが、X5_3_2との差はわずか0.0216ms（約0.5%）である。

さらに、同一反復ブロック内で両者を比較すると、**7回中5回はX5_3_2の方が高速**だった。これは、中央値だけでX6_3_3を明確な最適構成と判定するのが適切ではないことを示している。

一方、X6_3_3はX4_2_2に対して7回中5回高速だった。

また、X5_3_2とX6_3_3には3.1ms程度の非常に高速な試行が存在する。これらの試行ではLayersとFinalの両方が短縮されているため、測定環境に何らかの変化が生じた可能性もある。貼り付けたテキスト（1）

現時点では、次のように評価するのが妥当である。

| 評価項目 | 最有力構成 |
|---|---|
| 計測中央値での最高性能 | X6_3_3 |
| スレッド数に対する性能効率 | X4_2_2 |
| 5〜6スレッド間の優劣 | 未確定 |
| 明確な再現性がある現象 | 4スレッドの2+2配置の優位性 |

---

## 7. 総合考察

今回の実験から、llama2.cのCPU推論において以下の傾向が確認された。

まず、E-coreのスレッド数を単純に増加させても、推論性能が単調に向上するわけではない。実際、単一L2クラスタ内では3スレッドから4スレッドへの増加で性能が悪化した。また、全8E-coreを使用した構成よりも、4〜6E-coreを使用する構成の方が高速だった。

次に、スレッド配置が推論性能に大きな影響を及ぼすことが確認された。特に4スレッドでは、同一L2クラスタ内に集中させた構成より、2つのクラスタに2スレッドずつ分散した構成が約1.4倍高速だった。この結果は前回の実験とも整合しており、再現性のある観察といえる。

一方、5〜7スレッドでは、クラスタ間の分配比率だけでなく、どちらのクラスタに多くスレッドを配置するかによって大きな性能差が発生した。この非対称性はL2キャッシュ容量だけでは説明しきれず、OpenMPのスレッド配置、共有実行資源、CPU周波数などの影響を考慮する必要がある。

さらに、ステージ別の分析から、Transformer LayersとFinal Projectionで最適なスレッド数が異なることが分かった。X4_2_2はLayers部分で最速だった一方、X6_3_3はFinal Projectionで最速だった。

したがって、今後はスレッド数・CPU affinityの全体最適化に加え、**演算ステージごとに適切なスレッド数と配置を選択する手法**が有望と考えられる。

### 次の実験で検証すべき仮説

今回の結果から、次の3つが優先的な検証対象となる。

1. **スレッド配置仮説：** A/Bクラスタ間の非対称性は、OpenMPスレッドの配置・移動によって生じている。
2. **共有資源競合仮説：** 同一L2クラスタ内で4スレッドを使用した際の性能低下は、キャッシュまたはクラスタ共有資源の競合によって生じている。
3. **ステージ別最適化仮説：** Transformer Layersでは4スレッド、Final Projectionでは6スレッドを利用することで、固定スレッド数の場合より高速化できる。

特に3番目は、今回得られたステージ別実行時間と直接対応しているため、具体的な最適化手法として検討する価値が高い。

ただし、ステージ間でスレッド数を切り替える際にはOpenMPの同期・再配置コストが発生し得るため、ステージ単独の実行時間の改善が必ずしも全体性能の改善につながるとは限らない。

**今回の実験の最大の成果は、「最適なスレッド数は6」という単一の結果ではなく、スレッド数・L2クラスタ配置・演算ステージによって最適な並列実行構成が変化することを、実測値から示した点にある。**
