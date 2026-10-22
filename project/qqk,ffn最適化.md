
# 導入
## qqkffn最適化
### build
```shell
mkdir -p build
gcc -Ofast -march=native -fopenmp run/run.c -lm -o build/run-fused
```

### 実行
```shell
env OMP_NUM_THREADS=10 \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close \
    OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
    OMP_SCHEDULE='dynamic,16' \
    taskset -c 0,2,4,5,6,7,8,9,10,11 \
    ./build/run-fused build/stories15M.bin \
    -z tokenizer.bin -n 256 -t 0 -s 42 -i "Once upon a time"
```


## 通常版
### build
```shell
mkdir -p build
gcc -I. -Ofast -march=native -fopenmp \
    run/run_upstream.c -lm -o build/run-upstream
```

### 実行
```shell
env OMP_NUM_THREADS=10 \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close \
    OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
    taskset -c 0,2,4,5,6,7,8,9,10,11 \
    ./build/run-upstream build/stories15M.bin \
    -z tokenizer.bin -n 256 -t 0 -s 42 -i "Once upon a time"
```



# 仕組みの解説



まず，もとのコードでは以下のようなmatmulを使っていた。今回はこれをさらに改造して最適化していくことを考える。

```c

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
```

これは数式で表すと
$$
\underbrace{y}_{d\times1}
=
\underbrace{W}_{d\times n}
\underbrace{x}_{n\times1}
$$
$$\begin{pmatrix}
y_0\\y_1\\y_2\\\vdots\\y_{d-1}
\end{pmatrix}
=
\begin{pmatrix}
w_{0,0}&w_{0,1}&\cdots&w_{0,n-1}\\
w_{1,0}&w_{1,1}&\cdots&w_{1,n-1}\\
\vdots&\vdots&\ddots&\vdots\\
w_{d-1,0}&w_{d-1,1}&\cdots&w_{d-1,n-1}
\end{pmatrix}
\begin{pmatrix}
x_0\\x_1\\\vdots\\x_{n-1}
\end{pmatrix}$$
ということを行っている。

## qkvの統合と並列化

元のコードでは以下のようにq,k,vをそれぞれ計算してる。
```c
matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
```

<details>
<summary>qkvの解説</summary>
Transformerでは 以下のような計算を行っている。

$q= W_Qx$　$k= W_Kx$　$v= W_Vx$

ここでの $x$ は入力されたベクトルであり，$W$ はモデルの重みです。ここでは，$W$ は`model.bin`から取ってきており，$x$ はforwardごとに更新されていくものです。この$w$,$k$,$v$を全て処理したら，


$\text{score}_t=\frac{q^\top k_t}{\sqrt{d_{\mathrm{head}}}}$　$a_t=\operatorname{softmax}(\text{score})_t$　$o=\sum_t a_t v_t$

このような形で処理していく。ここで重要なのは$q$,$k$,$v$はそれぞれ独立に計算され，最後にまとめて用いるということである。
</details>

この処理は独立であるので，並列化が可能である。$q$,$k$,$v$を計算するのに必要な$W_Q$,$W_K$,$W_V$はそれぞれ，

$W_Q$：ポインタがw->wq + l*dim*dim で $\text{dim} \times \text{dim}$

$W_K$：ポインタがw->wk + l*dim*kv_dim で $\text{dim}\times \text{kv}_\text{dim}$

$W_V$：ポインタがw->wv + l*dim*kv_dim で $\text{dim}\times \text{kv}_\text{dim}$

これらについて$q= W_Qx$　$k= W_Kx$　$v= W_Vx$を独立に計算せず。


$$\begin{pmatrix}
q\\k\\v
\end{pmatrix}
=
\underbrace{
\begin{pmatrix}
W_Q\\W_K\\W_V
\end{pmatrix}
}_{(dim+2kv\_dim)\times dim}
x$$

このようにまとめて計算する。columnは変わらないので，qkvの重みをそれぞれrowが$0〜\text{dim}$，$\text{dim}〜\text{dim}+\text{kv}_\text{dim}$， $\text{dim}+\text{kv}_\text{dim}〜\text{dim}+2\text{kv}_\text{dim}$，の部分に分けたという形である。

これを以下のようにopenMPで並列化した。これで今までは$q$→$k$→$v$と計算していたところを並列に計算できる。


```c
static void matmul_qkv(
    float *q, float *k, float *v, float *x,
    float *wq, float *wk, float *wv,
    int dim, int kv_dim
) {
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < dim + 2 * kv_dim; row++) {
        int i;
        float *out, *w;

        if (row < dim) {
            i = row;
            out = q;
            w = wq;
        }
        else if (row < dim + kv_dim) {
            i = row - dim;
            out = k;
            w = wk;
        }
        else {
            i = row - dim - kv_dim;
            out = v;
            w = wv;
        }

        float val = 0.0f;
        for (int j = 0; j < dim; j++) {
            val += w[i * dim + j] * x[j];
        }

        out[i] = val;
    }
}
```

## ffnの結合と並列化

ffnとは以下のように計算をする，Forwardの最後の方の部分である。
$$\mathrm{FFN}(x)=W_2\left(\mathrm{SiLU}(W_1x)\odot W_3x\right)$$


流れとしては，まずrmsnormで

`rmsnorm(s->xb, x,w->rms_ffn_weight + l*dim, dim);`

と正規化する。次にこの正規化入力$x$について，$a = W_1z$ 　$b = W_3z$とを行う。


```c
// W1
matmul(s->hb, s->xb,
       w->w1 + l*dim*hidden_dim,
       dim, hidden_dim);

// W3
matmul(s->hb2, s->xb,
       w->w3 + l*dim*hidden_dim,
       dim, hidden_dim);
```

次に，$h=\operatorname{SiLU}(a)\odot b$を通したあと，$W2$をかけて$y=W_2h$とする。

```c
matmul(s->xb, s->hb,
       w->w2 + l*dim*hidden_dim,
       hidden_dim, dim);
```

このうち，$W1$，$W2$をかける部分，そして，その２つから$h$を計算する部分の３つを並列して行うことができる。

まず，$W1$,$W2$をかける部分２つが並列できるのは当然だが，$h$を計算するには$a$と$b$が揃わないとだめだと感じるかもしれない。実際，ミクロには$a$,$b$が必要なのだが，行列全体を待つ必要はない。というのも各行列の要素を考えると，以下のようになっている。


$$
a_i = \sum_{j=0}^{d-1} W_{1,i,j}z_j\\
b_i = \sum_{j=0}^{d-1} W_{3,i,j}z_j\\
h_i = \operatorname{SiLU}(a_i)b_i
$$


ここで，$h_i$は$a_i$，$b_i$のみに依存しており，$a_k$,$b_k$($k\ne i$)に依存していない。そのためこれらは並列して実行ができる。そのコードが以下である。

```c
static void ffn_up_gate(
    float *hb, float *hb2,
    float *x, float *up, float *gate,
    int dim, int hidden_dim
) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < hidden_dim; i++) {
        float val = 0.0f, g = 0.0f;

        // ① W1のi行目を計算
        for (int j = 0; j < dim; j++)
            val += up[i * dim + j] * x[j];

        // ② W3のi行目を計算
        for (int j = 0; j < dim; j++)
            g += gate[i * dim + j] * x[j];

        // ③ SiLUと要素積
        hb2[i] = g;
        val *= 1.0f / (1.0f + expf(-val));
        hb[i] = val * g;
    }
}
```