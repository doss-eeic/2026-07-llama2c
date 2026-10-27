# 全体のアルゴリズム

CPUトポロジーの構築(L2,SMT関係からDomainを作成)→キャリブレーション(同じ形のDomainを同じものとして扱う)→CPU配置探索→Runtime設定最適化→上位候補の独立確認

```python
search = LayoutSearch(...) #CPUトポロジーの構築
search.warmup() #warmup

if calibrated is None:
    calibrated = self.calibrate(...) #キャリブレーション

search.prepare(calibrated) #CPU配置探索
search.grow() #CPU配置探索
search.tune_runtime() #Runtime設定最適化
results = search.confirm() #上位候補の独立確認

return results[0]
```

# 主要データ構造
まず以下ではPコア4個(２つずつL2キャッシュ，物理コアを共有)，Eコア8個(4つずつL2キャッシュ共有，物理コア独立)，LP_Eコア2個(２つはL2キャッシュ共有)。CPUの組み合わせは$2^14-1=16383$通りなので10^4以上の組み合わせの個数になる。

## 1. Build_Domain
ここではUnionFindを用いてDomainを構築していく。

・各CPUを独立した頂点として始める
・同じL2キャッシュを共有するCPU同士を連結
・同じ物理コアに属する論理CPU同士も連結
・連結成分を一つのDomainとする。

これにより
P P E E E E LP_E
P P E E E E LP_E

は

P P E-E E-E LP_E
| | | | | |  |
P P E-E E-E LP_E

このように結び付けられる。これによってP-Pには3状態，E-E-E-Eには5状態，LP_E-LP_Eには3状態あるので $3^2*5^2*3-1=674$通りになる。


データ構造としては，##notimplemented なので，ここから後々使うShapeとして，`domain.shape #(1,2)`のようなものを考える。これはDomainにある物理コアの個数が1個，論理CPUが2個であるという意味である。

## 2. DomainCalibrator
今回，E-E-E-Eの２つやP-Pの２つはおそらく同じものなのだと考えられる。これを自明と捉えて同一視するのは良くないので，同一の`shape`を持つものについては，適切に計測を行ったうえで同一のものと考える。

### A：各論理CPUの単体性能
```python
single = self._measure([
    (cpu,)
    for domain in domains
    for cpu in domain.cpus
])
```
内包表記ですこし見にくいので，展開した形にすると
```python
candidates = []
for domain in domains:
    for cpu in domain.cpus:
        candidates.append((cpu,))
single = self._measure(candidates)
```
ということである。

### B:Domain全体の性能
```python
full = self._measure([
    domain.cpus
    for group in by_shape.values()
    if len(group) > 1
    for domain in group
])
```

これに付いても展開して
```python
candidates = []

for group in by_shape.value():
    if len(group) > 1:
        for domain in group:
            candidates.append(domain.cpus)

full = self._measure(candidates)
```
このようになっている。by_shapeはshapeごとにまとめたものである。

## 2.1 Profileの生成
各Domainについて，以下を記録する


| 指標 | 意味 |
|---|---|
|per_cpu_means	|  各論理CPUの平均ms/token|
|single_mean	|  各論理CPU平均の平均|
|single_std	    |  各CPUの繰り返し測定のばらつき|
|full_mean	    |  Domain全体を使用した平均ms/token|
|full_std       |  Domain全体の測定のばらつき|

## 2.2 同型判定 (_similar)

２つのDomainが同型と判別される条件は以下

0. `shape`が一致している
1. 量Domainとも測定のばらつきが閾値以下
2. CPUごとの性能をソートして比較し，全て閾値以内
3. Domain全体の性能も閾値以内

$$\left|\log\frac{t_A}{t_B}\right|\leq \varepsilon$$

この式によって性能差の判定をしている。

## 2.3 DomainClassの生成(_Classify)

似ているDomainをクラスにまとめる。このときはUnionFindのような連結ではなく，追加する際にそのクラスの全てのメンバと類似していることを条件にする。

つまり，類似を便宜上`=`で表すとして

$A=B, B=C, A\ne C$　の状況を考える。まず$A$ に$B$を加えるときは$A=B$より追加できて，$(A,B)$。次に$C$を加えるとき，$B=C$であるが，$A\ne C$なので，追加できない。$A=B, B=C, A= C$のときのみ追加できる。

## 3. CPU配置探索(LayoutSearch.grow())


CPU配置を $S$、使用可能なCPU集合を $C$、実行時間を $T(S)$ とすると、各ステップで

$$S_{k+1} =
\underset{c \in C\setminus S_k}{\arg\min}
\ T(S_k\cup\{c\})
$$
を考えて行く。ただし，Topologicalに同一視できる組み合わせは同じものとして考える。

## 3.1 探索の概形

```python
def grow(self):
    best = ()
    for _ in self.cpus:
        stage = {}
        for mask in self.children(best):
            stage[mask] = self.layouts[mask] = (
                self.measure(self.make(mask)).ms_per_token
            )
        self.stages.append(stage)
        best = min(stage, key=stage.get)
```

ここで，
| 関数 | 説明 |
|---|---|
|`self.children(best)`|bestからCPUを１つ追加した候補を生成|
|`stage`|現段階の候補・性能|
|`self.layouts[mask]`|探索全体で共有する候補・性能|
|`self.stages`|各段階の候補・性能を持つ|


## 3.2 同型配置の正規化(Key())

```python
def key(self, mask):
    used = set(mask)
    return tuple(sorted(
        (class_id,
         tuple(sorted(
             (len(core), len(used & set(core)))
             for core in domain.cores
         )))
        for class_id, domain in self.domains
    ))
```

こちらも内包表記で見にくいので，
```python
def key(self, mask):
    used = set(mask)
    result = []

    for class_id, domain in self.domain:
        cores = []

        for core in domain.cores:
            core_size = len(core)
            used_count = len(used & set(core))

            cores.append((core_size, used_count))
        
        cores.sort()

        result.append((class_id, tuple(cores)))
    result.sort()

    return tuple(result)
```
となっている。例えば

```python
domain.cores = ((0, 1), (2, 3))

mask_A = (0, 2)
mask_B = (1, 3)
```
このようにドメインの中にこのように##notimplemented

なっているとき，`maskA`は`((2,1),(2,1))`つまり，１つ目の２コアの中の１つ，２つ目の２コアの中の１つを使っている。同様に`maskB`も`((2,1),(2,1))`となる。同じPコアをつかっていても`mask_C = (0, 1)`とすると`((2,2),(2,0))`となり，前述の２例と異なる。

## 3.3 候補生成(children())
先程の探索の概形にあった，候補を生成する`children()`は内部で

```python
def children(self, parent):
    found = {}
    for cpu in self.cpus:
        if cpu not in parent:
            mask = tuple(sorted(parent + (cpu,)))
            found.setdefault(self.key(mask), mask)
    return list(found.values())
```
となっており，
1. 未使用CPUを1つ選んで配置に追加する
2. key()で正規化する
3. 同じキーの候補は最初の1つだけ残す

と正規化しながら候補を生成している。

CPUが $N$ 個の場合、全配置探索なら最大 $2^N-1$ 通りであるが，この手法では最大でも
$$\sum_{k=1}^{N}(N-k+1)
=\frac{N(N+1)}{2}$$
