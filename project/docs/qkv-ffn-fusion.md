# QKV・FFN融合のみの実装

比較元：未改造upstreamの`350e04fe35433e6d2941dce5a1f53308f87058eb`。
出力専用の人数・schedule切替は削除し、全演算で同じスレッド数を使う。
融合処理を`src/fused.h`へ分離し、`run.c`のQKVとFFN up/gate/SwiGLUを置き換える。

- QKV：Q/K/Vの出力行を連続した仕事番号として扱い、並列領域を3回から1回へ減らす。
- FFN：up/gateの2内積とSwiGLUを1行ごとに計算する。領域を2回から1回へ減らし、
  SwiGLUも並列化する。gateの値は従来どおり`hb2`にも保存する。
- 内積の加算順序と重み形式は維持する。`-Ofast`によるコード生成の丸め差はあり得る。
- 両方ともupstreamのmatmulと同じ、schedule指定なしの`omp parallel for`を使う。CPU探索や独自ワーカー管理は追加しない。
- OpenMPなしでも直列実行できる。

既存のLinux測定入口をそのまま利用できる。

```sh
uv run src/cpuprofiler/benchmark_distribute.py
```

`run.c`のupstreamとの差分は融合ヘッダのincludeとQKV・FFNの呼び出し置換のみ。
`OUTPUT_*`と`OMP_SCHEDULE`は使用しない。融合の速度効果を比較する際は、
upstreamと融合版でCPU集合・人数・ビルドオプション・モデル・入力を揃える。

数値検証：

```sh
python3 -m unittest discover -s tests -p 'test_fused.py' -v
CC=gcc TEST_OPENMP=1 python3 -m unittest discover -s tests -p 'test_fused.py' -v
```

未改造upstreamソースと合成モデルの全logits・gate値・KV cacheを比較する。
MHA/GQA、端数の出る行数、複数コンテキスト、OpenMPでは1/4/14スレッドを検証。
O2はビット一致、Ofastは絶対誤差1e-6＋相対誤差1e-5以内で判定する。
OpenMP版のテストには`-fopenmp`対応コンパイラが必要。
