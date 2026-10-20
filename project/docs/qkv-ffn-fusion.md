# QKV・FFN融合のみの実装

基点：`origin/dev/cpu-profiler`の`666aa86`。分配設定とoutputの人数切替は基点のまま。
融合処理を`src/fused.h`へ分離し、`run.c`のQKVとFFN up/gate/SwiGLUを置き換える。

- QKV：Q/K/Vの出力行を連続した仕事番号として扱い、並列領域を3回から1回へ減らす。
- FFN：up/gateの2内積とSwiGLUを1行ごとに計算する。領域を2回から1回へ減らし、
  SwiGLUも並列化する。gateの値は従来どおり`hb2`にも保存する。
- 内積の加算順序と重み形式は維持する。`-Ofast`によるコード生成の丸め差はあり得る。
- 両方とも既存matmulと同じ`schedule(runtime)`を使う。CPU探索や独自ワーカー管理は追加しない。
- OpenMPなしでも直列実行できる。

既存のLinux測定入口をそのまま利用できる。

```sh
uv run src/cpuprofiler/benchmark_distribute.py
```

QKV/FFN融合の追加だけでは、この基点に既存のoutputチーム人数変更の費用は解消しない。
融合の速度効果を切り分ける際は、融合前と融合後でCPU集合・人数・スケジュール・
モデル・入力を揃える。今後の探索空間の整理は別変更にする。

数値検証：

```sh
python3 -m unittest discover -s tests -p 'test_fused.py' -v
CC=gcc TEST_OPENMP=1 python3 -m unittest discover -s tests -p 'test_fused.py' -v
```

基点の未融合ソースと合成モデルの全logits・gate値・KV cacheを比較する。
MHA/GQA、端数の出る行数、複数コンテキスト、OpenMPでは1/4/14人・static/dynamicを検証。
O2はビット一致、Ofastは絶対誤差1e-6＋相対誤差1e-5以内で判定する。
OpenMP版のテストには`-fopenmp`対応コンパイラが必要。

## 未改造版との測定

[2026-10-20の比較結果・設定と注意点](qkv-ffn-benchmark.md)。
