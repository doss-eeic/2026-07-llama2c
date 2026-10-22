# QKV・FFN融合のみの実装

比較元：未改造upstreamの`350e04fe35433e6d2941dce5a1f53308f87058eb`。
出力専用の人数・schedule切替は削除し、全演算で同じスレッド数を使う。
融合処理を`src/fused.h`へ分離し、`run/run.c`のQKVとFFN up/gate/SwiGLUを置き換える。

- QKV：Q/K/Vの出力行を連続した仕事番号として扱い、並列領域を3回から1回へ減らす。
- FFN：up/gateの2内積とSwiGLUを1行ごとに計算する。領域を2回から1回へ減らし、
  SwiGLUも並列化する。gateの値は従来どおり`hb2`にも保存する。
- 内積の加算順序と重み形式は維持する。`-Ofast`によるコード生成の丸め差はあり得る。
- 通常matmul・QKV・FFNは全て`schedule(runtime)`を使い、全体共通の`OMP_SCHEDULE`で分配する。CPU探索や独自ワーカー管理は追加しない。
- OpenMPなしでも直列実行できる。

既存のLinux測定入口をそのまま利用できる。

```sh
make runomp runupstream
uv run python src/cpuprofiler/benchmark.py
```

`run/run.c`の演算部分のupstreamとの差分は融合ヘッダのinclude、QKV・FFNの呼び出し置換、
通常matmulへの`schedule(runtime)`追加。`OUTPUT_*`は使用しない。
ベンチマークの`omp_schedule`の既定値は`dynamic,16`で、出力にも同じ設定を使う。
未改造upstreamは`OMP_SCHEDULE`を参照しないため、対upstreamの性能差には
分配方法の差も含まれる。比較する際は、
upstreamと融合版でCPU集合・人数・ビルドオプション・モデル・入力を揃える。

検証コードと実験結果は[dev/optブランチ](https://github.com/doss-eeic/2026-07-llama2c/tree/dev/opt)に保存する。
このブランチでは`tests/`と`results/`を含めない。

重み配置の追加機能は[hugepage.md](hugepage.md)を参照。既定は従来のmmap。
