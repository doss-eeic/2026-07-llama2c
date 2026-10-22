# 二段階argmax

`dev/sakurada`の`d7e1a399`から、FP32と二段階argmaxの切り替えだけを移植した。
本体は`src/classifier.h`に分離し、近似のみ・検証モードや層別計時は含めない。

- `-a 0`（既定）：従来のFP32出力層。
- `-a 1`：int8重みで全語彙を評価し、最大になり得る候補だけFP32で再計算。
- `-g 144`（既定）：量子化グループの要素数。正数かつモデルのdimを割り切る値。

起動時に出力重みをグループ単位でint8に量子化し、scaleを保存する。
各トークンで近似スコアと誤差幅を計算し、上限が全行の下限の最大値に届く行を残す。
誤差幅は移植元と同じ`0.51 * sum(scale * sum(abs(x))) + 1e-6`を使う。
残った行をFP32で再計算し、それ以外を`-FLT_MAX`にする。
これは全logitsを復元する機能ではなく、greedyの選択を維持するための処理。
浮動小数点演算の誤差余裕も移植元に従っており、任意の入力に対する厳密な証明は追加していない。

`-t 0`専用。温度が0以外なら注意を表示してFP32に戻す。
量子化バッファの確保はLinux用で、`-w 1`時にはこちらにもTHPを要求する。
既存のQKV・FFN融合と共通`OMP_SCHEDULE`を維持する。

```sh
env OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE OMP_SCHEDULE='dynamic,16' \
    ./build/run-fused build/stories15M.bin -t 0 -a 1 -g 144 -w 1 -n 256 -s 42
```

Pythonでは`BenchmarkConfig(two_stage_argmax=True, group_size=144, temperature=0.0, ...)`。
未改造upstreamでは`two_stage_argmax=False`を使う。
