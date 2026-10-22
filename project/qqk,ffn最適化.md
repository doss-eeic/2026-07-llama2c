
# qqk,ffn-opt

## build
```shell
mkdir -p build
gcc -Ofast -march=native -fopenmp run.c -lm -o build/run-fused
```

## 実行
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


# 通常版
## build
```shell
mkdir -p build/upstream
git show 350e04fe35433e6d2941dce5a1f53308f87058eb:run.c \
    > build/upstream/run.c

gcc -Ofast -march=native -fopenmp \
    build/upstream/run.c -lm -o build/run-upstream
```

## 実行
```shell
env OMP_NUM_THREADS=10 \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close \
    OMP_PLACES='{0},{2},{4},{5},{6},{7},{8},{9},{10},{11}' \
    taskset -c 0,2,4,5,6,7,8,9,10,11 \
    ./build/run-upstream build/stories15M.bin \
    -z tokenizer.bin -n 256 -t 0 -s 42 -i "Once upon a time"
```