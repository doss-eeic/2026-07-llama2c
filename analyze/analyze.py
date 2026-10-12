import re
from pathlib import Path
from collections import defaultdict
from statistics import median

THREADS = [1, 2, 4, 8]
RUNS = 5
BENCHMARK_DIR = Path("benchmark")

STAGES = {
    "001": "embedding-lookup",
    "002": "attention-norm",
    "003": "qkv-projection",
    "004": "apply-rope",
    "005": "compute-attention",
    "006": "attention-output",
    "007": "ffn-norm",
    "008": "ffn-up-projection",
    "009": "ffn-swiglu",
    "010": "ffn-down-projection",
    "011": "final-projection",
    "102": "layer-compute-total",
    "103": "forward-compute-total",
}

# 合計時間との二重計上を避けるため、個別処理のみ
DETAIL_STAGES = [f"{i:03d}" for i in range(1, 12)]
LAYER_STAGES = [f"{i:03d}" for i in range(2, 11)]

pattern = re.compile(
    r"\b(\d{3}):[\w-]+:{1,2}\s*"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s+seconds"
)


def parse_log(path):
    data = defaultdict(list)

    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                stage_id = match.group(1)
                elapsed = float(match.group(2))

                if stage_id in STAGES:
                    data[stage_id].append(elapsed)

    return data


def median_per_call(runs, stage_id):
    """処理1回当たりの平均時間の中央値 [µs]"""
    values = [
        sum(data[stage_id]) / len(data[stage_id]) * 1e6
        for data in runs
    ]
    return median(values)


def median_per_token(runs, stage_id):
    """1トークン当たりの処理時間の中央値 [ms]"""
    values = [
        sum(data[stage_id]) / len(data["103"]) * 1e3
        for data in runs
    ]
    return median(values)


# ========================================
# ログ読み込み
# ========================================

results = {}
all_counts = set()

for threads in THREADS:
    runs = []

    for i in range(1, RUNS + 1):
        path = BENCHMARK_DIR / f"t{threads}_r{i}.log"

        data = parse_log(path)

        # 必須ステージの存在確認
        missing = [s for s in STAGES if not data[s]]
        if missing:
            raise ValueError(
                f"{path}: Missing stages: {missing}"
            )

        tokens = len(data["103"])
        layers = len(data["102"])

        # 1トークンにつき1回の処理を確認
        for stage in ["001", "011"]:
            if len(data[stage]) != tokens:
                raise ValueError(
                    f"{path}: stage {stage} count mismatch"
                )

        # 各レイヤーで1回の処理を確認
        for stage in LAYER_STAGES:
            if len(data[stage]) != layers:
                raise ValueError(
                    f"{path}: stage {stage} count mismatch"
                )

        if layers % tokens != 0:
            raise ValueError(f"{path}: Invalid layer count")

        n_layers = layers // tokens

        all_counts.add((tokens, layers))
        runs.append(data)

        print(
            f"threads={threads}, run={i}, "
            f"layers={layers}, tokens={tokens}, "
            f"n_layers={n_layers}"
        )

    results[threads] = runs

if len(all_counts) != 1:
    print("\nWARNING: Token/layer counts differ across runs!")


# ========================================
# 1. 処理1回当たりの平均時間
# ========================================

print("\n=== Median time per call (µs) ===")

print(f"{'Stage':<28}", end="")
for t in THREADS:
    print(f"{t:>12} threads", end="")
print()

for stage_id, stage_name in STAGES.items():
    print(f"{stage_name:<28}", end="")

    for threads in THREADS:
        value = median_per_call(results[threads], stage_id)
        print(f"{value:>12.3f}", end="")

    print()


# ========================================
# 2. 1トークン当たりの処理時間
# ========================================

print("\n=== Time per token (ms) ===")

print(f"{'Stage':<28}", end="")
for t in THREADS:
    print(f"{t:>12} threads", end="")
print()

for stage_id, stage_name in STAGES.items():
    print(f"{stage_name:<28}", end="")

    for threads in THREADS:
        value = median_per_token(results[threads], stage_id)
        print(f"{value:>12.4f}", end="")

    print()


# ========================================
# 3. Forward全体に占める割合
# ========================================

print("\n=== Forward time breakdown (%) ===")

print(f"{'Stage':<28}", end="")
for t in THREADS:
    print(f"{t:>12} threads", end="")
print()

for stage_id in DETAIL_STAGES:
    stage_name = STAGES[stage_id]
    print(f"{stage_name:<28}", end="")

    for threads in THREADS:
        forward_time = median_per_token(
            results[threads], "103"
        )
        stage_time = median_per_token(
            results[threads], stage_id
        )

        percentage = stage_time / forward_time * 100

        print(f"{percentage:>12.2f}", end="")

    print()


# ========================================
# 4. Layer / Final / Embedding の構成比
# ========================================

print("\n=== Major component breakdown ===")

for threads in THREADS:
    runs = results[threads]

    forward = median_per_token(runs, "103")
    layers = median_per_token(runs, "102")
    embedding = median_per_token(runs, "001")
    final = median_per_token(runs, "011")

    print(f"\nThreads: {threads}")
    print(f"  Forward:   {forward:.4f} ms (100%)")

    for name, value in [
        ("Layers", layers),
        ("Embedding", embedding),
        ("Final", final),
    ]:
        print(
            f"  {name:<10} {value:.4f} ms "
            f"({value / forward * 100:.2f}%)"
        )

    # 各処理時間の合計とForward時間が整合するか
    accounted = layers + embedding + final
    print(f"  Difference: {forward - accounted:.6f} ms")


# ========================================
# 5. スレッド数による高速化率
# ========================================

print("\n=== Speedup ===")

baseline_layer = median_per_call(results[1], "102")
baseline_forward = median_per_token(results[1], "103")

print(
    f"{'Threads':<10}"
    f"{'Layer speedup':>18}"
    f"{'Forward speedup':>20}"
)

for threads in THREADS:
    layer = median_per_call(results[threads], "102")
    forward = median_per_token(results[threads], "103")

    print(
        f"{threads:<10}"
        f"{baseline_layer / layer:>18.3f}x"
        f"{baseline_forward / forward:>19.3f}x"
    )