from .benchmark import BenchmarkConfig, BenchmarkRunner
from .topology import CPUTopology
from dataclasses import replace
from itertools import product
import random
import struct
import time

class AutoTuner:
    def __init__(self, topology: CPUTopology, runner: BenchmarkRunner):
        self.topology = topology
        self.runner = runner
        self.history = []

    def evaluate(self, config):
        result = self.runner.run(config)
        self.history.append(result)
        return result

    def _topology_candidates(self, preferred=()):
        """SMT占有とL2共有を保ち、同じ型の配置を1つにまとめる。"""
        available = set(self.topology.cpus)
        clusters, preferred_key, covered = [], [], set()
        for group in self.topology.l2_groups + self.topology.physical_cores:
            cpus = (set(group) & available) - covered
            if not cpus:
                continue
            covered.update(cpus)
            cores = [tuple(sorted(cpus & set(core))) for core in self.topology.physical_cores]
            cores = [core for core in cores if core]
            shape = tuple(sorted(map(len, cores)))
            preferred_key.append((shape, tuple(sorted((len(core), len(set(core) & set(preferred)))
                                                     for core in cores))))
            options = {}
            for counts in product(*(range(len(core) + 1) for core in cores)):
                key = tuple(sorted((len(core), n) for core, n in zip(cores, counts)))
                mask = tuple(cpu for core, n in zip(cores, counts) for cpu in core[:n])
                options.setdefault(key, mask)
            clusters.append((shape, options))
        kinds = sorted({shape for shape, _ in clusters})
        capacities = [sum(sum(shape) for shape, _ in clusters if shape == kind) for kind in kinds]
        preferred_key = tuple(sorted(preferred_key))
        representatives = {}
        for choices in product(*(options.items() for _, options in clusters)):
            key = tuple(sorted((shape, choice[0]) for (shape, _), choice in zip(clusters, choices)))
            cpus = tuple(sorted(cpu for _, mask in choices for cpu in mask))
            if cpus and key not in representatives:
                if key == preferred_key:
                    cpus = tuple(sorted(preferred))
                counts = [sum(len(mask) for (shape, _), (_, mask) in zip(clusters, choices)
                              if shape == kind) / capacity for kind, capacity in zip(kinds, capacities)]
                representatives[key] = (cpus, (*counts, len(cpus) / len(available)))
        return dict(representatives.values())

    def _make_config(self, config, mask):
        return replace(config, cpus=mask, threads=len(mask), omp_places=None)

    def _tuning_candidates(self, config, masks, groups):
        return [replace(self._make_config(config, mask), omp_schedule=schedule, group_size=g)
                for mask in masks
                for schedule in dict.fromkeys((config.omp_schedule, "static", "dynamic,16", "dynamic,64"))
                for g in dict.fromkeys(groups)]

    def kasikoi_serch(self, config=None, budget=56, top_k=3, repeats=3, seed=42):
        """人数全域→近傍探索→schedule/g調整→独立確認。最良の平均結果を返す。

        budgetは探索測定数。別途warmup1回と上位＋基準構成をrepeats回ずつ測る。
        configの最適化・モデル入力設定を維持し、層内と出力を同人数にする。
        """
        started, history_start = time.perf_counter(), len(self.history)
        cpus = tuple(sorted(self.topology.cpus))
        if not cpus or budget < len(cpus) or top_k < 1 or repeats < 1:
            raise ValueError("budget must cover all thread counts; top_k and repeats must be positive")
        config = config or BenchmarkConfig(cpus=cpus, threads=len(cpus))
        if not config.cpus or not set(config.cpus) <= set(cpus):
            raise ValueError("config.cpus must be available CPUs")
        candidates = self._topology_candidates(config.cpus)
        rng, measured, layouts = random.Random(seed), {}, {}

        def make(mask):
            return self._make_config(config, mask)

        def measure(case):
            result = self.evaluate(case)
            measured[case] = result
            return result

        self.evaluate(make(tuple(config.cpus)))  # 順位に使わないウォームアップ
        for n in range(1, len(cpus) + 1):
            mask = rng.choice([mask for mask in candidates if len(mask) == n])
            if n == len(config.cpus):
                mask = tuple(sorted(config.cpus))  # 既知の固定構成も残す
            layouts[mask] = measure(make(mask)).ms_per_token

        cpu_budget = max(len(cpus), budget * 3 // 4)
        while len(measured) < cpu_budget:
            remaining = [mask for mask in candidates if mask not in layouts]
            if not remaining:
                break
            leaders = sorted(layouts, key=layouts.get)[:top_k]
            nearby = [mask for mask in remaining
                      if any(len(set(mask) ^ set(parent)) <= 2 for parent in leaders)]
            pool = nearby + rng.sample(remaining, min(8, len(remaining)))
            rng.shuffle(pool)  # 同人数の配置にCPU番号順の優先を付けない

            def predict(mask):
                # L2は候補生成に使い、予測には種類別の人数だけを使う。
                def distance(other):
                    return sum((a - b) ** 2 for a, b in zip(candidates[mask], candidates[other]))
                known = sorted((other for other in layouts if other in candidates), key=distance)[:3]
                weights = [1 / (0.05 + distance(other)) for other in known]
                return sum(layouts[other] * w for other, w in zip(known, weights)) / sum(weights)

            mask = rng.choice(remaining) if len(measured) % 5 == 0 else min(pool, key=predict)
            layouts[mask] = measure(make(mask)).ms_per_token

        groups = [config.group_size]
        if config.two_stage_argmax and len(measured) < budget:
            with self.runner.model_path.open("rb") as model:
                dim, = struct.unpack("i", model.read(4))
            groups += [g for g in (96, 144, 288) if dim % g == 0]
        tuning = self._tuning_candidates(config, sorted(layouts, key=layouts.get)[:top_k], groups)
        rng.shuffle(tuning)
        for case in tuning:
            if len(measured) >= budget:
                break
            if case not in measured:
                measure(case)

        finalists = sorted(measured, key=lambda case: measured[case].ms_per_token)[:top_k]
        baseline = make(tuple(sorted(config.cpus)))
        if baseline not in finalists:
            finalists.append(baseline)
        confirmed = {case: [] for case in finalists}
        for _ in range(repeats):
            rng.shuffle(finalists)
            for case in finalists:
                confirmed[case].append(self.evaluate(case))
        results = [replace(rows[-1], tokens_per_second=1000 / (sum(r.ms_per_token for r in rows) / len(rows)))
                   for rows in confirmed.values()]
        results.sort(key=lambda result: result.ms_per_token)
        self.last_search = {"seconds": time.perf_counter() - started,
                            "measurements": len(self.history) - history_start,
                            "exploration": len(measured), "baseline": baseline, "confirmed": results}
        return results[0]

    def search(self, args="Sample"):
        candidates = self.generate_fixed_candidates(args)

        best = None

        for config in candidates:
            result = self.evaluate(config)

            if (
                best is None
                or result.tokens_per_second > best.tokens_per_second
            ):
                best = result

        return best

    def generate_fixed_candidates(self, args="All"):
        if args == "All":
            return [BenchmarkConfig((cpu_id,), 1) for cpu_id in self.topology.cpus]
        elif args == "Sample":
            return [
                BenchmarkConfig((0,), 1),
                BenchmarkConfig((0, 2), 2),
                BenchmarkConfig((0, 1, 2, 3), 4),
                BenchmarkConfig((4, 5, 8, 9), 4),
                BenchmarkConfig((0, 2, 4, 8), 4),
            ]
        else:
            raise ValueError("Invalid argument for generate_fixed_candidates")

    def generate_beam_candidates(self,pre_results, n=3):
        # ここで，一つ前の結果をもとに，次の候補を作成する。pre_resultsは，前回のベンチマーク結果のリスト。構成は(BenchmarkConfig, BenchmarkResult)のタプル。
        sorted_results = sorted(pre_results, key=lambda x: x[1].tokens_per_second, reverse=True)

        top_n_configs = [result[0] for result in sorted_results[:n]]

        new_candidates = set()

        for config in top_n_configs:
            for cpu_id in self.topology.cpus:
                if cpu_id not in config.cpus:
                    new_cpus = tuple(sorted(config.cpus + (cpu_id,)))
                    new_threads = min(len(new_cpus), self.topology.n_logical_cpus)

                    new_candidates.add(BenchmarkConfig(new_cpus, new_threads))
        return list(new_candidates)
    
    def beam_search(self, depth=0, max_depth=4, pre_results=None, n=3):
        if depth >= max_depth:
            return []

        if pre_results is None:
            if n > self.topology.n_logical_cpus:
                n = self.topology.n_logical_cpus
            if n <= 0:
                raise ValueError("n must be greater than 0")
            candidates = self.generate_fixed_candidates()
        else:
            candidates = self.generate_beam_candidates(pre_results, n)

        results = []
        for config in candidates:
            result = self.evaluate(config)
            results.append((config, result))

        # 再帰的に次の深さの探索を行う
        deeper_results = self.beam_search(depth + 1, max_depth, results)
        results.extend(deeper_results)

        return results
