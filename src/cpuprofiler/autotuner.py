from .benchmark import BenchmarkConfig, BenchmarkRunner
from .topology import CPUTopology

class AutoTuner:
    def __init__(self, topology: CPUTopology, runner: BenchmarkRunner):
        self.topology = topology
        self.runner = runner
        self.history = []

    def evaluate(self, config):
        result = self.runner.run(config)
        self.history.append(result)
        return result

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