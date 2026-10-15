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

    def search(self):
        candidates = self.generate_fixed_candidates()

        best = None

        for config in candidates:
            result = self.evaluate(config)

            if (
                best is None
                or result.tokens_per_second > best.tokens_per_second
            ):
                best = result

        return best

    def generate_fixed_candidates(self):
        return [
            BenchmarkConfig((0,), 1),
            BenchmarkConfig((0, 2), 2),
            BenchmarkConfig((0, 1, 2, 3), 4),
            BenchmarkConfig((4, 5, 8, 9), 4),
            BenchmarkConfig((0, 2, 4, 8), 4),
        ]