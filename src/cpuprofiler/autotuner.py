from .benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .topology import CPUTopology

# 探索する区間
#   None   : 全体（taskset で全体の cpu を決める。評価は tok/s）
#   "body" : 出力層以外（run.c の -B。評価は body_ms_per_token）
#   "cls"  : 出力層（run.c の -C。評価は cls_ms_per_token）
SECTIONS = (None, "body", "cls")


class AutoTuner:
    def __init__(self, topology: CPUTopology, runner: BenchmarkRunner):
        self.topology = topology
        self.runner = runner
        self.history = []

    def evaluate(self, config):
        result = self.runner.run(config)
        self.history.append(result)
        return result

    @staticmethod
    def make_config(cpus, section=None, fixed=None):
        # section の cpu を cpus にした構成を作る。
        # もう片方の区間は fixed を使う。fixed が None なら cpus と同じにする
        # （区間ごとのスレッド数が同じなので、余ったスレッドがもう片方の cpu を邪魔しない）
        cpus = tuple(sorted(cpus))
        if section is None:
            # 出力層も同じスレッド数にする（output_threads の既定値 8 を使わない）
            return BenchmarkConfig(cpus, len(cpus), output_threads=len(cpus))

        other = fixed if fixed is not None else cpus
        if section == "body":
            return BenchmarkConfig(body_cpus=cpus, cls_cpus=other)
        if section == "cls":
            return BenchmarkConfig(body_cpus=other, cls_cpus=cpus)
        raise ValueError(f"Invalid section: {section}")

    @staticmethod
    def section_cpus(config, section=None):
        # 構成のうち、section の cpu を返す
        if section is None:
            return config.cpus
        if section == "body":
            return config.body_cpus
        if section == "cls":
            return config.cls_cpus
        raise ValueError(f"Invalid section: {section}")

    @staticmethod
    def score(result: BenchmarkResult, section=None):
        # 大きいほど良い値を返す
        if section is None:
            return result.tokens_per_second

        ms = (
            result.body_ms_per_token if section == "body"
            else result.cls_ms_per_token
        )
        if ms is None:
            raise RuntimeError(
                "Section timer not found. Rebuild build/omp-run from run.c"
            )
        return -ms

    def best(self, results, section=None):
        # (BenchmarkConfig, BenchmarkResult) のリストから一番良いものを返す
        return max(results, key=lambda x: self.score(x[1], section))

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

    def generate_fixed_candidates(self, args="All", section=None, fixed=None):
        if args == "All":
            return [
                self.make_config((cpu_id,), section, fixed)
                for cpu_id in self.topology.cpus
            ]
        elif args == "Sample":
            return [
                self.make_config(cpus, section, fixed)
                for cpus in [
                    (0,),
                    (0, 2),
                    (0, 1, 2, 3),
                    (4, 5, 8, 9),
                    (0, 2, 4, 8),
                ]
            ]
        else:
            raise ValueError("Invalid argument for generate_fixed_candidates")

    def generate_beam_candidates(self, pre_results, n=3, section=None, fixed=None):
        # ここで，一つ前の結果をもとに，次の候補を作成する。pre_resultsは，前回のベンチマーク結果のリスト。構成は(BenchmarkConfig, BenchmarkResult)のタプル。
        # section の cpu だけを 1 つずつ増やす。もう片方の区間は fixed のまま
        sorted_results = sorted(
            pre_results,
            key=lambda x: self.score(x[1], section),
            reverse=True
        )

        top_n_configs = [result[0] for result in sorted_results[:n]]

        new_candidates = set()

        for config in top_n_configs:
            cpus = self.section_cpus(config, section)
            for cpu_id in self.topology.cpus:
                if cpu_id not in cpus:
                    new_candidates.add(
                        self.make_config(cpus + (cpu_id,), section, fixed)
                    )
        return list(new_candidates)

    def beam_search(self, depth=0, max_depth=8, pre_results=None, n=3, section=None, fixed=None):
        # section の cpu を beam search する（section は SECTIONS を参照）。
        # 深さ d では d+1 個の cpu を使う構成を試す
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section}")

        if depth >= max_depth:
            return []

        if pre_results is None:
            if n > self.topology.n_logical_cpus:
                n = self.topology.n_logical_cpus
            if n <= 0:
                raise ValueError("n must be greater than 0")
            candidates = self.generate_fixed_candidates("All", section, fixed)
        else:
            candidates = self.generate_beam_candidates(pre_results, n, section, fixed)

        results = []
        for config in candidates:
            result = self.evaluate(config)
            results.append((config, result))

        # 再帰的に次の深さの探索を行う
        deeper_results = self.beam_search(depth + 1, max_depth, results, n, section, fixed)
        results.extend(deeper_results)

        return results

    def split_beam_search(self, max_depth_body=8, max_depth_cls=8, n=3, rounds=1):
        # 出力層とそれ以外を交互に beam search して、それぞれの cpu を決める。
        #   1. 出力層以外を探す（1 周目は出力層も候補と同じ cpu、2 周目からは見つかった出力層の cpu）
        #   2. 見つかった出力層以外の cpu に固定して、出力層を探す
        # 最後に見つかった組み合わせで全体を測り、(BenchmarkConfig, BenchmarkResult) を返す
        body_best = None
        cls_best = None

        for _ in range(rounds):
            results = self.beam_search(
                max_depth=max_depth_body, n=n, section="body", fixed=cls_best
            )
            body_best = self.best(results, "body")[0].body_cpus

            results = self.beam_search(
                max_depth=max_depth_cls, n=n, section="cls", fixed=body_best
            )
            cls_best = self.best(results, "cls")[0].cls_cpus

        config = BenchmarkConfig(body_cpus=body_best, cls_cpus=cls_best)
        return config, self.evaluate(config)
