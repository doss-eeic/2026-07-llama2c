from .benchmarkconfig import BenchmarkConfig
from .benchmark import BenchmarkRunner
from .topology import CPUTopology, CalibratedTopology, DomainClass, DomainProfile, build_domains
from dataclasses import fields, replace
import math
import random
import statistics
import struct
import time
from collections import defaultdict


def mean_result(rows):
    """同じ構成の複数回の結果を、ms/tokenの平均を持つ1つの結果にまとめる。"""
    return replace(rows[-1], tokens_per_second=1000 / statistics.fmean(r.ms_per_token for r in rows))


class AutoTuner:
    def __init__(self, topology: CPUTopology, runner: BenchmarkRunner):
        self.topology = topology
        self.runner = runner
        self.history = []  # 実際に走らせた測定だけ
        self.cache = {}  # BenchmarkConfig→結果。evaluate(cached=True)でだけ読み書きする

    def evaluate(self, config, cached=False):
        """cached=Trueなら同じ構成の既知の結果(キャリブレーションの平均を含む)を再利用する。"""
        if cached and config in self.cache:
            return self.cache[config]
        result = self.runner.run(config)
        self.history.append(result)
        if cached:
            self.cache[config] = result
        return result

    def calibrate(self, config=None, repeats=3, tolerance=0.05, seed=0):
        calibrated = DomainCalibrator(self.topology, self.evaluate, config, repeats, tolerance, seed).calibrate()
        self.cache.update(calibrated.results)
        return calibrated

    def kasikoi2_search(self, config=None, top_k=3, repeats=3, seed=42,
                       calibrated=None, calibration_repeats=3, tolerance=0.05):
        """1スレッドから最良配置に1CPUずつ足していくgreedy探索。

        warmup → キャリブレーション(calibratedが無いとき) → CPU探索 → 補助設定調整 → 独立確認。
        追加後の配置はDomainClassで正規化し、同型の配置は1回だけ測る。
        キャリブレーションを含め、同じ構成の結果は再利用する(warmupと独立確認は毎回測る)。
        補助設定(OpenMPのschedule/wait/spincount/proc_bind/dynamic/places、g)は、最良配置で項目ごとに
        上位値へ絞り込んでから、CPU探索の上位top_k配置で組み合わせを探す。
        """
        if calibration_repeats < 1 or tolerance <= 0:
            raise ValueError("calibration_repeats and tolerance must be positive")
        started, history_start = time.perf_counter(), len(self.history)
        search = LayoutSearch(self, config, top_k, repeats, seed)
        self.cache.clear()
        search.warmup()
        if calibrated is None:
            calibrated = self.calibrate(search.config, calibration_repeats, tolerance, seed)
        self.cache.update(calibrated.results)  # 1CPU構成と同型候補のDomain全体はCPU探索で測り直さない
        search.prepare(calibrated)
        search.grow()
        search.tune_runtime()
        results = search.confirm()
        self.last_search = {"seconds": time.perf_counter() - started,
                            "measurements": len(self.history) - history_start,
                            "exploration": len(search.measured), "baseline": search.baseline,
                            "confirmed": results, "calibrated": calibrated, "stages": search.stages,
                            "tuned": search.tuned, "screened": search.screened}
        return results[0]


class DomainCalibrator:
    """構造(shape)が同じDomainを実測し、性能も揃うものだけを同型クラスにまとめる。

    1. 各論理CPUを1スレッドで測る(repeats巡、巡ごとに順序をシャッフル)。
    2. 同じshapeのDomainが複数ある場合だけ、Domain全体も全CPUで測る。
    3. 同じshapeの中で、単一CPU性能(ソートしたCPU別平均)と全体性能の差が
       すべてのペアでtolerance以内のものを1クラスにする(完全連結なので推移性の問題を避ける)。
       ばらつきがtoleranceを超えるDomainは同型と言えないので単独クラスに残す。
    """

    def __init__(self, topology, benchmark, config=None, repeats=3, tolerance=0.05, seed=0):
        if repeats < 1 or tolerance <= 0:
            raise ValueError("repeats and tolerance must be positive")
        self.topology, self.benchmark = topology, benchmark
        cpus = tuple(sorted(topology.cpus))
        self.config = config or BenchmarkConfig(cpus=cpus, threads=len(cpus))
        self.repeats, self.tolerance, self.rng = repeats, tolerance, random.Random(seed)
        self.results = {}

    def calibrate(self):
        domains = build_domains(self.topology)
        by_shape = defaultdict(list)
        for domain in domains:
            by_shape[domain.shape].append(domain)
        single = self._measure([(cpu,) for domain in domains for cpu in domain.cpus])
        full = self._measure([domain.cpus for group in by_shape.values() if len(group) > 1
                              for domain in group])
        profiles = {domain.id: self._profile(domain, single, full) for domain in domains}
        groups = [group for _, members in sorted(by_shape.items())
                  for group in self._classify(members, profiles)]
        groups.sort(key=lambda group: min(domain.id for domain in group))
        classes = tuple(DomainClass(i, tuple(sorted(group, key=lambda d: d.id)))
                        for i, group in enumerate(groups))
        return CalibratedTopology(self.topology, domains, profiles, classes, self.results)

    def _measure(self, cases):
        runs = {cpus: [] for cpus in cases}
        for _ in range(self.repeats):
            order = list(runs)
            self.rng.shuffle(order)  # ターボや温度の時系列バイアスをCPU番号に乗せない
            for cpus in order:
                case = replace(self.config, cpus=cpus, threads=len(cpus), omp_places=None)
                runs[cpus].append(self.benchmark(case))
        for rows in runs.values():
            self.results[rows[0].config] = mean_result(rows)
        return {cpus: [r.ms_per_token for r in rows] for cpus, rows in runs.items()}

    @staticmethod
    def _std(values):
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def _profile(self, domain, single, full):
        runs = [single[(cpu,)] for cpu in domain.cpus]
        per_cpu = tuple(statistics.fmean(r) for r in runs)
        whole = full.get(domain.cpus)
        return DomainProfile(
            domain_id=domain.id,
            single_mean=statistics.fmean(per_cpu),
            single_std=math.sqrt(statistics.fmean(self._std(r) ** 2 for r in runs)),
            per_cpu_means=per_cpu,
            full_mean=None if whole is None else statistics.fmean(whole),
            full_std=None if whole is None else self._std(whole),
        )

    def _noisy(self, p):
        return (p.single_std / p.single_mean > self.tolerance
                or p.full_mean is not None and p.full_std / p.full_mean > self.tolerance)

    def _similar(self, a, b):
        def close(x, y):
            return abs(math.log(x / y)) <= self.tolerance
        if self._noisy(a) or self._noisy(b):
            return False
        if not all(close(x, y) for x, y in zip(sorted(a.per_cpu_means), sorted(b.per_cpu_means))):
            return False
        return a.full_mean is None or b.full_mean is None or close(a.full_mean, b.full_mean)

    def _classify(self, domains, profiles):
        groups = []
        for domain in sorted(domains, key=lambda d: profiles[d.id].single_mean):
            for group in groups:
                if all(self._similar(profiles[domain.id], profiles[other.id]) for other in group):
                    group.append(domain)
                    break
            else:
                groups.append([domain])
        return groups


class LayoutSearch:
    """kasikoi2_searchの状態と各段階。

    measuredは探索で使った結果(BenchmarkConfig→結果、キャッシュから再利用したものを含む)、layoutsはCPU配置→ms/token、
    stagesはスレッド数ごとの(測定した配置→ms/token)、tunedは上位配置ごとの補助設定調整後の構成。
    """
    # CPU配置が決まった後に試す補助設定。
    # 1. スクリーニング: CPU探索の最良配置で、既定設定から1項目だけ変えて全候補を測り、項目ごとに上位TOP_VALUES個を残す。
    # 2. 組み合わせ: 上位top_k配置それぞれで、残した値だけを座標降下で1項目ずつ最良値に固定する。
    #    1周で何か変わればTUNE_PASSES周まで繰り返す(scheduleとg、waitとspincountの相互作用を拾う)。
    RUNTIME_OPTIONS = {
        "omp_schedule": ("static", "static,1", "static,4", "static,16",
                         "dynamic,1", "dynamic,4", "dynamic,8", "dynamic,16", "dynamic,32", "dynamic,64",
                         "guided", "guided,1", "guided,2", "guided,4", "guided,8", "guided,16", "auto"),
        "group_size": None,  # two_stage_argmax時だけ、dimの約数(MIN_GROUP以上)すべて
        "omp_wait_policy": (None, "ACTIVE", "PASSIVE"),
        "gomp_spincount": (None, 0, 1000, 10000, 100000, 1000000, 10000000),
        # OMP_PLACESは各CPUを1つずつ並べるので、proc_bindはclose/spreadで差が出ずfalse(固定なし)だけ比べる
        "omp_proc_bind": ("close", "false"),
        "omp_dynamic": (False, True),
        # スレッドを置くCPUの順。0番スレッド(マスター)が先頭のCPUに乗る。速さはキャリブレーションの1CPU値
        "omp_places": ("id", "fast_first", "slow_first"),
    }
    MIN_GROUP = 16
    TOP_VALUES = 3
    TUNE_PASSES = 2

    def __init__(self, tuner, config, top_k, repeats, seed):
        cpus = tuple(sorted(tuner.topology.cpus))
        if not cpus or top_k < 1 or repeats < 1:
            raise ValueError("top_k and repeats must be positive")
        config = config or BenchmarkConfig(cpus=cpus, threads=len(cpus))
        if not config.cpus or not set(config.cpus) <= set(cpus):
            raise ValueError("config.cpus must be available CPUs")
        # CPU探索(warmup・キャリブレーションを含む)はOpenMP設定をBenchmarkConfigの既定値に戻して行い、
        # 入力configの値は補助設定の候補と基準構成にだけ使う。gはモデル依存なので入力値のまま。
        defaults = {f.name: f.default for f in fields(BenchmarkConfig)}
        self.input = config
        self.tuner, self.cpus = tuner, cpus
        self.config = replace(config, **{name: defaults[name] for name in self.RUNTIME_OPTIONS
                                         if name != "group_size"})
        self.top_k, self.repeats = top_k, repeats
        self.rng, self.measured, self.layouts = random.Random(seed), {}, {}
        self.stages, self.tuned, self.screened = [], [], {}
        self.baseline = replace(config, cpus=tuple(sorted(config.cpus)), threads=len(config.cpus), omp_places=None)

    def make(self, mask, **changes):
        return replace(self.config, cpus=mask, threads=len(mask), omp_places=None, **changes)

    def measure(self, case):
        result = self.tuner.evaluate(case, cached=True)
        self.measured[case] = result
        return result

    def warmup(self):
        self.tuner.evaluate(self.make(tuple(self.config.cpus)))  # 順位に使わない

    def prepare(self, calibrated):
        self.domains = [(c.id, domain) for c in calibrated.classes for domain in c.domains]
        self.single_ms = {cpu: mean for domain in calibrated.domains
                          for cpu, mean in zip(domain.cpus, calibrated.profiles[domain.id].per_cpu_means)}

    def key(self, mask):
        """同じDomainClass内でのDomainの入れ替え、同じ大きさのコアの入れ替えを同一視する。"""
        used = set(mask)
        return tuple(sorted(
            (class_id, tuple(sorted((len(core), len(used & set(core))) for core in domain.cores)))
            for class_id, domain in self.domains))

    def children(self, parent):
        """parentに1CPU足した配置。同型は1つにまとめる。"""
        found = {}
        for cpu in self.cpus:
            if cpu not in parent:
                mask = tuple(sorted(parent + (cpu,)))
                found.setdefault(self.key(mask), mask)
        return list(found.values())

    def grow(self):
        best = ()
        for _ in self.cpus:
            stage = {}
            for mask in self.children(best):
                stage[mask] = self.layouts[mask] = self.measure(self.make(mask)).ms_per_token
            self.stages.append(stage)
            best = min(stage, key=stage.get)

    def runtime_options(self):
        """試す補助設定。入力configの値も候補に入れる(omp_placesは配置ごとに作るので除く)。"""
        options = {}
        for name, values in self.RUNTIME_OPTIONS.items():
            if name == "group_size":
                if not self.config.two_stage_argmax:
                    continue
                with self.config.model.open("rb") as model:
                    dim, = struct.unpack("i", model.read(4))
                values = [g for g in range(self.MIN_GROUP, dim + 1) if dim % g == 0]
            if name != "omp_places":
                values = (getattr(self.input, name), *values)
            options[name] = tuple(dict.fromkeys(values))
        return options

    def places(self, mask, order):
        ordered = sorted(mask, key=self.single_ms.get, reverse=order == "slow_first")
        if order == "id" or ordered == sorted(mask):
            return None  # BenchmarkRunnerがCPU番号順に並べる
        return ",".join(f"{{{cpu}}}" for cpu in ordered)

    def variants(self, base, name, values):
        """baseの1項目だけ変えた構成。値→構成で、同じ構成になる値は先の1つだけ残す。"""
        cases = {}
        for value in values:
            setting = self.places(base.cpus, value) if name == "omp_places" else value
            case = replace(base, **{name: setting})
            if case not in cases.values():
                cases[value] = case
        return cases

    def screen(self, mask, options):
        """既定設定から1項目だけ変えて全候補を測り、項目ごとに速い順の上位TOP_VALUES個の値を残す。"""
        base = self.make(mask)
        for name, values in options.items():
            cases = self.variants(base, name, values)
            order = list(cases)
            self.rng.shuffle(order)
            ms = {value: self.measure(cases[value]).ms_per_token for value in order}
            self.screened[name] = tuple(sorted(ms, key=ms.get)[:self.TOP_VALUES])

    def tune_runtime(self):
        masks = sorted(self.layouts, key=self.layouts.get)[:self.top_k]
        self.screen(masks[0], self.runtime_options())
        for mask in masks:
            best = self.make(mask)
            for _ in range(self.TUNE_PASSES):
                start = best
                for name, values in self.screened.items():
                    cases = list(dict.fromkeys((best, *self.variants(best, name, values).values())))
                    self.rng.shuffle(cases)
                    best = min(cases, key=lambda case: self.measure(case).ms_per_token)
                if best == start:
                    break
            self.tuned.append(best)

    def confirm(self):
        """調整後の各配置と基準構成を、巡ごとに順序を変えてrepeats回測り直し、平均で順位を付ける。"""
        finalists = list(dict.fromkeys((*self.tuned, self.baseline)))
        confirmed = {case: [] for case in finalists}
        for _ in range(self.repeats):
            self.rng.shuffle(finalists)
            for case in finalists:
                confirmed[case].append(self.tuner.evaluate(case))
        return sorted(map(mean_result, confirmed.values()), key=lambda result: result.ms_per_token)
