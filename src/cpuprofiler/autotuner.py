from .benchmark import BenchmarkConfig, BenchmarkRunner
from .topology import CPUTopology, CalibratedTopology, DomainClass, DomainProfile, build_domains
from dataclasses import fields, replace
from itertools import product
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

    def _domain_kinds(self, calibrated=None):
        """(型, 物理コア列)の一覧。calibratedがあれば実測で検証したDomainClass.idを型にする。"""
        if calibrated is not None:
            return [(("class", c.id), domain.cores) for c in calibrated.classes for domain in c.domains]
        available, kinds, covered = set(self.topology.cpus), [], set()
        for group in self.topology.l2_groups + self.topology.physical_cores:
            cpus = (set(group) & available) - covered
            if not cpus:
                continue
            covered.update(cpus)
            cores = [tuple(sorted(cpus & set(core))) for core in self.topology.physical_cores]
            cores = [core for core in cores if core]
            kinds.append((tuple(sorted(map(len, cores))), cores))
        return kinds

    def _topology_candidates(self, preferred=(), calibrated=None):
        """SMT占有とL2共有を保ち、同じ型の配置を1つにまとめる。"""
        available = set(self.topology.cpus)
        clusters, preferred_key, sizes = [], [], defaultdict(int)
        for shape, cores in self._domain_kinds(calibrated):
            sizes[shape] += sum(map(len, cores))
            preferred_key.append((shape, tuple(sorted((len(core), len(set(core) & set(preferred)))
                                                     for core in cores))))
            options = {}
            for counts in product(*(range(len(core) + 1) for core in cores)):
                key = tuple(sorted((len(core), n) for core, n in zip(cores, counts)))
                mask = tuple(cpu for core, n in zip(cores, counts) for cpu in core[:n])
                options.setdefault(key, mask)
            clusters.append((shape, options))
        kinds = sorted({shape for shape, _ in clusters})
        capacities = [sizes[kind] for kind in kinds]
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

    def kasikoi_serch(self, config=None, budget=56, top_k=3, repeats=3, seed=42, calibrated=None):
        """人数全域→近傍探索→schedule/g調整→独立確認。最良の平均結果を返す。

        budgetは探索測定数。別途warmup1回と上位＋基準構成をrepeats回ずつ測る。
        configの最適化・モデル入力設定を維持し、層内と出力を同人数にする。
        calibratedを渡すと、shapeではなく実測で同型と判定したDomainだけを対称に扱う。
        キャリブレーションの測定はbudgetに含めない。
        """
        started, history_start = time.perf_counter(), len(self.history)
        cpus = tuple(sorted(self.topology.cpus))
        if not cpus or budget < len(cpus) or top_k < 1 or repeats < 1:
            raise ValueError("budget must cover all thread counts; top_k and repeats must be positive")
        config = config or BenchmarkConfig(cpus=cpus, threads=len(cpus))
        if not config.cpus or not set(config.cpus) <= set(cpus):
            raise ValueError("config.cpus must be available CPUs")
        candidates = self._topology_candidates(config.cpus, calibrated)
        rng, measured, layouts = random.Random(seed), {}, {}

        def make(mask):
            return replace(config, cpus=mask, threads=len(mask), omp_places=None)

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
        tuning = [replace(make(mask), omp_schedule=schedule, group_size=g)
                  for mask in sorted(layouts, key=layouts.get)[:top_k]
                  for schedule in dict.fromkeys((config.omp_schedule, "static", "dynamic,16", "dynamic,64"))
                  for g in dict.fromkeys(groups)]
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

    def kasikoi2_serch(self, config=None, top_k=3, repeats=3, seed=42,
                       calibrated=None, calibration_repeats=3, tolerance=0.05):
        """1スレッドから最良配置に1CPUずつ足していくgreedy探索。

        warmup → キャリブレーション(calibratedが無いとき) → CPU探索 → 補助設定調整 → 独立確認。
        追加後の配置はDomainClassで正規化し、同型の配置は1回だけ測る。
        キャリブレーションを含め、同じ構成の結果は再利用する(warmupと独立確認は毎回測る)。
        補助設定(OpenMPのschedule/wait/proc_bind、g)はCPU探索の上位top_k配置だけで探す。
        """
        started, history_start = time.perf_counter(), len(self.history)
        search = LayoutSearch(self, config, top_k, repeats, seed)
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
                            "tuned": search.tuned}
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
    """kasikoi2_serchの状態と各段階。

    measuredは探索で使った結果(BenchmarkConfig→結果、キャッシュから再利用したものを含む)、layoutsはCPU配置→ms/token、
    stagesはスレッド数ごとの(測定した配置→ms/token)、tunedは上位配置ごとの補助設定調整後の構成。
    """
    # CPU配置が決まった後に試す補助設定。座標降下でこの順に1項目ずつ最良値に固定し、
    # 1周で何か変わればTUNE_PASSES周まで繰り返す(scheduleとg、waitとspincountの相互作用を拾う)。
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
        self.stages, self.tuned = [], []
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
                with self.tuner.runner.model_path.open("rb") as model:
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

    def tune_runtime(self):
        """CPU探索の上位top_k配置それぞれで、補助設定を1項目ずつ最良値に固定していく(座標降下)。"""
        options = self.runtime_options()
        for mask in sorted(self.layouts, key=self.layouts.get)[:self.top_k]:
            best = self.make(mask)
            for _ in range(self.TUNE_PASSES):
                start = best
                for name, values in options.items():
                    if name == "omp_places":
                        values = [self.places(mask, order) for order in values]
                    cases = list(dict.fromkeys(replace(best, **{name: value}) for value in values))
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
