"""Deployable CPU layouts and an incremental, topology-aware latency search.

Only the standard library is needed. The objective is log latency relative to a
contemporaneous control. No measurements of unqueried candidates enter a policy.
"""
from dataclasses import asdict, dataclass
import itertools
import math
import random

from .topology import CPUTopology, read


@dataclass(frozen=True)
class Machine:
    performance: tuple[tuple[int, ...], ...]
    efficient_l2: tuple[tuple[int, ...], ...]
    low_power: tuple[int, ...]
    evidence: tuple[dict, ...] = ()

    @classmethod
    def discover(cls):
        """SMT identifies P cores; frequency tiers distinguish E and LP-E here.

        Fail on ambiguous layouts rather than silently assuming Intel CPU IDs.
        Frequency is a classification hint, not a model of measured performance.
        """
        topology = CPUTopology()
        if len({c.package_id for c in topology.cpus.values()}) != 1:
            raise ValueError('This search space requires a single CPU package')
        p = tuple(tuple(g) for g in topology.physical_cores if len(g) > 1)
        if not p:
            raise ValueError('No visible SMT P cores; provide an explicit Machine')
        singles = [g[0] for g in topology.physical_cores if len(g) == 1]
        freq = {i: int(read(topology.cpus[i].path / 'cpufreq/cpuinfo_max_freq', '0')) for i in singles}
        tiers = sorted(set(freq.values()))
        if not singles or 0 in tiers or len(tiers) > 2:
            raise ValueError('Ambiguous E/LP-E frequency tiers; provide an explicit Machine')
        e = {i for i in singles if freq[i] == tiers[-1]}
        low = tuple(i for i in singles if i not in e)
        groups = tuple(tuple(i for i in g if i in e) for g in topology.l2_groups if set(g) & e)
        if set(itertools.chain.from_iterable(groups)) != e:
            raise ValueError('Incomplete E-core L2 topology')
        evidence = tuple({'cpu': i, 'siblings': c.core_cpus_list, 'l2': c.l2_cache,
                          'max_freq_khz': read(c.path / 'cpufreq/cpuinfo_max_freq'),
                          'package': c.package_id, 'core': c.core_id}
                         for i, c in topology.cpus.items())
        return cls(p, groups, low, evidence)

    @property
    def p_primary(self):
        return tuple(g[0] for g in self.performance)

    @property
    def p_smt(self):
        return tuple(i for g in self.performance for i in g[1:])

    @property
    def e(self):
        return tuple(itertools.chain.from_iterable(self.efficient_l2))


@dataclass(frozen=True)
class Candidate:
    layer: tuple[int, ...]
    output: tuple[int, ...]
    backend: str
    dynamic: int
    output_dynamic: int
    row4: int
    qkv: int = 1
    ffn: int = 1

    def __post_init__(self):
        if (not self.layer or len(set(self.layer)) != len(self.layer)
                or len(set(self.output)) != len(self.output)
                or not set(self.layer) <= set(self.output)):
            raise ValueError('Layer CPUs must be a nonempty subset of distinct output CPUs')
        if self.backend not in ('openmp', 'pthread'):
            raise ValueError('Unsupported search backend')
        if any(v not in (0, 1) for v in (self.dynamic, self.output_dynamic, self.row4, self.qkv, self.ffn)):
            raise ValueError('Search switches must be binary')

    @property
    def cpus(self):
        # Both implementations use the prefix for layers and the full pool for output.
        return (*self.layer, *(i for i in self.output if i not in self.layer))

    @property
    def name(self):
        layer = '-'.join(map(str, self.layer))
        extra = '-'.join(str(i) for i in self.output if i not in self.layer) or 'same'
        return f'L{layer}_O{extra}_{self.backend}_d{self.dynamic}o{self.output_dynamic}r{self.row4}q{self.qkv}f{self.ffn}'

    @classmethod
    def from_dict(cls, obj):
        return cls(**{**obj, 'layer': tuple(obj['layer']), 'output': tuple(obj['output'])})


def layouts(machine):
    """Keep every physical P core, sweep E cores at two-core granularity.

    Compact/spread E placements cover cache sharing; SMT is all-or-none. Keep
    LP-E out of layers except the all-CPU boundary. Output extensions are tested
    at the likely 8/10/12-worker knees, not at every possible subset.
    """
    fixed = set()
    compact = machine.e
    spread = tuple(i for group in itertools.zip_longest(*machine.efficient_l2)
                   for i in group if i is not None)
    counts = sorted({0, len(compact), *range(2, len(compact), 2)})
    for smt, count, order in itertools.product((False, True), counts, (compact, spread)):
        p = machine.p_primary + (machine.p_smt if smt else ())
        # Canonical rank order makes equivalent full E selections deduplicate.
        fixed.add(p + tuple(sorted(order[:count])))
    full = machine.p_primary + machine.p_smt + machine.e + machine.low_power
    fixed.add(full)
    pairs = {(c, c) for c in fixed}
    for layer in fixed:
        # Partial E selections are two representatives per L2 domain or all E.
        ecount = len(set(layer) & set(machine.e))
        if ecount not in (len(machine.e) // 2, len(machine.e)) or layer == full:
            continue
        for target in (machine.p_primary + machine.e,
                       machine.p_primary + machine.p_smt + machine.e, full):
            if set(layer) <= set(target):
                output = (*layer, *(i for i in target if i not in layer))
                pairs.add((layer, output))
    return sorted(pairs)


def catalog(machine):
    return [Candidate(layer, output, backend, d, o, r)
            for (layer, output), backend, d, o, r in itertools.product(
                layouts(machine), ('openmp', 'pthread'), (0, 1), (0, 1), (0, 1))]


def features(candidate, machine):
    """Topology/kernel main effects, curvature, and specified interactions.

    Features depend only on the catalog/topology, never on held-out timings.
    Normalized counts keep the fixed ridge prior comparable across machines.
    """
    c = candidate
    primary, smt, e, low = map(set, (machine.p_primary, machine.p_smt, machine.e, machine.low_power))
    def counts(cpus):
        selected = set(cpus)
        occupancy = [len(selected & set(g)) for g in machine.efficient_l2]
        return [len(selected & smt) / max(1, len(smt)),
                len(selected & e) / max(1, len(e)),
                len(selected & low) / max(1, len(low)),
                sum(n > 0 for n in occupancy) / max(1, len(occupancy)),
                sum(n*n for n in occupancy) / max(1, sum(len(g)**2 for g in machine.efficient_l2))]
    ls, le, ll, lc, lshare = counts(c.layer)
    os, oe, ol, oc, oshare = counts(c.output)
    switch = int(set(c.layer) != set(c.output))
    b, d, o, r = int(c.backend == 'pthread'), c.dynamic, c.output_dynamic, c.row4
    return [1., ls, le, ll, lc, lshare, os, oe, ol, oc, oshare,
            le*le, oe*oe, switch, b, d, o, r,
            b*d, b*o, b*r, d*o, d*r, o*r,
            b*switch, d*le, o*oe, r*oe, b*le, b*ls,
            d*ls, o*os, o*switch, r*switch, c.qkv, c.ffn, c.qkv*c.ffn]


class SearchPolicy:
    """Ask/tell policies sharing exactly the same finite candidate space.

    model: feature-spread initial design, recursive ridge posterior, lower
    confidence bound, 1/5 uniform exploration. beam: best observed layout's
    one-factor neighbors plus 1/5 random restarts. random: uniform without
    replacement. All choices are online and reproducible from observed values.
    """
    def __init__(self, candidates, machine, method='model', seed=0):
        if method not in ('model', 'model_no_topology', 'beam', 'random') or not candidates:
            raise ValueError('Nonempty candidates and a supported method are required')
        self.candidates = candidates
        self.method = method
        self.rng = random.Random(seed)
        self.x = [features(c, machine) for c in candidates]
        if method == 'model_no_topology':
            self.x = [[x[0], *x[14:24], *x[34:]] for x in self.x]
        p = len(self.x[0])
        self.inverse = [[float(i == j) for j in range(p)] for i in range(p)]  # ridge=1
        self.mean = [0.] * len(candidates)  # prior log(relative latency)=0
        self.variance = [sum(v*v for v in x) for x in self.x]
        self.observed = {}
        self.remaining = set(range(len(candidates)))
        self.initial = []
        if method.startswith('model'):
            # Farthest-first is structural only; random first point breaks ties.
            available = list(range(len(candidates)))
            self.rng.shuffle(available)
            first = available.pop()
            self.initial.append(first)
            distances = {i: sum((a-b)**2 for a, b in zip(self.x[i], self.x[first])) for i in available}
            for _ in range(min(11, len(available))):
                chosen = max(available, key=lambda i: distances[i])
                self.initial.append(chosen)
                available.remove(chosen)
                for i in available:
                    distances[i] = min(distances[i], sum((a-b)**2 for a,b in zip(self.x[i],self.x[chosen])))

    def ask(self):
        if not self.remaining:
            raise StopIteration
        if self.method == 'random':
            return self.rng.choice(sorted(self.remaining))
        if self.method.startswith('model') and len(self.observed) < len(self.initial):
            return self.initial[len(self.observed)]
        if not self.observed or len(self.observed) % 5 == 4:
            return self.rng.choice(sorted(self.remaining))
        if self.method.startswith('model'):
            return min(sorted(self.remaining), key=lambda i: self.mean[i] - .05*math.sqrt(max(0., self.variance[i])))
        best = min(self.observed, key=self.observed.get)
        c = self.candidates[best]
        def distance(i):
            other = self.candidates[i]
            # Treat a deployable layer/output allocation as one atomic action;
            # swapping it can both remove and add CPUs, unlike the old add-only beam.
            return int((c.layer, c.output) != (other.layer, other.output)) + sum(
                getattr(c,k) != getattr(other,k) for k in ('backend','dynamic','output_dynamic','row4','qkv','ffn'))
        neighbors = [i for i in sorted(self.remaining) if distance(i) == 1]
        # No unseen neighbor: restart without looking at other timings.
        return self.rng.choice(neighbors or sorted(self.remaining))

    def tell(self, index, log_relative_latency):
        if index not in self.remaining or not math.isfinite(log_relative_latency):
            raise ValueError('Each candidate needs one finite observation')
        self.remaining.remove(index)
        self.observed[index] = log_relative_latency
        if not self.method.startswith('model'):
            return
        x = self.x[index]
        u = [sum(a*b for a,b in zip(row,x)) for row in self.inverse]
        denominator = 1. + sum(a*b for a,b in zip(x,u))
        residual = log_relative_latency - self.mean[index]
        for i, xi in enumerate(self.x):
            covariance = sum(a*b for a,b in zip(xi,u))
            self.mean[i] += covariance * residual / denominator
            self.variance[i] = max(0., self.variance[i] - covariance*covariance/denominator)
        for i in range(len(x)):
            for j in range(len(x)):
                self.inverse[i][j] -= u[i]*u[j] / denominator

    def recommend(self):
        if not self.observed:
            raise ValueError('No measured candidates')
        return min(self.observed, key=self.observed.get)
