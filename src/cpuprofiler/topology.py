import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


def read(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def parse_cpu_list(value):
    """'0-3,8,10-11' -> [0,1,2,3,8,10,11]"""
    cpus = []

    for part in value.split(","):
        if not part:
            continue

        if "-" in part:
            start, end = map(int, part.split("-"))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))

    return sorted(set(cpus))


class CPU:
    def __init__(self, cpu_id):
        self.cpu_id = cpu_id
        self.path = SYS_CPU / f"cpu{cpu_id}"
        self.topology_path = self.path / "topology"
        self.cache_path = self.path / "cache"

    @property
    def is_online(self):
        # cpu0/onlineから読むと/onlineがないケースに対応できないので，こちらから読む
        return self.cpu_id in parse_cpu_list(
            read(SYS_CPU / "online")
        )

    @property
    def is_available(self):
        return (
            self.is_online
            and self.cpu_id in os.sched_getaffinity(0)
        )

    @property
    def core_id(self):
        value = read(self.topology_path / "core_id")
        return int(value) if value else None

    @property
    def package_id(self):
        value = read(self.topology_path / "physical_package_id")
        return int(value) if value else None

    # core_cpus_listとthread_siblings_listはともに，同じ物理コアに属する論理CPUを返す。
    # 例) cat /sys/devices/system/cpu/cpu0/topology/core_cpus_listの結果は0-1
    @property
    def core_cpus_list(self):
        value = read(self.topology_path / "core_cpus_list")

        if not value:
            value = read(self.topology_path / "thread_siblings_list")

        return parse_cpu_list(value)

    @property
    def l2_cache(self):
        for index in self.cache_path.glob("index*"):
            level = read(index / "level")
            cache_type = read(index / "type")

            if level == "2" and cache_type in ("Unified", "Data"):
                return {
                    "size": read(index / "size"),
                    "shared_cpus": parse_cpu_list(
                        read(index / "shared_cpu_list")
                    )
                }

        return None
    
    def ln_cache(self,n):
        for index in self.cache_path.glob("index*"):
            level = read(index / "level")
            cache_type = read(index / "type")

            if level == str(n) and cache_type in ("Unified", "Data"):
                return {
                    "size": read(index / "size"),
                    "shared_cpus": parse_cpu_list(
                        read(index / "shared_cpu_list")
                    )
                }

        return None

    def __repr__(self):
        return (
            f"CPU(id={self.cpu_id}, "
            f"core={self.core_id}, "
            f"siblings={self.core_cpus_list}, "
            f"l2={self.l2_cache})"
        )
class CPUTopology:
    def __init__(self):
        # Linuxが認識しているオンラインCPU
        online = set(
            parse_cpu_list(read(SYS_CPU / "online"))
        )

        # 現在のプロセスが利用できるCPU
        allowed = set(os.sched_getaffinity(0))

        # 実際に使用可能な論理CPU
        self.available_cpu_ids = sorted(online & allowed)

        # CPUオブジェクトを生成
        self.cpus = {
            cpu_id: CPU(cpu_id)
            for cpu_id in self.available_cpu_ids
        }

        # 物理コア単位に分類
        self.physical_cores = self._group_physical_cores()

        # L2キャッシュ共有単位に分類
        self.l2_groups = self._group_l2_cache()

    def _group_physical_cores(self):
        groups = {}

        for cpu in self.cpus.values():
            siblings = cpu.core_cpus_list

            if siblings:
                # SMTを含む同一物理コアのCPU集合
                key = ("siblings", tuple(siblings))
                # ("siblings", (0, 1))というものをキーにする。
            elif cpu.package_id is not None and cpu.core_id is not None:
                # siblingsが取得できない場合
                key = ("core", cpu.package_id, cpu.core_id)

            else:
                # トポロジー不明なら別コアとして扱う
                key = ("unknown", cpu.cpu_id)
            
            # valueはリストなので，同じKeyを持つところにCPUを追加する。
            groups.setdefault(key, []).append(cpu.cpu_id)

        return [
            sorted(group)
            for group in groups.values()
        ]

    def _group_l2_cache(self):
        groups = {}

        for cpu in self.cpus.values():
            cache = cpu.l2_cache

            if cache is None:
                continue

            # CPU(1).l2_cacheは{'size': '256K', 'shared_cpus': [0, 1]}のような辞書を返す。同じL2キャッシュを共有するものがshared_cpusに入る。
            shared = cache["shared_cpus"]

            if not shared:
                continue

            # 同じL2キャッシュを共有するCPUをまとめる。keyを(0,1)と共有するタプルとする。
            key = tuple(shared)

            # sharedの中でavailableのもののみを残して，valueとする。
            groups[key] = sorted(
                set(shared) & set(self.available_cpu_ids)
            )

        return list(groups.values())

    @cached_property
    def domains(self):
        """L2共有 ∨ 同一物理コアの連結成分。"""
        return build_domains(self)

    @property
    def representative_cpus(self):
        """
        各物理コアから1つの論理CPUを選ぶ
        """
        return [
            self.cpus[min(group)]
            for group in self.physical_cores
        ]

    @property
    def n_logical_cpus(self):
        return len(self.cpus)

    @property
    def n_physical_cores(self):
        return len(self.physical_cores)

    def print_summary(self):
        print("=== CPU Topology ===")

        print(f"Logical CPUs: {self.n_logical_cpus}")
        print(f"Physical Cores: {self.n_physical_cores}")

        print("\nPhysical Cores:")
        for i, group in enumerate(self.physical_cores):
            print(f"  Core {i:2d}: CPUs {group}")

        print("\nL2 Cache Groups:")
        for i, group in enumerate(self.l2_groups):
            cpu = self.cpus[group[0]]
            size = cpu.l2_cache["size"]

            print(f"  L2 {i}: CPUs {group}, Size={size}")

        print("\nRepresentative CPUs:")
        print([cpu.cpu_id for cpu in self.representative_cpus])

# ここからTopologyを使った探索のための新しい構造, UnionFindで実装。

@dataclass(frozen=True)
class Domain:
    id: int
    cores: tuple[tuple[int, ...], ...]
    cpus: tuple[int, ...]
    l2_ids: tuple[int, ...]

    @property
    def shape(self):
        """物理コアごとの論理CPU数。同じshapeのDomainだけが同型の候補になる。"""
        return tuple(sorted(len(core) for core in self.cores))


def build_domains(topology):
    """topologyはcpus, l2_groups, physical_coresを持てばよい(CPUTopology.domainsからも使う)。"""
    available = set(topology.cpus)
    parent = {cpu: cpu for cpu in available}

    def find(cpu):
        while parent[cpu] != cpu:
            parent[cpu] = parent[parent[cpu]]
            cpu = parent[cpu]
        return cpu

    def union_group(group):
        cpus = sorted(set(group) & available)
        if not cpus:
            return
        root = find(cpus[0])
        for cpu in cpus[1:]:
            parent[find(cpu)] = root

    # L2共有と物理コアの両方でCPUを結合
    for group in topology.l2_groups:
        union_group(group)
    for core in topology.physical_cores:
        union_group(core)

    # 連結成分を抽出
    components = defaultdict(set)
    for cpu in available:
        components[find(cpu)].add(cpu)

    domains = []
    for i, cpus in enumerate(sorted(components.values(), key=min)):
        cores = tuple(
            tuple(sorted(set(core) & cpus))
            for core in topology.physical_cores
            if set(core) & cpus
        )
        l2_ids = tuple(
            j for j, group in enumerate(topology.l2_groups)
            if set(group) & cpus
        )

        domains.append(Domain(
            id=i,
            cores=cores,
            cpus=tuple(sorted(cpus)),
            l2_ids=l2_ids,
        ))

    return tuple(domains)


# ここから実測によるDomainの同型判定の結果。Domainは物理構造のまま不変にし、測定情報は外側に持つ。

@dataclass(frozen=True)
class DomainProfile:
    domain_id: int
    single_mean: float  # 各CPUを1スレッドで測った平均の平均 [ms/token]
    single_std: float  # 同一CPUでの実行間ばらつき(プールした標準偏差)。CPU間の差はper_cpu_meansで見る
    per_cpu_means: tuple[float, ...]  # domain.cpusの順
    full_mean: float | None = None  # Domainの全CPUで同時実行 [ms/token]。比較相手がいないDomainは測らない
    full_std: float | None = None


@dataclass(frozen=True)
class DomainClass:
    id: int
    domains: tuple[Domain, ...]


@dataclass
class CalibratedTopology:
    topology: CPUTopology
    domains: tuple[Domain, ...]
    profiles: dict[int, DomainProfile]
    classes: tuple[DomainClass, ...]

    def class_of(self, domain):
        return next(c for c in self.classes if domain in c.domains)

    def print_summary(self):
        print("=== Calibrated Domains ===")
        for c in self.classes:
            print(f"Class {c.id}:")
            for domain in c.domains:
                p = self.profiles[domain.id]
                full = "-" if p.full_mean is None else f"{p.full_mean:.3f}±{p.full_std:.3f}"
                print(f"  Domain {domain.id}: cores={domain.cores} "
                      f"single={p.single_mean:.3f}±{p.single_std:.3f} full={full} ms/token")

if __name__ == "__main__":
    print("===read関数で特定ファイルを読める===")
    print("\n/sys/devices/system/cpu/online：onlineのCPUリスト")
    print(read(SYS_CPU / "online"))
    print("\n/sys/devices/system/cpu/cpu0/topology/core_id：CPU0の物理コアID")
    print(read(SYS_CPU / "cpu0" / "topology" / "core_id"))
    print("\n/sys/devices/system/cpu/cpu0/topology/core_cpus_list：CPU0の物理コアを共有する論理CPUリスト")
    print(read(SYS_CPU / "cpu0" / "topology" / "core_cpus_list"))
    print("\n実行環境で利用可能なCPUのリスト")
    print(os.sched_getaffinity(0))
    print()

    print("===CPUクラスの利用例===")
    print("\nCPU(0)の情報")
    print(CPU(0))
    print("\nCPU(0)のL3キャッシュ情報：/sys/devices/system/cpu/cpu0/cache/index3の内容をまとめている。")
    print(CPU(0).ln_cache(3))
    print("\nCPU(1)の物理コアを共有する論理CPUリスト")
    print(CPU(1).core_cpus_list)
    print()
    print("===CPUTopologyクラスの利用例===")
    print("\nl2_groups: L2キャッシュを共有するCPUのグループ")
    print(CPUTopology().l2_groups)
    print("\nphysical_cores: 各物理コアを使っている論理CPUのグループ")
    print(CPUTopology().physical_cores)

    print("\n===CPUTopologyのまとめ表示===")
    print(CPUTopology().print_summary())
    
