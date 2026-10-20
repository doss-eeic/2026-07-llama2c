"""外部ポリシーでstayを1回実行する。mainは固定構成とoutput切替の比較用。"""

import csv
from dataclasses import replace
from datetime import datetime
import os
import subprocess

from cpuprofiler.benchmark_detail import parse_profile, summarize
from cpuprofiler.benchmarkconfig import ROOT, StayBenchmarkConfig


def run(config: StayBenchmarkConfig):
    """生ログと使用ポリシーを保存し、(全体の集計, stage別の集計)を返す。"""
    policy = config.policy.read_bytes()
    output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy_path = output / "policy.txt"
    policy_path.write_bytes(policy)
    command = [str(config.binary.resolve()), str(config.model.resolve()),
               "-c", str(policy_path), "-z", str(config.tokenizer.resolve()),
               "-n", str(config.steps), "-t", str(config.temperature),
               "-s", str(config.seed), "-i", config.prompt]
    # stay自身が全ルールのCPUにworkerを固定する。tasksetでdefaultに絞らない。
    env = {**os.environ, "LLAMA_PROFILE_PATH": str(output / "profile.jsonl")}
    env.pop("LLAMA_STAY_TIMING_ONLY", None)
    if config.timing_only:
        env["LLAMA_STAY_TIMING_ONLY"] = "1"
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                       check=True, timeout=config.timeout)
    profile = parse_profile((output / "profile.jsonl").read_text())
    return summarize(profile["timings"], config.skip_tokens, config.stages)


def main():
    # CPU番号は既存のLinux測定機用。コア種別は番号だけでは断定しない。
    # configs = {
    #     "P1": (0,),
    #     "P2": (0, 2),
    #     "P4_HT": (0, 1, 2, 3),
    #     "E4": (4, 5, 6, 7),
    #     "E4_alt": (8, 9, 10, 11),
    # }
    # policies = {name: (cpus, cpus) for name, cpus in configs.items()}

    configs = {
    "P2": (0, 2),
    "P4_HT": (0, 1, 2, 3),
    "P4_89": (0, 1, 2, 3, 8, 9),
    "P4_4589": (0, 1, 2, 3, 4, 5, 8, 9),
    }

    policies = {
        name: (cpus, cpus)
        for name, cpus in configs.items()
    }

    policies["P4_output_89"] = (
        configs["P4_HT"],
        configs["P4_89"],
    )

    policies["P4_output_4589"] = (
        configs["P4_HT"],
        configs["P4_4589"],
    )
    output_root = ROOT / "results" / datetime.now().strftime("stay_%Y%m%d_%H%M%S_%f")
    policy_root = output_root / "policies"
    policy_root.mkdir(parents=True)
    print(f"Results: {output_root}", flush=True)

    results = []
    for name, (cpus, output_cpus) in policies.items():
        default_list = ",".join(map(str, cpus))
        output_list = ",".join(map(str, output_cpus))
        policy = policy_root / f"{name}.txt"
        policy.write_text(f"-2 default {default_list}\n-1 output {output_list}\n")
        summary, stages = run(StayBenchmarkConfig(policy=policy, output=output_root / name))
        results.append({"config": name, "cpus": default_list, "output_cpus": output_list,
                        **summary, **{s["stage"]: s["ms_per_token"] for s in stages}})
        print(f"{name}: {summary['forward_ms_per_token']:.3f} ms/token", flush=True)

        # 途中で失敗しても完了済みの結果を残す。
        keys = list(dict.fromkeys(k for row in results for k in row))
        with (output_root / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

def main2():
    """固定CPU構成と、演算別CPU切り替えの性能を比較する。"""

    P4 = (0, 1, 2, 3)
    P8 = (0, 1, 2, 3, 4, 5, 8, 9)

    # (default, ffn, output)
    configs = {
        "Baseline":     (P4, P4, P4),
        "All8":         (P8, P8, P8),
        "Output8":      (P4, P4, P8),
        "FFN_Output8":  (P4, P8, P8),
    }

    output_root = (
        ROOT / "results" /
        datetime.now().strftime("stay_%Y%m%d_%H%M%S_%f")
    )
    policy_root = output_root / "policies"
    policy_root.mkdir(parents=True)

    print(f"Results: {output_root}", flush=True)

    results = []

    for name, (default, ffn, output_cpus) in configs.items():

        def cpu_list(cpus):
            return ",".join(map(str, cpus))

        # ポリシー生成
        rules = [f"-2 default {cpu_list(default)}"]

        if ffn != default:
            for stage in ("ffn_up", "ffn_gate", "ffn_down"):
                rules.append(f"-2 {stage} {cpu_list(ffn)}")

        if output_cpus != default:
            rules.append(f"-1 output {cpu_list(output_cpus)}")

        policy = policy_root / f"{name}.txt"
        policy.write_text("\n".join(rules) + "\n")

        # ベンチマーク実行
        summary, stages = run(
            StayBenchmarkConfig(
                policy=policy,
                output=output_root / name,
            )
        )

        row = {
            "config": name,
            "cpus": cpu_list(default),
            "ffn_cpus": cpu_list(ffn),
            "output_cpus": cpu_list(output_cpus),
            **summary,
            **{
                s["stage"]: s["ms_per_token"]
                for s in stages
            },
        }

        results.append(row)

        print(
            f"{name}: "
            f"{summary['forward_ms_per_token']:.3f} ms/token",
            flush=True,
        )

        # 各構成の実行後にCSV保存
        keys = list(dict.fromkeys(
            k for row in results for k in row
        ))

        with (output_root / "summary.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


def run2(config: StayBenchmarkConfig):
    """outputだけ切り替えるstay2を1回実行する。"""
    return run(replace(config, binary=ROOT / "build/stay2-run-detail"))


def main_stay2():
    """stay2でP4_HT固定、P4X4固定、outputのみP4X4を比較する。"""
    p4_ht = (0, 1, 2, 3)
    p4x4 = (0, 1, 2, 3, 4, 5, 8, 9)
    policies = {
        "P4_HT": (p4_ht, p4_ht),
        "P4X4": (p4x4, p4x4),
        "P4_HT_output_P4X4": (p4_ht, p4x4),
    }
    output_root = ROOT / "results" / datetime.now().strftime("stay2_output_%Y%m%d_%H%M%S_%f")
    policy_root = output_root / "policies"
    policy_root.mkdir(parents=True)
    print(f"Results: {output_root}", flush=True)
    results = []
    for name, (cpus, output_cpus) in policies.items():
        default_list = ",".join(map(str, cpus))
        output_list = ",".join(map(str, output_cpus))
        policy = policy_root / f"{name}.txt"
        policy.write_text(f"-2 default {default_list}\n-1 output {output_list}\n")
        summary, stages = run2(StayBenchmarkConfig(policy=policy, output=output_root / name))
        results.append({"config": name, "cpus": default_list, "output_cpus": output_list,
                        **summary, **{s["stage"]: s["ms_per_token"] for s in stages}})
        print(f"{name}: {summary['forward_ms_per_token']:.3f} ms/token", flush=True)
        keys = list(dict.fromkeys(k for row in results for k in row))
        with (output_root / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main_stay2()
