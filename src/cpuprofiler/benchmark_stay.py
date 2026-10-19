"""外部ポリシーでstayを1回実行する。mainは固定構成とoutput切替の比較用。"""

import csv
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
    configs = {
        "P2": (0, 2),
        "P4_HT": (0, 1, 2, 3),
        "P4_45": (0, 1, 2, 3, 4, 5),
        "P4_67": (0, 1, 2, 3, 6, 7),
        "P4_89": (0, 1, 2, 3, 8, 9),
        "P4_1011": (0, 1, 2, 3, 10, 11),
        "P4_4567": (0, 1, 2, 3, 4, 5, 6, 7),
        "P4_4589": (0, 1, 2, 3, 4, 5, 8, 9),
        "P4_891011": (0, 1, 2, 3, 8, 9, 10, 11),
    }
    policies = {name: (cpus, cpus) for name, cpus in configs.items()}
    policies["P4_HT_output_P4X4"] = (configs["P4_HT"], configs["P4_4589"])
    policies["P4_HT_output_P4_89"] = (configs["P4_HT"], configs["P4_89"])

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


if __name__ == "__main__":
    main()
