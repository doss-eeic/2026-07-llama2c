"""Search CPU subsets per (Transformer layer, operator), then measure the policy.

Linux example:
  python -m cpuprofiler.autotune_detail --cpus 0,1,2,3 --output benchmark/detail-search
  python -m cpuprofiler.autotune_detail --analyze benchmark/detail-search
"""
import argparse
from dataclasses import asdict
from itertools import combinations
import json
import hashlib
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from .benchmarkconfig import BenchmarkConfig, ROOT
from .benchmark_detail import parse_profile, run

STAGES = ("q_proj", "k_proj", "v_proj", "attention", "attn_out",
          "ffn_up", "ffn_gate", "ffn_down", "output")


def cpu_tuple(text):
    cpus = tuple(sorted(int(cpu) for cpu in text.split(",")))
    if not cpus or cpus[0] < 0 or len(set(cpus)) != len(cpus):
        raise ValueError("CPU IDs must be distinct nonnegative integers")
    return cpus


def generate_candidates(cpus, max_workers=None):
    """Exhaustive subsets within the explicitly supplied search domain."""
    cpus = cpu_tuple(",".join(map(str, cpus)))
    maximum = len(cpus) if max_workers is None else max_workers
    if not 1 <= maximum <= len(cpus):
        raise ValueError("max_workers must be between 1 and CPU count")
    count = sum(math.comb(len(cpus), n) for n in range(1, maximum + 1))
    if count > 4095:
        raise ValueError("over 4095 subsets: restrict --max-workers or use --candidate")
    return [subset for n in range(1, maximum + 1) for subset in combinations(cpus, n)]


def load_trial(directory, skip_tokens):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    cpus = cpu_tuple(",".join(map(str, config["cpus"])))
    if config["threads"] != len(cpus):
        raise ValueError("stay policies require one worker per CPU")
    profile = parse_profile((directory / "profile.jsonl").read_text())
    forwards = [r for r in profile["timings"] if r["stage"] == "forward_total"]
    ids = {r["forward_id"] for r in forwards if r["forward_id"] >= skip_tokens}
    if not ids or len({r["forward_id"] for r in forwards}) != len(forwards):
        raise ValueError("no measured tokens or duplicate forwards")
    values = {}
    seen = set()
    for row in profile["timings"]:
        if row["forward_id"] not in ids or row["stage"] not in STAGES:
            continue
        key = (row["layer"], row["stage"])
        identity = (row["forward_id"], key)
        if identity in seen:
            raise ValueError("duplicate operator timing")
        seen.add(identity)
        duration = row["duration_ns"] / 1e6
        if duration < 0 or not math.isfinite(duration):
            raise ValueError("invalid operator duration")
        values.setdefault(key, []).append(duration)
    expected = {(layer, stage) for layer in range(profile["meta"]["n_layers"])
                for stage in STAGES if stage != "output"} | {(-1, "output")}
    if set(values) != expected or any(len(v) != len(ids) for v in values.values()):
        raise ValueError("incomplete operator timings")
    return {"cpus": cpus, "profile": profile,
            "operators": {key: statistics.mean(v) for key, v in values.items()},
            "forward_ms": statistics.mean(r["duration_ns"] / 1e6 for r in forwards
                                           if r["forward_id"] in ids)}


def select_policy(trials):
    """Use medians of independent trials, never sums containing layer_total."""
    if not trials:
        raise ValueError("no trials")
    keys = set(trials[0]["operators"])
    workload = [(r["pos"], r["input_token"]) for r in trials[0]["profile"]["timings"]
                if r["stage"] == "forward_total"]
    if not workload:
        raise ValueError("empty workload")
    grouped = {}
    for trial in trials:
        if set(trial["operators"]) != keys:
            raise ValueError("operator/layer coverage differs across trials")
        actual = [(r["pos"], r["input_token"]) for r in trial["profile"]["timings"]
                  if r["stage"] == "forward_total"]
        if actual != workload or trial["profile"]["meta"] != trials[0]["profile"]["meta"]:
            raise ValueError("workload/model metadata differs across trials")
        grouped.setdefault(trial["cpus"], []).append(trial)
    baseline = min(grouped, key=lambda cpus: statistics.median(t["forward_ms"] for t in grouped[cpus]))
    choices = []
    for layer, stage in sorted(keys):
        medians = {cpus: statistics.median(t["operators"][(layer, stage)] for t in group)
                   for cpus, group in grouped.items()}
        best = min(medians, key=lambda cpus: (medians[cpus], len(cpus), cpus))
        choices.append({"layer": layer, "stage": stage, "cpus": list(best),
                        "median_ms": medians[best], "baseline_ms": medians[baseline]})
    return {"schema": 1, "baseline_cpus": list(baseline), "choices": choices,
            "baseline_forward_ms": statistics.median(t["forward_ms"] for t in grouped[baseline]),
            "estimated_operator_ms": sum(c["median_ms"] for c in choices),
            "note": "Independent operator minima are estimates; validation measures synchronization too."}


def write_policy(path, default_cpus, choices=()):
    lines = ["-2 default " + ",".join(map(str, default_cpus))]
    lines += [f'{c["layer"]} {c["stage"]} ' + ",".join(map(str, c["cpus"])) for c in choices]
    Path(path).write_text("\n".join(lines) + "\n")


def analyze(output, skip_tokens):
    trials = [load_trial(path.parent, skip_tokens) for path in sorted(output.glob("trial-*/config.json"))]
    result = select_policy(trials)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    write_policy(output / "policy.txt", result["baseline_cpus"], result["choices"])
    return result, trials


def save_config(config):
    # benchmark_detail creates the directory; write configuration after successful execution.
    values = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    (config.output / "config.json").write_text(json.dumps(values, indent=2) + "\n")


def blocked_jobs(candidates, repeats, seed):
    randomizer = random.Random(seed)
    jobs = []
    for repeat in range(repeats):
        block = list(candidates)
        randomizer.shuffle(block)
        jobs.extend((repeat, candidate) for candidate in block)
    return jobs


def measure_policies(output, args, policies, replay, expected_count):
    expected_inputs = list(enumerate(map(int, replay.read_text().split())))
    jobs = blocked_jobs(list(policies), args.repeats, args.order_seed)
    results, reference = [], None
    for repeat, name in jobs:
        directory = output / f"validation-{name}-{repeat}"
        directory.mkdir()
        env = {**os.environ, "LLAMA_PROFILE_PATH": str(directory / "profile.jsonl"),
               "LLAMA_STAY_TIMING_ONLY": "1"}
        command = [str(args.stay_binary.resolve()), str(args.model.resolve()),
                   "-z", str(args.tokenizer.resolve()), "-c", str(policies[name].resolve()),
                   "-r", str(replay.resolve()), "-t", "0", "-s", str(args.seed)]
        with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            subprocess.run(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr,
                           check=True, timeout=args.timeout)
        profile = parse_profile((directory / "profile.jsonl").read_text())
        tokens = [(r["pos"], r["input_token"], r["next_token"]) for r in profile["tokens"]]
        if len(tokens) != expected_count or [(pos, token) for pos, token, _ in tokens] != expected_inputs:
            raise ValueError("replay token count differs")
        if reference is None:
            reference = tokens
        if tokens != reference:
            raise ValueError("greedy token IDs differ between policies")
        durations = [r["duration_ns"] / 1e6 for r in profile["timings"]
                     if r["stage"] == "forward_total" and r["forward_id"] >= args.skip_tokens]
        if len(durations) != expected_count - args.skip_tokens:
            raise ValueError("measured replay forward count differs")
        mean = statistics.mean(durations)
        results.append({"policy": name, "repeat": repeat, "forward_ms_per_token": mean,
                        "forward_tokens_per_second": 1000 / mean})
        print(f"validate {name} repeat={repeat}: {mean:.4f} ms/token", flush=True)
    medians = {name: statistics.median(r["forward_ms_per_token"] for r in results if r["policy"] == name)
               for name in policies}
    return results, medians


def validate(output, args, selection, trials):
    """Compare mixed and ALL uniform candidates using identical replay and pool code."""
    replay = output / "replay.txt"
    records = [r for r in trials[0]["profile"]["timings"] if r["stage"] == "forward_total"]
    replay.write_text("\n".join(str(r["input_token"]) for r in records) + "\n")
    candidates = sorted({t["cpus"] for t in trials})
    policies = {"mixed": output / "policy.txt"}
    for index, cpus in enumerate(candidates):
        path = output / f"uniform-{index}.txt"
        write_policy(path, cpus)
        policies[f"uniform-{index}"] = path
    results, medians = measure_policies(output, args, policies, replay, len(records))
    best = min(medians, key=medians.get)
    (output / "recommended-policy.txt").write_text(policies[best].read_text())
    report = {"trials": results, "median_forward_ms": medians, "best_policy": best,
              "greedy_tokens_match": True, "best_policy_path": str(policies[best]),
              "mixed_speedup_vs_best_uniform": min(v for k, v in medians.items() if k != "mixed") / medians["mixed"]}
    if args.refine_rounds:
        report["refinement"] = refine(output, args, selection, candidates, replay,
                                       len(records), best, medians[best])
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def refine(output, args, selection, candidates, replay, expected_count, initial, initial_ms):
    """Coordinate descent on real forward latency, including worker wake/sync costs.

    This is a local search, not exhaustive enumeration of all operator assignments.
    """
    current = [dict(choice) for choice in selection["choices"]]
    if initial.startswith("uniform-"):
        cpus = candidates[int(initial.split("-")[1])]
        current = [{**choice, "cpus": list(cpus)} for choice in current]
    history = []
    for round_id in range(args.refine_rounds):
        changed = False
        for key, choice in enumerate(current):
            directory = output / f"refine-{round_id}-{key}"
            directory.mkdir()
            policies, assignments = {}, {}
            for index, cpus in enumerate(candidates):
                name = f"candidate-{index}"
                variant = [dict(c) for c in current]
                variant[key] = {**choice, "cpus": list(cpus)}
                path = directory / f"{name}.txt"
                write_policy(path, selection["baseline_cpus"], variant)
                policies[name], assignments[name] = path, variant
            results, medians = measure_policies(directory, args, policies, replay, expected_count)
            incumbent = next(name for name, variant in assignments.items()
                             if variant[key]["cpus"] == current[key]["cpus"])
            winner = min(medians, key=medians.get)
            # Ties preserve the incumbent; all candidates include the current assignment.
            if medians[winner] < medians[incumbent]:
                current = assignments[winner]
                changed = True
            else:
                winner = incumbent
            history.append({"round": round_id, "layer": choice["layer"], "stage": choice["stage"],
                            "selected_cpus": current[key]["cpus"], "median_forward_ms": medians,
                            "trials": results})
            (output / "refinement.json").write_text(json.dumps(history, indent=2) + "\n")
        if not changed:
            break
    # Re-test the final assignment against the original best policy on fresh blocks.
    final_path = output / "refined-policy.txt"
    write_policy(final_path, selection["baseline_cpus"], current)
    comparison = output / "refinement-final"
    comparison.mkdir()
    original_path = output / "recommended-policy.txt"
    results, medians = measure_policies(comparison, args,
        {"original": original_path, "refined": final_path}, replay, expected_count)
    winner = min(medians, key=medians.get)
    if winner == "refined":
        original_path.write_text(final_path.read_text())
    return {"steps": len(history), "initial_median_ms": initial_ms,
            "final_comparison": medians, "trials": results, "recommended": winner}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=cpu_tuple, help="search all nonempty subsets of these CPUs")
    parser.add_argument("--candidate", action="append", type=cpu_tuple, help="explicit CPU set; repeatable")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze", type=Path, help="analyze existing trial directories without execution")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/omp-run-detail")
    parser.add_argument("--stay-binary", type=Path, default=ROOT / "build/stay-run-detail")
    parser.add_argument("--model", type=Path, default=ROOT / "build/stories15M.bin")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.bin")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--skip-tokens", type=int, default=1)
    parser.add_argument("--refine-rounds", type=int, default=0,
                        help="optional coordinate-descent passes using actual forward times (expensive)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-validate", action="store_true", help="only estimate policy; no speedup conclusion")
    args = parser.parse_args(argv)
    if args.refine_rounds < 0:
        parser.error("refine-rounds must be nonnegative")
    if args.no_validate and args.refine_rounds:
        parser.error("refinement requires validation")
    if args.repeats < 1 or args.skip_tokens < 0 or args.steps <= args.skip_tokens:
        parser.error("require repeats >= 1 and steps > skip_tokens >= 0")
    if args.analyze:
        selection, _ = analyze(args.analyze.resolve(), args.skip_tokens)
        print(json.dumps(selection, indent=2))
        return
    if args.output is None or bool(args.cpus) == bool(args.candidate):
        parser.error("provide --output and exactly one of --cpus / --candidate")
    candidates = generate_candidates(args.cpus, args.max_workers) if args.cpus else sorted(set(args.candidate))
    if not hasattr(os, "sched_getaffinity"):
        parser.error("benchmark execution requires Linux CPU affinity")
    if not set(cpu for cpus in candidates for cpu in cpus) <= os.sched_getaffinity(0):
        parser.error("candidate CPUs are outside process affinity")
    for path in (args.binary, args.model, args.tokenizer, *(() if args.no_validate else (args.stay_binary,))):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["allowed_cpus"] = sorted(os.sched_getaffinity(0))
    manifest["candidates"] = candidates
    manifest["sha256"] = {}
    for path in (args.binary, args.model, args.tokenizer,
                 *(() if args.no_validate else (args.stay_binary,))):
        with path.open("rb") as file:
            manifest["sha256"][str(path.resolve())] = hashlib.file_digest(file, "sha256").hexdigest()
    (output / "search.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = blocked_jobs(candidates, args.repeats, args.order_seed)
    for index, (repeat, cpus) in enumerate(jobs):
        config = BenchmarkConfig(cpus=cpus, threads=len(cpus), output=output / f"trial-{index:05d}",
                                 binary=args.binary, model=args.model, tokenizer=args.tokenizer,
                                 steps=args.steps, seed=args.seed, prompt=args.prompt,
                                 skip_tokens=args.skip_tokens, timeout=args.timeout)
        summary, _ = run(config)
        save_config(config)
        print(f'{index + 1}/{len(jobs)} CPUs={cpus} repeat={repeat}: {summary["forward_ms_per_token"]:.4f} ms/token', flush=True)
    selection, trials = analyze(output, args.skip_tokens)
    if not args.no_validate:
        report = validate(output, args, selection, trials)
        print(f'Saved best measured policy: {output / "recommended-policy.txt"}')
    else:
        print(f"Estimated policy: {output / 'policy.txt'} (not validated)")


if __name__ == "__main__":
    main()
