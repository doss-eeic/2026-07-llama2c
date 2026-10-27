"""Topology-aware CPU autotuning with kasikoi2."""
from .benchmarkconfig import BenchmarkConfig
from .benchmark import BenchmarkResult, BenchmarkRunner
from .topology import CPU, CPUTopology
from .autotuner import AutoTuner
from .export import export_tuned


def main(argv=None):
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("topology", help="show available Linux CPUs")
    tune = commands.add_parser("kasikoi2", help="calibrate CPUs and tune inference")
    tune.add_argument("--binary", type=Path, default=BenchmarkConfig.binary)
    tune.add_argument("--model", type=Path, default=BenchmarkConfig.model)
    tune.add_argument("--tokenizer", type=Path, default=BenchmarkConfig.tokenizer)
    tune.add_argument("--steps", type=int, default=BenchmarkConfig.steps)
    tune.add_argument("--group-size", type=int, default=BenchmarkConfig.group_size)
    tune.add_argument("--top-k", type=int, default=3)
    tune.add_argument("--repeats", type=int, default=3)
    tune.add_argument("--calibration-repeats", type=int, default=3)
    tune.add_argument("--tolerance", type=float, default=0.05)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--timeout", type=float, default=BenchmarkConfig.timeout)
    tune.add_argument("--output", type=Path, default=BenchmarkConfig.binary.with_name("run-tuned.sh"),
                      help="shell script output with the best configuration")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "kasikoi2")
    args = parser.parse_args(arguments)
    if sys.platform != "linux":
        parser.error("CPU topology and autotuning require Linux (sysfs and taskset)")
    topology = CPUTopology()
    if args.command == "topology":
        topology.print_summary()
        return
    config = BenchmarkConfig(
        cpus=tuple(sorted(topology.cpus)), threads=topology.n_logical_cpus,
        binary=args.binary.resolve(), model=args.model.resolve(), tokenizer=args.tokenizer.resolve(),
        steps=args.steps, seed=args.seed, timeout=args.timeout,
        optimize_weights=True, two_stage_argmax=True, group_size=args.group_size,
    )
    tuner = AutoTuner(topology, BenchmarkRunner())
    result = tuner.kasikoi2_search(
        config, top_k=args.top_k, repeats=args.repeats, seed=args.seed,
        calibration_repeats=args.calibration_repeats, tolerance=args.tolerance,
    )
    tuner.last_search["calibrated"].print_summary()
    print(f"Best configuration: {result.config}")
    print(f"Best performance: {result.tokens_per_second:.2f} tokens/sec")
    print(f"Search: {tuner.last_search['seconds']:.2f} sec, "
          f"{tuner.last_search['measurements']} measurements")
    output = export_tuned(result, args.output)
    print(f"Saved tuned shell script: {output}")
