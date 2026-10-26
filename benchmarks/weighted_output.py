"""Sequential, reproducible benchmarks for dev/omp-tmp (Python standard library).

Run from the repository root; see docs/weighted-output-benchmark.md.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    cpus: tuple[int, ...]
    layer: int
    output: int = 0
    p: int = 0
    weight: int = 2
    spin: int | None = None
    wait: str | None = "ACTIVE"
    variant: str = "weighted"
    schedule: str = "static"

    @property
    def id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def unique(cases):
    return list({c.id: c for c in cases}.values())


def coarse():
    layouts = [
        ((0, 2, 4, 5, 6, 7), 2, (2, 4, 6), (1, 2, 3, 4)),
        ((0, 2, 1, 3, 4, 5, 8, 9), 4, (4, 6, 8), (1, 2, 3)),
        ((0, 2, 4, 5, 6, 7, 8, 9, 10, 11), 2, (2, 4, 6, 8, 9, 10), (1, 2, 3, 4)),
        ((0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11), 4, (4, 6, 8, 9, 10, 12), (1, 2, 3)),
        ((0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 2, (2, 8, 10), (1, 2, 3)),
        ((0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 4, (4, 8, 10, 12, 14), (1, 2, 3)),
    ]
    cases = []
    for cpus, p, layers, weights in layouts:
        for layer in layers:
            cases.append(Case(cpus, layer))
            for weight in weights:
                cases.append(Case(cpus, layer, len(cpus), p, weight))
    # Pure P cores / SMT, without an enlarged placement list.
    for cpus in ((0,), (0, 2), (0, 2, 1, 3)):
        cases.append(Case(cpus, len(cpus)))
    return unique(cases)


def read_case(data):
    return Case(**(data | {"cpus": tuple(data["cpus"])}))


def selected(previous, split, count):
    return [read_case(r["config"]) for r in previous
            if bool(r["config"]["output"]) == split][:count]


def followup(previous, final):
    weighted = selected(previous, True, 3)
    baseline = selected(previous, False, 2)
    if final:
        best = weighted[0]
        cases = weighted + baseline + [replace(best, weight=2), replace(best, weight=1)]
        # Keep an actual layer/output count change even if weighting alone wins.
        count_split = next((read_case(r["config"]) for r in previous
                            if r["config"]["output"] and
                            r["config"]["layer"] != r["config"]["output"]), None)
        if count_split is not None:
            cases.append(count_split)
        cases += [replace(baseline[0], variant="upstream"),
                  replace(best, output=0, p=0),
                  replace(baseline[0], variant="main_fp32"),
                  replace(baseline[0], variant="main_fp32_huge")]
        # The two configurations from master check_top(), plus two guided placement
        # controls used in the prior experiment. All use an independently built master.
        for cpus, schedule, wait in [
            ((0, 1, 2, 3, 6, 7, 9, 10, 11), "static", None),
            ((0, 1, 2, 3, 6, 7, 9, 10, 11), "guided,16", "ACTIVE"),
            ((0, 2, 3, 5, 6, 7, 8, 10, 11), "guided,16", "ACTIVE"),
            ((1, 2, 3, 5, 6, 7, 8, 10, 11), "guided,16", "ACTIVE"),
        ]:
            cases.append(Case(cpus, 9, variant="main_int8", schedule=schedule, wait=wait))
        return unique(cases)
    cases = []
    for c in weighted + baseline:
        for spin in (None, 0, 100, 1000, 10000):
            cases.append(replace(c, spin=spin))
        cases.append(replace(c, wait="PASSIVE", spin=None))
    best = weighted[0]
    for layer in range(max(1, best.layer - 1), min(len(best.cpus), best.layer + 1) + 1):
        for output in range(max(best.p + 1, best.output - 1), min(len(best.cpus), best.output + 1) + 1):
            for weight in range(max(1, best.weight - 1), best.weight + 2):
                cases.append(replace(best, layer=layer, output=output, weight=weight))
    return unique(cases)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def telemetry():
    return dict(
        ac={str(p): read(p) for p in Path("/sys/class/power_supply").glob("*/online")},
        temperature_mC={read(p.parent / "type") or str(p.parent): read(p)
                        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")},
        frequency_kHz={p.parent.parent.name: read(p)
                       for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")},
        loadavg=read("/proc/loadavg"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("coarse", "tune", "final"), default="coarse")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--upstream-ref", default="350e04fe35433e6d2941dce5a1f53308f87058eb")
    parser.add_argument("--main-ref", default="origin/master")
    args = parser.parse_args()
    if args.repeats < 1 or (args.phase != "coarse" and args.previous is None):
        parser.error("positive repeats and --previous for tune/final are required")
    model, output = args.model.resolve(), args.output.resolve()
    if not model.is_file():
        parser.error(f"model not found: {model}")
    output.mkdir(parents=True, exist_ok=False)
    cases = coarse() if args.phase == "coarse" else followup(
        json.loads((args.previous / "ranking.json").read_text()), args.phase == "final")
    for c in cases:
        if not set(c.cpus) <= os.sched_getaffinity(0) or not 1 <= c.layer <= len(c.cpus):
            raise ValueError(c)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT)
    metadata = dict(arguments=vars(args), time_utc=datetime.now(timezone.utc).isoformat(),
                    commit=git("rev-parse", "HEAD").decode().strip(),
                    source_sha256=hashlib.sha256((ROOT / "run.c").read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256((ROOT / "tokenizer.bin").read_bytes()).hexdigest(),
                    cpu=subprocess.check_output(["lscpu"], text=True),
                    compiler=subprocess.check_output(["gcc", "--version"], text=True),
                    affinity=sorted(os.sched_getaffinity(0)), telemetry_before=telemetry(),
                    source_refs={v: git("rev-parse", ref).decode().strip()
                                 for v, ref in (("upstream", args.upstream_ref), ("main", args.main_ref))})
    (output / "cases.json").write_text(json.dumps([asdict(c) | {"id": c.id} for c in cases], indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))
    with tempfile.TemporaryDirectory(prefix="llama-omp-benchmark-") as work:
        work = Path(work)
        sources = {"weighted": ROOT / "run.c", "upstream": work / "upstream.c", "main": work / "main/run/run.c"}
        sources["upstream"].write_bytes(git("show", f"{args.upstream_ref}:run.c"))
        sources["main"].parent.mkdir(parents=True)
        sources["main"].write_bytes(git("show", f"{args.main_ref}:run/run.c"))
        (work / "main/src").mkdir()
        for name in ("fused.h", "classifier.h"):
            (work / "main/src" / name).write_bytes(git("show", f"{args.main_ref}:src/{name}"))
        binaries = {}
        for name, source in sources.items():
            binary = work / f"run-{name}"
            subprocess.run(["gcc", "-Ofast", "-march=native", "-fopenmp", str(source), "-lm", "-o", str(binary)], check=True)
            binaries[name] = binary
        metadata["binary_sha256"] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in binaries.items()}
        rows, reference = [], None

        def measure(c, repeat):
            nonlocal reference
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("OMP_", "GOMP_", "OUTPUT_", "PERSISTENT_", "LLAMA_OUT_"))}
            env.update(OMP_NUM_THREADS=str(c.layer), OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close",
                       OMP_PLACES=",".join(f"{{{cpu}}}" for cpu in c.cpus), OMP_SCHEDULE=c.schedule,
                       LLAMA_OUT_THREADS=str(c.output), LLAMA_OUT_P=str(c.p), LLAMA_OUT_WP=str(c.weight))
            if c.wait is not None:
                env["OMP_WAIT_POLICY"] = c.wait
            if c.spin is not None:
                env["GOMP_SPINCOUNT"] = str(c.spin)
            variant = "main" if c.variant.startswith("main") else c.variant
            cmd = ["taskset", "-c", ",".join(map(str, c.cpus)), str(binaries[variant]), str(model),
                   "-z", str(ROOT / "tokenizer.bin"), "-n", "256", "-t", "0", "-s", "42", "-i", "Once upon a time"]
            if c.variant in ("main_fp32_huge", "main_int8"):
                cmd += ["-w", "1"]
            if c.variant == "main_int8":
                cmd += ["-a", "1", "-g", "144"]
            before = telemetry()
            started = time.monotonic()
            result = subprocess.run(cmd, env=env, capture_output=True, check=True, timeout=120)
            elapsed = time.monotonic() - started
            after = telemetry()
            match = re.search(rb"achieved tok/s:\s*([\d.]+)", result.stderr)
            if not match or not math.isfinite(tps := float(match[1])) or tps <= 0:
                raise RuntimeError(result.stderr.decode())
            digest = hashlib.sha256(result.stdout).hexdigest()
            if reference is None:
                reference = digest
            (output / "stdout").mkdir(exist_ok=True)
            (output / "stdout" / f"{digest}.txt").write_bytes(result.stdout)
            row = dict(case=c.id, repeat=repeat, config=asdict(c), tokens_per_second=tps,
                       ms_per_token=1000 / tps, process_seconds=elapsed,
                       stdout_sha256=digest, matches_reference=digest == reference,
                       stderr=result.stderr.decode(), before=before, after=after,
                       finished_utc=datetime.now(timezone.utc).isoformat())
            with (output / ("warmup.jsonl" if repeat < 0 else "runs.jsonl")).open("a") as f:
                f.write(json.dumps(row) + "\n")
            if digest != reference:
                raise RuntimeError(f"Generation differs from FP32 upstream: {c}")
            return row

        # Both source variants are warmed up; no warmup contributes to ranking.
        measure(replace(cases[0], variant="upstream", output=0, p=0), -1)
        measure(cases[0], -1)
        print(f"{args.phase}: {len(cases)} cases x {args.repeats}; {output}", flush=True)
        rng = random.Random(42)
        for repeat in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for c in order:
                row = measure(c, repeat)
                rows.append(row)
                print(f"{repeat+1:02d}/{args.repeats} {c.id} {c.variant} {c.layer}->{c.output or c.layer} "
                      f"P={c.p} W={c.weight} spin={c.spin} {c.wait}: {row['tokens_per_second']:.2f}", flush=True)
            ranking = []
            for c in cases:
                samples = [r for r in rows if r["case"] == c.id]
                ms = [r["ms_per_token"] for r in samples]
                rates = [r["tokens_per_second"] for r in samples]
                ranking.append(dict(case=c.id, config=asdict(c), repeats=len(samples),
                                    tokens_per_second=1000 / statistics.mean(ms),
                                    arithmetic_mean_tokens_per_second=statistics.mean(rates),
                                    ms_per_token=statistics.mean(ms),
                                    stdev_ms=statistics.stdev(ms) if len(ms) > 1 else 0,
                                    min_tokens_per_second=min(rates), max_tokens_per_second=max(rates)))
            ranking.sort(key=lambda r: r["ms_per_token"])
            (output / "ranking.json").write_text(json.dumps(ranking, indent=2))
            print("Top: " + ", ".join(f"{r['case']}={r['tokens_per_second']:.2f}" for r in ranking[:5]), flush=True)
        metadata["telemetry_after"] = telemetry()
        (output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))


if __name__ == "__main__":
    main()
