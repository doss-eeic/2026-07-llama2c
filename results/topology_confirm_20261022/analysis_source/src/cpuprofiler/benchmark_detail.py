"""Profile parser restored from dev/wataru-2 (66870ea)."""

import json
import statistics

def parse_profile(text):
    """Read our JSONL output; keep only completion and count checks."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or rows[0]["event"] != "meta" or rows[-1]["event"] != "end":
        raise ValueError("incomplete profile log")

    timings = [r for r in rows if r["event"] == "timing"]
    end = rows[-1]

    forwards = sum(r["stage"] == "forward_total" for r in timings)
    if len(timings) != end["timing_records"] or forwards != end["forward_calls"] or not forwards:
        raise ValueError("profile record count mismatch")
    tokens = [r for r in rows if r["event"] == "token"]
    if len(tokens) != end["token_records"]:
        raise ValueError("profile token count mismatch")
    return {"meta": rows[0], "timings": timings,
            "tokens": tokens, "end": end}


def summarize(records, skip_tokens, selected_stages=None):
    forwards = [r for r in records if r["stage"] == "forward_total"]
    measured = [r for r in forwards if r["forward_id"] >= skip_tokens]
    if not measured:
        raise ValueError("no measured tokens remain after skip_tokens")
    totals = {}
    for record in records:
        if record["forward_id"] >= skip_tokens and (selected_stages is None or record["stage"] in selected_stages):
            totals.setdefault(record["stage"], []).append(record["duration_ns"] / 1e6)
    stages = [
        {"stage": stage, "calls": len(values),
         "mean_call_ms": statistics.mean(values),
         "ms_per_token": sum(values) / len(measured)}
        for stage, values in sorted(totals.items())
    ]
    durations = [r["duration_ns"] / 1e6 for r in measured]
    return {"forward_calls": len(forwards), "measured_tokens": len(measured),
            "forward_ms_per_token": statistics.mean(durations)}, stages


def main():
    if __package__:
        from .benchmark_distribute import main as benchmark
    else:
        from benchmark_distribute import main as benchmark
    benchmark(profile=True)


if __name__ == "__main__":
    main()
