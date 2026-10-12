#!/usr/bin/env python3
"""Analyze instrumented llama2.c L2 scaling / CPU affinity experiments.

Standard-library only. Typical use:
  python3 analyze/analyze_l2_scaling.py --dir benchmark_affinity_stories15M/<timestamp>_l2_scaling
  python3 analyze/analyze_l2_scaling.py  # picks newest *_l2_scaling run

Produces analysis/{runs,cases,stages,comparisons,paired,perf}.csv and report.txt.
All time metrics are compute-only from stage logs, NOT end-to-end wall time.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

STAGE_NAMES = {
    '001': 'embedding',
    '002': 'attention_norm',
    '003': 'qkv',
    '004': 'rope',
    '005': 'compute_attention',
    '006': 'attention_output',
    '007': 'ffn_norm',
    '008': 'ffn_up',
    '009': 'swiglu',
    '010': 'ffn_down',
    '011': 'final',
    '102': 'layers',
    '103': 'forward',
}
PROJECTION = ('003', '006', '008', '010', '011')
TIMING_RE = re.compile(
    r'\b(?P<stage>\d{3}):[\w-]+:{1,2}\s*'
    r'(?P<seconds>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+seconds\b'
)


def get_dir(argument: str | None) -> Path:
    if argument:
        path = Path(argument).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f'Not a directory: {path}')
        return path
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / 'benchmark_affinity_stories15M'
    choices = sorted(
        (d for d in root.glob('*_l2_scaling') if d.is_dir()
         and (d / 'manifest.tsv').is_file()), key=lambda d: d.name
    )
    if not choices:
        raise ValueError(f'No *_l2_scaling experiment directories under {root}; pass --dir.')
    return choices[-1]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_timing(path: Path) -> dict:
    counts = Counter()
    totals = defaultdict(float)
    with path.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            match = TIMING_RE.search(line)
            if not match or match['stage'] not in STAGE_NAMES:
                continue
            k = match['stage']
            counts[k] += 1
            totals[k] += float(match['seconds'])
    missing = STAGE_NAMES.keys() - counts.keys()
    if missing:
        raise ValueError(f'{path}: missing stage records: {sorted(missing)}')
    ntok, nlayer_calls = counts['103'], counts['102']
    if ntok <= 0 or nlayer_calls % ntok != 0:
        raise ValueError(f'{path}: inconsistent token/layer counts')
    for stage in STAGE_NAMES:
        needed = ntok if stage in ('001', '011', '103') else nlayer_calls
        if counts[stage] != needed:
            raise ValueError(f'{path}: stage {stage} count {counts[stage]} != {needed}')
    to_ms = 1000 / ntok
    result = {
        'tokens': ntok,
        'layer_calls': nlayer_calls,
        'layers_per_token': nlayer_calls // ntok,
        'forward_ms': totals['103'] * to_ms,
        'layer_ms': totals['102'] * to_ms,
        'final_ms': totals['011'] * to_ms,
        'projection_ms': sum(totals[s] for s in PROJECTION) * to_ms,
        'accounting_error_ms': (
            totals['103'] - totals['001'] - totals['102'] - totals['011']
        ) * to_ms,
        'stages_ms': {s: totals[s] * to_ms for s in STAGE_NAMES},
        'totals_seconds': dict(totals),
    }
    return result


def med(values) -> float:
    return st.median(values)


def mad(values) -> float:
    middle = med(values)
    return med(abs(v - middle) for v in values)


def format_ms(x: float) -> str:
    return f'{x:.4f}'


def read_perf(path: Path) -> dict[str, float]:
    """Parse generic `perf stat -x,` output (PMU event strings preserved)."""
    events: dict[str, float] = {}
    with path.open(newline='', encoding='utf-8', errors='replace') as f:
        for fields in csv.reader(f):
            if len(fields) < 3 or fields[0].lstrip().startswith('#'):
                continue
            value_text = fields[0].strip()
            event_name = fields[2].strip()
            if not event_name or not value_text or '<' in value_text:
                continue
            try:
                value = float(value_text)
            except ValueError:
                continue
            if math.isfinite(value):
                events[event_name] = value
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', help='Timestamped *_l2_scaling experiment folder')
    args = parser.parse_args()
    base = get_dir(args.dir)
    config = read_tsv(base / 'config.tsv')
    manifest = read_tsv(base / 'manifest.tsv')
    if not config or not manifest:
        raise ValueError('Empty config.tsv or manifest.tsv. Was the benchmark completed?')
    cases = {r['case']: r for r in config}
    observed: dict[str, list[dict]] = defaultdict(list)
    warnings: list[str] = []
    run_rows: list[dict] = []
    all_hashes = set()
    for line in manifest:
        name = line['case']
        if name not in cases:
            raise ValueError(f'Unknown case {name!r} in manifest')
        log = base / line['log']
        if not log.is_file():
            raise ValueError(f'Log not found: {log}')
        m = read_timing(log)
        m['run'] = int(line['run'])
        m['sha256'] = line['sha256']
        observed[name].append(m)
        all_hashes.add(line['sha256'])
        run_rows.append({
            'case': name, 'run': m['run'], 'threads': cases[name]['threads'],
            'cpus': cases[name]['cpus'], 'tokens': m['tokens'],
            'layer_calls': m['layer_calls'], 'forward_ms_per_token': m['forward_ms'],
            'layers_ms_per_token': m['layer_ms'], 'final_ms_per_token': m['final_ms'],
            'projection_ms_per_token': m['projection_ms'],
            'projection_share_pct': m['projection_ms'] / m['forward_ms'] * 100,
            'tokens_per_compute_second': 1000 / m['forward_ms'],
            'accounting_error_ms_per_token': m['accounting_error_ms'],
            'stdout_sha256': m['sha256'],
        })
        if abs(m['accounting_error_ms']) > .01 * m['forward_ms']:
            warnings.append(f'Timing accounting error >1%: {name} run={m["run"]}')

    if len(all_hashes) > 1:
        warnings.append(f'Stdout differs across runs: {len(all_hashes)} distinct SHA256 hashes. '
                        'Verify generated output and any timing/status text included in stdout.')
    expected_tokens = {(m['tokens'], m['layer_calls']) for rr in observed.values() for m in rr}
    if len(expected_tokens) != 1:
        warnings.append(f'Inconsistent generation lengths: {sorted(expected_tokens)}')
    repeat_counts = {name: len(rr) for name, rr in observed.items()}
    if len(set(repeat_counts.values())) > 1 or len(observed) != len(cases):
        warnings.append(f'Incomplete/unbalanced cases: expected={len(cases)}, '
                        f'observed={len(observed)}; repetitions={repeat_counts}')
    for name, rr in observed.items():
        idx = [r['run'] for r in rr]
        if len(idx) != len(set(idx)):
            warnings.append(f'Duplicate round indices for {name}')
    # Independent stage medians are NOT necessarily additive.
    case_rows: list[dict] = []
    stage_rows: list[dict] = []
    for name in sorted(observed, key=lambda x: (int(cases[x]['threads']), x)):
        cfg, rr = cases[name], observed[name]
        times = [r['forward_ms'] for r in rr]
        fwd = med(times)
        proj_pooled = sum(r['projection_ms'] for r in rr) / sum(times) * 100
        fmin, fmax = min(times), max(times)
        case_rows.append({
            'case': name, 'threads': int(cfg['threads']), 'cpus': cfg['cpus'],
            'cluster_a': int(cfg['cluster_a']), 'cluster_b': int(cfg['cluster_b']),
            'runs': len(rr), 'tokens': '/'.join(str(x) for x in sorted({r['tokens'] for r in rr})),
            'forward_p50_ms': fwd, 'forward_mean_ms': st.mean(times),
            'forward_sd_ms': st.stdev(times) if len(times) >= 2 else 0,
            'forward_mad_ms': mad(times), 'forward_min_ms': fmin, 'forward_max_ms': fmax,
            'forward_range_pct_of_p50': (fmax - fmin) / fwd * 100,
            'layer_p50_ms': med(r['layer_ms'] for r in rr),
            'final_p50_ms': med(r['final_ms'] for r in rr),
            'projection_p50_ms': med(r['projection_ms'] for r in rr),
            'projection_share_pct_pooled': proj_pooled,
            'compute_tps_from_p50': 1000 / fwd,
        })
        for code, label in STAGE_NAMES.items():
            stage_p50_ms = med(r['stages_ms'][code] for r in rr)
            stage_pooled_share = sum(r['stages_ms'][code] for r in rr) / sum(times) * 100
            divisor = rr[0]['layers_per_token'] if code in ('002','003','004','005','006','007','008','009','010','102') else 1
            stage_rows.append({
                'case': name, 'stage_id': code, 'stage': label,
                'median_ms_per_token': stage_p50_ms,
                'median_us_per_call': stage_p50_ms * 1000 / divisor,
                'share_of_forward_pct_pooled': stage_pooled_share,
            })

    lookup = {r['case']: r for r in case_rows}
    if 'P1' in lookup:
        base_ms = lookup['P1']['forward_p50_ms']
        for row in case_rows:
            row['speedup_vs_P1'] = base_ms / row['forward_p50_ms']

    # Comparison against fastest same-cluster arrangement at equal thread count.
    comparison_rows = []
    for threads in sorted({r['threads'] for r in case_rows}):
        thread_cases = [r for r in case_rows if r['threads'] == threads]
        best = min(thread_cases, key=lambda r: r['forward_p50_ms'])
        same_cluster = [r for r in thread_cases if r['cluster_a'] == 0 or r['cluster_b'] == 0]
        control = min(same_cluster, key=lambda r: r['forward_p50_ms']) if same_cluster else None
        for r in thread_cases:
            comparison_rows.append({
                'threads': threads, 'case': r['case'],
                'best_same_n_case': best['case'],
                'speedup_vs_best_same_n': best['forward_p50_ms'] / r['forward_p50_ms'],
                'best_single_cluster_case': control['case'] if control else '',
                'speedup_vs_best_single_cluster': (
                    control['forward_p50_ms'] / r['forward_p50_ms'] if control else ''
                ),
            })

    # Block-paired ratios mitigate temporal drift. Two cases are not run
    # simultaneously, so this does not remove all within-block confounding.
    paired_rows = []
    pairs = [('A2', 'X2_1_1'), ('B2', 'X2_1_1'),
             ('A3', 'X3_2_1'), ('B3', 'X3_1_2'),
             ('A4', 'X4_2_2'), ('B4', 'X4_2_2'),
             ('X4_2_2', 'X5_3_2'), ('X4_2_2', 'X6_3_3'),
             ('X4_2_2', 'X8_4_4'), ('P1', 'X4_2_2')]
    for left, right in pairs:
        if left not in observed or right not in observed:
            continue
        lmap = {r['run']: r['forward_ms'] for r in observed[left]}
        rmap = {r['run']: r['forward_ms'] for r in observed[right]}
        paired_runs = sorted(lmap.keys() & rmap.keys())
        ratios = [lmap[n] / rmap[n] for n in paired_runs]
        if not ratios:
            continue
        paired_rows.append({
            'left': left, 'right': right, 'blocks': len(ratios),
            'median_left_over_right': med(ratios),
            'mad_ratio': mad(ratios),
            'min_ratio': min(ratios), 'max_ratio': max(ratios),
            'note': 'ratio>1 means right faster'
        })

    perf_rows = []
    perf_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in sorted((base / 'perf').glob('*_p*.csv')):
        match = re.fullmatch(r'(.+)_p\d+\.csv', p.name)
        if not match or match[1] not in cases:
            continue
        for event, value in read_perf(p).items():
            perf_samples[match[1], event].append(value)
    for (name, event), vals in sorted(perf_samples.items()):
        perf_rows.append({
            'case': name, 'event': event, 'runs': len(vals),
            'median_count': med(vals), 'min_count': min(vals), 'max_count': max(vals)
        })

    out = base / 'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out / 'runs.csv', run_rows, list(run_rows[0]))
    write_csv(out / 'cases.csv', case_rows, list(case_rows[0]))
    write_csv(out / 'stages.csv', stage_rows, list(stage_rows[0]))
    write_csv(out / 'comparisons.csv', comparison_rows, list(comparison_rows[0]))
    write_csv(out / 'paired.csv', paired_rows,
              ['left','right','blocks','median_left_over_right','mad_ratio','min_ratio','max_ratio','note'])
    write_csv(out / 'perf.csv', perf_rows,
              ['case','event','runs','median_count','min_count','max_count'])

    lines: list[str] = []
    def show(s: str = '') -> None:
        lines.append(s)
        print(s)

    show(f'LLAMA2.C L2 SCALING / AFFINITY BENCHMARK: {base}')
    show(f'Cases={len(case_rows)}, valid measured runs={len(run_rows)}')
    show('Compute-only forward timing. Each case: median of per-run mean ms/token.')
    show('Stage medians are computed independently, so layer+final need not sum to forward.')
    show('')
    show(f'{"Case":<12} {"CPU":<23} {"thr":>3} {"A/B":>5} {"p50(ms)":>9} '
         f'{"MAD":>8} {"min-max(ms)":>18} {"P1 x":>7}')
    for r in case_rows:
        speedup = r.get('speedup_vs_P1')
        speed = f'{speedup:.3f}' if speedup is not None else '-'
        show(f'{r["case"]:<12} {r["cpus"]:<23} {r["threads"]:>3} '
             f'{r["cluster_a"]}/{r["cluster_b"]:>3} {r["forward_p50_ms"]:9.4f} '
             f'{r["forward_mad_ms"]:8.4f} '
             f'{r["forward_min_ms"]:8.4f}-{r["forward_max_ms"]:8.4f} {speed:>7}')
    show('')
    show('Best per thread count:')
    for threads in sorted({r['threads'] for r in case_rows}):
        group = [r for r in case_rows if r['threads'] == threads]
        best = min(group, key=lambda x: x['forward_p50_ms'])
        show(f'  {threads} threads: {best["case"]} {best["forward_p50_ms"]:.4f} ms/token')

    show('')
    show('Targeted comparisons (ratio>1 means RHS faster, medians of full run):')
    targeted = [('A2','X2_1_1'),('B2','X2_1_1'),
                ('A3','X3_2_1'),('B3','X3_1_2'),
                ('A4','X4_2_2'),('B4','X4_2_2'),
                ('X4_2_2','X5_3_2'),('X4_2_2','X6_3_3'),
                ('X4_2_2','X8_4_4'),('P1','X4_2_2')]
    for left, right in targeted:
        if left in lookup and right in lookup:
            l, r = lookup[left]['forward_p50_ms'], lookup[right]['forward_p50_ms']
            show(f'  {left:<9} -> {right:<9}: {l/r:.3f}x ({l:.4f} -> {r:.4f} ms)')

    if paired_rows:
        show('')
        show('Block-paired comparisons (median within-round ratio; >1 means RHS faster):')
        for r in paired_rows:
            show(f'  {r["left"]:<9} -> {r["right"]:<9}: '
                 f'{r["median_left_over_right"]:.3f}x '
                 f'(MAD={r["mad_ratio"]:.3f}, n={r["blocks"]})')

    best = min(case_rows, key=lambda x: x['forward_p50_ms'])
    show(f'\nOverall fastest: {best["case"]} {best["forward_p50_ms"]:.4f} ms/token')
    show(f'Pooled projection share for fastest case: '
         f'{best["projection_share_pct_pooled"]:.2f}%')
    if perf_rows:
        show(f'Perf counters: {len(perf_rows)} case/event aggregates in perf.csv')
        show('Generic cache-misses is NOT an L2-miss count. Hybrid CPU PMUs may differ.')
    if warnings:
        show('\nWARNINGS:')
        for warning in sorted(set(warnings)):
            show(f'  - {warning}')
    show('\nFiles: analysis/{runs,cases,stages,comparisons,paired,perf}.csv, report.txt')
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
