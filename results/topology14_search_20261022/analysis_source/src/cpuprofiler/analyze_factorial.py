"""Blocked paired contrasts: positive values mean saved milliseconds per token."""
import csv
import json
import itertools
from pathlib import Path
import random
import statistics

FACTORS = ('backend', 'dynamic', 'qkv', 'ffn', 'row4')


def interval(values):
    """Resample complete replicate blocks, not correlated cells within a block."""
    rng = random.Random(42)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return statistics.mean(values), boot[int(.025*len(boot))], boot[int(.975*len(boot))]


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def analyze(output):
    output = Path(output)
    with (output / 'trials.csv').open() as file:
        rows = list(csv.DictReader(file))
    if (output / 'plan.json').exists():
        plan = json.loads((output / 'plan.json').read_text())
        if len(rows) != plan['repeat'] * len(plan['candidates']):
            raise ValueError('Run does not contain every planned replicate')
    for row in rows:
        row['ms_per_token'] = float(row['ms_per_token'])
        row['tokens_per_second'] = float(row['tokens_per_second'])
        for key in ('dynamic', 'qkv', 'ffn', 'row4', 'replicate'):
            row[key] = int(row[key])
    counts = {row['name']: sum(r['name'] == row['name'] for r in rows) for row in rows}
    if len(set(counts.values())) != 1:
        raise ValueError('Incomplete replicate blocks')
    replicates = sorted({r['replicate'] for r in rows})
    if len({r['stdout_sha256'] for r in rows}) != 1:
        raise ValueError('Generation differs across configurations')
    cells = []
    for name in dict.fromkeys(row['name'] for row in rows):
        selected = [r for r in rows if r['name'] == name]
        t = [r['ms_per_token'] for r in selected]
        mean, lo, hi = interval(t)
        meta = {k: selected[0][k] for k in ('name','family','layout','backend','dynamic','qkv','ffn','row4')}
        cells.append({**meta, 'runs':len(t), 'mean_ms_per_token':mean,
                      'mean_tokens_per_second':statistics.mean(r['tokens_per_second'] for r in selected),
                      'median_tokens_per_second':statistics.median(r['tokens_per_second'] for r in selected),
                      'ci95_low_ms':lo, 'ci95_high_ms':hi})
    write_csv(output / 'cells.csv', cells)
    factorial = [r for r in rows if r['family'] == 'factorial']
    effects, interactions = [], []
    for layout in dict.fromkeys(r['layout'] for r in factorial):
        selected = [r for r in factorial if r['layout'] == layout]
        lookup = {(r['replicate'], (int(r['backend']=='pthread'),r['dynamic'],r['qkv'],r['ffn'],r['row4'])):r['ms_per_token']
                  for r in selected}
        assert len(lookup) == len(replicates)*32
        for fixed_backend in (None,0,1):
            scope = 'both' if fixed_backend is None else ('pthread' if fixed_backend else 'openmp')
            available = list(range(5)) if fixed_backend is None else list(range(1,5))
            def time_at(rep, bits):
                return lookup[rep,tuple(bits)]
            def pairs(indices):
                rest = [i for i in range(5) if i not in indices and not (fixed_backend is not None and i==0)]
                for values in itertools.product((0,1),repeat=len(rest)):
                    bits = [0]*5
                    if fixed_backend is not None: bits[0] = fixed_backend
                    for i,value in zip(rest,values):bits[i]=value
                    yield bits
            for factor in available:
                block_effects=[]
                for rep in replicates:
                    diffs=[]
                    for bits in pairs((factor,)):
                        bits[factor]=0;t0=time_at(rep,bits)
                        bits[factor]=1;t1=time_at(rep,bits)
                        diffs.append(t0-t1)
                    block_effects.append(statistics.mean(diffs))
                mean,lo,hi=interval(block_effects)
                effects.append({'layout':layout,'scope':scope,'factor':FACTORS[factor],
                                'saved_ms_per_token':mean,'ci95_low':lo,'ci95_high':hi,
                                'blocks':len(replicates)})
            for a,b in itertools.combinations(available,2):
                block_effects=[]
                for rep in replicates:
                    diffs=[]
                    for bits in pairs((a,b)):
                        times={}
                        for va,vb in itertools.product((0,1),repeat=2):
                            bits[a]=va;bits[b]=vb;times[va,vb]=time_at(rep,bits)
                        diffs.append(times[1,0]+times[0,1]-times[1,1]-times[0,0])
                    block_effects.append(statistics.mean(diffs))
                mean,lo,hi=interval(block_effects)
                interactions.append({'layout':layout,'scope':scope,'factor_a':FACTORS[a],'factor_b':FACTORS[b],
                                     'extra_saved_ms_per_token':mean,'ci95_low':lo,'ci95_high':hi,
                                     'blocks':len(replicates)})
    write_csv(output / 'effects.csv',effects)
    write_csv(output / 'interactions.csv',interactions)
    best = sorted(cells,key=lambda row:row['mean_ms_per_token'])[:12]
    print('Fastest exploratory configurations:',flush=True)
    for row in best:
        print(f'{row["name"]}: {row["mean_tokens_per_second"]:.2f} tokens/s, {row["mean_ms_per_token"]:.4f} ms/token',flush=True)
    return cells,effects,interactions
