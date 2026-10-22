"""Validate recorded blocks, estimate isolated effects, and replay online searches.

The replay oracle is real recorded data, not a synthetic latency function.
Evaluation blocks never enter the search policy. Outputs preserve every query.
"""
import argparse
import bisect
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import time

from .analyze_factorial import interval, write_csv
from .benchmark_topology import control, dump, sha
from .topology_search import Candidate, Machine, SearchPolicy, catalog


def load(root):
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text())
    obj = json.loads((root/'machine.json').read_text())
    machine = Machine(tuple(map(tuple,obj['performance'])), tuple(map(tuple,obj['efficient_l2'])), tuple(obj['low_power']))
    candidates = [Candidate.from_dict(c) for c in plan['candidates']]
    if len(candidates) != len(set(candidates)):
        raise ValueError('Duplicate plan candidates')
    with (root/'trials.csv').open() as file:
        rows = list(csv.DictReader(file))
    if not (root/'complete.json').exists():
        raise ValueError('Benchmark is incomplete')
    if len(rows) != json.loads((root/'complete.json').read_text())['measurements']:
        raise ValueError('Measurement count mismatch')
    for row in rows:
        for key in ('sequence','replicate','order'):
            row[key] = int(row[key])
        for key in ('start_s','end_s','tokens_per_second','ms_per_token','wall_s'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Non-finite timing')
        if row['ms_per_token'] <= 0 or row['tokens_per_second'] <= 0:
            raise ValueError('Nonpositive timing')
        if not math.isclose(row['ms_per_token']*row['tokens_per_second'],1000.,rel_tol=1e-8):
            raise ValueError('Timing units disagree')
        if row['end_s'] <= row['start_s'] or not math.isclose(row['end_s']-row['start_s'],row['wall_s']):
            raise ValueError('Invalid elapsed time')
        directory = root/row['path']
        summary = json.loads((directory/'summary.json').read_text())
        if summary['tokens_per_second'] != row['tokens_per_second'] or sha(directory/'stdout.txt') != row['stdout_sha256']:
            raise ValueError('Raw output and CSV disagree')
        row['mid_s'] = (row['start_s']+row['end_s'])/2
    if [r['sequence'] for r in rows] != list(range(1,len(rows)+1)):
        raise ValueError('Duplicated/missing run sequence')
    if len({r['stdout_sha256'] for r in rows}) != 1:
        raise ValueError('Generation differs')
    measured = [r for r in rows if r['kind'] not in ('warmup','anchor')]
    expected_names = {c.name for c in candidates}
    if plan['args']['mode'] == 'online':
        with (root/'online_trace.csv').open() as file:
            trace = list(csv.DictReader(file))
        if len(trace) != plan['args']['budget'] or len({r['name'] for r in trace}) != len(trace):
            raise ValueError('Incomplete/duplicate online trace')
        expected_names = {r['name'] for r in trace}
        if not expected_names <= {c.name for c in candidates}:
            raise ValueError('Online trace is outside planned space')
    expected_names |= set(plan.get('reference_names',[] if plan['profile'] else ['upstream','qkvffn_branch']))
    replicates = list(range(1,plan['args']['repeats']+1))
    for rep in replicates:
        selected = [r for r in measured if r['replicate'] == rep]
        if len(selected) != len(expected_names) or {r['name'] for r in selected} != expected_names:
            raise ValueError(f'Incomplete/duplicate planned cells in block {rep}')
    if {r['replicate'] for r in measured} != set(replicates):
        raise ValueError('Unexpected replicate')
    # Interpolate only within the same replicate, bracketed by real controls.
    for rep in replicates:
        anchors = [r for r in rows if r['kind'] == 'anchor' and r['replicate'] == rep]
        timestamps = [r['mid_s'] for r in anchors]
        if len(anchors) < 2 or timestamps != sorted(timestamps):
            raise ValueError('Missing/unsorted controls')
        for row in [r for r in measured if r['replicate'] == rep]:
            i = bisect.bisect_right(timestamps,row['mid_s'])
            if i == 0 or i == len(anchors):
                raise ValueError('Unbracketed measurement')
            a,b = anchors[i-1:i+1]
            fraction = (row['mid_s']-a['mid_s'])/(b['mid_s']-a['mid_s'])
            anchor_ms = a['ms_per_token'] + fraction*(b['ms_per_token']-a['ms_per_token'])
            row['control_ms'] = anchor_ms
            row['relative_latency'] = row['ms_per_token']/anchor_ms
            row['log_relative_latency'] = math.log(row['relative_latency'])
    sources = json.loads((root/'sources.json').read_text())
    if any(sha(root/'source'/name) != digest for name,digest in sources.items()):
        raise ValueError('Saved source digest mismatch')
    dump(root/'validation.json',{'measurements':len(rows),'planned_candidates':len(candidates),
         'blocks':len(replicates),'raw_outputs_and_summaries_verified':True,
         'generation_sha256':rows[0]['stdout_sha256'],'saved_sources_verified':True,
         'controls_bracket_every_measured_cell':True})
    return machine,candidates,rows,measured


def summarize(root, candidates, rows, measured):
    """Keep raw latency and normalized latency separate; no cross-session pooling."""
    meta = {c.name:c for c in candidates}
    cells = []
    for name in sorted({r['name'] for r in measured}):
        selected = [r for r in measured if r['name'] == name]
        obj = asdict(meta[name]) if name in meta else {}
        cells.append({'name':name,'kind':selected[0]['kind'],
                      'layer':','.join(map(str,obj.get('layer',[]))),
                      'output':','.join(map(str,obj.get('output',[]))),
                      **{k:obj.get(k,'') for k in ('backend','dynamic','output_dynamic','row4','qkv','ffn')},
                      'runs':len(selected),
                      'mean_ms_per_token':statistics.mean(r['ms_per_token'] for r in selected),
                      'mean_tokens_per_second':statistics.mean(r['tokens_per_second'] for r in selected),
                      'geomean_relative_latency':math.exp(statistics.mean(r['log_relative_latency'] for r in selected)),
                      'min_relative_latency':min(r['relative_latency'] for r in selected),
                      'max_relative_latency':max(r['relative_latency'] for r in selected)})
    cells.sort(key=lambda c:c['geomean_relative_latency'])
    write_csv(root/'cells.csv',cells)
    write_csv(root/'normalized_trials.csv',measured)
    blocks = []
    for rep in sorted({r['replicate'] for r in measured}):
        selected = [r for r in rows if r['replicate'] == rep]
        anchors = [r['ms_per_token'] for r in selected if r['kind'] == 'anchor']
        blocks.append({'replicate':rep,'cells':sum(r['kind'] != 'anchor' for r in selected),
                       'controls':len(anchors),'mean_control_ms':statistics.mean(anchors),
                       'min_control_ms':min(anchors),'max_control_ms':max(anchors),
                       'sum_measured_wall_s':sum(r['wall_s'] for r in selected)})
    write_csv(root/'blocks.csv',blocks)
    return cells


def contrasts(root,candidates,measured):
    lookup = {(r['name'],r['replicate']):r for r in measured}
    specs = set(candidates)
    replicates = sorted({r['replicate'] for r in measured})
    results = []
    # Isolated treatments share backend, topology, scheduling, and row kernel.
    for c in candidates:
        if c.qkv or c.ffn:
            continue
        q,f,both = replace(c,qkv=1),replace(c,ffn=1),replace(c,qkv=1,ffn=1)
        if not {q,f,both} <= specs:
            raise ValueError('Incomplete fusion contrast')
        values = {k:[] for k in ('qkv_only','ffn_only','combined','synergy')}
        normalized = {k:[] for k in values}
        for rep in replicates:
            def effects(key):
                t0,tq,tf,tb = [lookup[x.name,rep][key] for x in (c,q,f,both)]
                return (t0-tq,t0-tf,t0-tb,tq+tf-tb-t0)
            for name,value in zip(values,effects('ms_per_token')): values[name].append(value)
            for name,value in zip(values,effects('relative_latency')): normalized[name].append(value)
        for name,vs in values.items():
            avg,lo,hi = interval(vs)
            results.append({'baseline':c.name,'treatment':name,'backend':c.backend,
                            'layer':','.join(map(str,c.layer)),'dynamic':c.dynamic,'row4':c.row4,
                            'blocks':len(replicates),'saved_ms_per_token':avg,'ci95_low':lo,'ci95_high':hi,
                            'saved_control_units':statistics.mean(normalized[name]),
                            'block_saved_ms':json.dumps(vs)})
    if results:
        write_csv(root/'fusion_contrasts.csv',results)
    return results


def replay(root,machine,measured,seeds=20,budgets=(12,24,48,96),raw=False):
    candidates = catalog(machine)
    lookup = {(r['name'],r['replicate']):r for r in measured if r['kind']=='adoption'}
    reps = sorted({r['replicate'] for r in measured})
    if len(reps) < 3:
        raise ValueError('Replay needs at least three independent blocks')
    if any((c.name,rep) not in lookup for c in candidates for rep in reps):
        raise ValueError('Missing adoption-space data')
    if not budgets or min(budgets)<1 or max(budgets)>len(candidates):
        raise ValueError('Invalid search budget')
    traces, results = [],[]
    methods = ('random','beam','model_no_topology','model')
    for heldout in reps:
        train = [rep for rep in reps if rep != heldout]
        def objective(row):
            return math.log(row['ms_per_token']) if raw else row['log_relative_latency']
        training = [statistics.mean(objective(lookup[c.name,rep]) for rep in train) for c in candidates]
        testing = [objective(lookup[c.name,heldout]) for c in candidates]
        wall = [sum(lookup[c.name,rep]['wall_s'] for rep in train) for c in candidates]
        oracle = min(testing)
        exhaustive = min(range(len(candidates)),key=lambda i:training[i])
        results.append({'method':'exhaustive','heldout':heldout,'seed':-1,'budget':len(candidates),
                        'selected':candidates[exhaustive].name,'heldout_regret_pct':100*math.expm1(testing[exhaustive]-oracle),
                        'training_regret_pct':0.,'simulated_run_wall_s':sum(wall),
                        'candidate_measurements':len(candidates)*len(train),'search_cpu_s':0.})
        for method in methods:
            for seed in range(seeds):
                start = time.process_time()
                policy = SearchPolicy(candidates,machine,method,seed)
                cost = 0.
                for step in range(1,max(budgets)+1):
                    index = policy.ask()
                    # The only access to timings by the search algorithm is this observation.
                    policy.tell(index,training[index])
                    cost += wall[index]
                    selected = policy.recommend()
                    regret = 100*math.expm1(testing[selected]-oracle)
                    traces.append({'method':method,'heldout':heldout,'seed':seed,'step':step,
                                   'queried':candidates[index].name,'observed_log_relative_latency':training[index],
                                   'incumbent':candidates[selected].name,'heldout_regret_pct':regret})
                    if step in budgets:
                        results.append({'method':method,'heldout':heldout,'seed':seed,'budget':step,
                                        'selected':candidates[selected].name,'heldout_regret_pct':regret,
                                        'training_regret_pct':100*math.expm1(training[selected]-min(training)),
                                        'simulated_run_wall_s':cost,'candidate_measurements':step*len(train),
                                        'search_cpu_s':time.process_time()-start})
        print(f'replay heldout block {heldout}: complete',flush=True)
    prefix = 'raw_' if raw else ''
    write_csv(root/(prefix+'search_trace.csv'),traces)
    write_csv(root/(prefix+'search_runs.csv'),results)
    summary = []
    for method in (*methods,'exhaustive'):
        for budget in (budgets if method != 'exhaustive' else (len(candidates),)):
            selected = [r for r in results if r['method']==method and r['budget']==budget]
            regrets = sorted(r['heldout_regret_pct'] for r in selected)
            summary.append({'method':method,'budget':budget,'replay_runs':len(selected),
                            'mean_heldout_regret_pct':statistics.mean(regrets),
                            'median_heldout_regret_pct':statistics.median(regrets),
                            'p90_heldout_regret_pct':regrets[math.ceil(.9*len(regrets))-1],
                            'within_1pct_rate':statistics.mean(v<=1. for v in regrets),
                            'within_3pct_rate':statistics.mean(v<=3. for v in regrets),
                            'mean_candidate_measurements':statistics.mean(r['candidate_measurements'] for r in selected),
                            'mean_simulated_run_wall_s':statistics.mean(r['simulated_run_wall_s'] for r in selected),
                            'mean_search_cpu_s':statistics.mean(r['search_cpu_s'] for r in selected)})
    write_csv(root/(prefix+'search_summary.csv'),summary)
    dump(root/(prefix+'replay_protocol.json'),{'seeds':list(range(seeds)),'budgets':budgets,'heldout_blocks':reps,
         'raw_latency_sensitivity':raw,
         'train': 'arithmetic mean log normalized latency from all other blocks',
         'objective':'minimize log relative latency',
         'recommendation':'lowest observed training latency; unobserved predictions never adopted',
         'regret':'heldout selected latency / heldout best latency - 1; hindsight oracle is optimistic',
         'measurement_cost':'budget * training block count; excludes separately reported control/warmup/confirmation',
         'caveat':'replay seeds reuse measured data; three heldout folds are not independent experiments',
         'full_catalog':len(candidates)})
    return summary


def select(root,candidates,cells,machine,online_sources=()):
    """Freeze finalists before independent confirmation; no confirmation data seen."""
    by_name = {c.name:c for c in candidates}
    adopted = set(catalog(machine))
    ranked = [by_name[r['name']] for r in cells if r['name'] in by_name and by_name[r['name']] in adopted]
    chosen = ranked[:4]
    raw_ranked = [by_name[r['name']] for r in sorted(cells,key=lambda r:r['mean_ms_per_token'])
                  if r['name'] in by_name and by_name[r['name']] in adopted]
    chosen.extend(raw_ranked[:4])
    # Keep the best from each backend and each SMT choice as challengers.
    for key in ('openmp','pthread'):
        chosen.append(next(c for c in ranked if c.backend==key))
    chosen.extend([control(machine),replace(control(machine),backend='pthread')])
    for smt in (False,True):
        chosen.append(next(c for c in ranked if bool(set(c.layer)&set(machine.p_smt)) == smt))
    chosen = list(dict.fromkeys(chosen))
    online_evidence = []
    for source in online_sources:
        plan = json.loads((source/'plan.json').read_text())
        if plan['args']['mode'] != 'online' or not (source/'complete.json').exists():
            raise ValueError('Need a completed online search')
        result = json.loads((source/'selection.json').read_text())
        winner = Candidate.from_dict(result['candidates'][0])
        if winner not in adopted:
            raise ValueError('Online winner outside adoption catalog')
        chosen.append(winner)
        online_evidence.append({'source':str(source),'trace_sha256':sha(source/'online_trace.csv')})
    chosen = list(dict.fromkeys(chosen))
    dump(root/'selection.json',{'rule':'normalized top4 + raw top4 + best each backend/SMT class + two historical controls + actual online winners',
                               'online_evidence':online_evidence,
                               'candidates':[asdict(c) for c in chosen]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--no-replay',action='store_true')
    parser.add_argument('--raw-replay',action='store_true',help='Also replay with raw latency as a sensitivity check')
    parser.add_argument('--include-online',action='append',type=Path,default=[],help='Include actual online winner in confirmation shortlist')
    args = parser.parse_args()
    source_hashes = {}
    for source in Path(__file__).parent.glob('*.py'):
        dest = args.source/'analysis_source'/'src'/'cpuprofiler'/source.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
        source_hashes[source.name] = sha(source)
    dump(args.source/'analysis_provenance.json',{'sources':source_hashes,
         'trials_sha256':sha(args.source/'trials.csv'), 'raw_replay':args.raw_replay,
         'no_replay':args.no_replay})
    machine,candidates,rows,measured = load(args.source)
    cells = summarize(args.source,candidates,rows,measured)
    plan = json.loads((args.source/'plan.json').read_text())
    if plan['args']['mode']!='online': contrasts(args.source,candidates,measured)
    if plan['args']['mode']=='sweep':
        select(args.source,candidates,cells,machine,args.include_online)
        if not args.no_replay:
            replay(args.source,machine,measured)
            if args.raw_replay: replay(args.source,machine,measured,raw=True)
    print('Top measured cells (normalized exploratory ranking):',flush=True)
    for row in cells[:8]:
        print(f"{row['name']}: {row['mean_tokens_per_second']:.2f} tok/s, relative={row['geomean_relative_latency']:.4f}",flush=True)


if __name__ == '__main__': main()
