"""Measure canonical masks1..14, compare actual searches, and confirm finalists."""
import argparse
import csv
from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import random
import time

from .benchmarkconfig import ROOT
from .benchmark_topology import Recorder,control,dump,sha
from .topology_search import Candidate,Machine
from .topology14 import Space,Policy


METHODS=('random','beam14','hierarchical','model_counts','model')


def stamp(rec,space,extra):
    path=rec.output/'plan.json'
    plan=json.loads(path.read_text())
    plan.update({'adoption_count':len(space),'canonical_fixed_masks':len(space.fixed)//2,
                 'canonical_layer_output_pairs':len(space.keys),
                 'kernel_space':{'backend':['openmp','pthread'],'layer_dynamic':[0,1],
                                 'output_dynamic':1,'row4':[0,1],'qkv':1,'ffn':1,
                                 'layer_chunk':16,'output_physical_chunk':32},
                 'fixed_depth_histogram':{str(n):sum(sum(space.layer_key(i))==n for i in space.fixed)//2 for n in range(1,15)},
                 **extra})
    dump(path,plan)
    (rec.output/'space_keys.json').write_text(json.dumps(space.keys,separators=(',',':'))+'\n')
    hashes=json.loads((rec.output/'sources.json').read_text())
    for name in ('topology14.py','benchmark_topology14.py','analyze_topology14.py'):
        source=Path(__file__).with_name(name)
        if not source.exists(): continue
        relative='src/cpuprofiler/'+name
        dest=rec.output/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes());hashes[relative]=sha(source)
    dump(rec.output/'sources.json',hashes)


def prior_best():
    path=ROOT/'results/topology_adoption_20261022/best_config.json'
    return Candidate.from_dict(json.loads(path.read_text())['candidate'])


def mirror(c,machine):
    mapping={}
    for a,b in ((machine.performance[0],machine.performance[1]),
                (machine.efficient_l2[0],machine.efficient_l2[1])):
        mapping.update(zip(a,b));mapping.update(zip(b,a))
    mapping.update(zip(machine.low_power,reversed(machine.low_power)))
    return replace(c,layer=tuple(mapping[i] for i in c.layer),output=tuple(mapping[i] for i in c.output))


def oracle(output,repeats=3):
    if repeats<3: raise ValueError('Use at least three independent blocks')
    machine=Machine.discover();space=Space(machine)
    fixed=[space.candidate(i) for i in space.fixed]
    symmetry={}
    for n in range(1,15):
        choices=[i for i in space.fixed if sum(space.layer_key(i))==n and i%8==2]
        index=random.Random(1400+n).choice(choices)
        for backend in ('openmp','pthread'):
            c=replace(space.candidate(index),backend=backend)
            alt=mirror(c,machine)
            if alt!=c and alt not in fixed: symmetry[alt.name]=c.name
    candidates={c.name:c for c in fixed}
    for name in symmetry:
        original=candidates[symmetry[name]]
        alt=mirror(original,machine);candidates[alt.name]=alt
    previous=prior_best();candidates[previous.name]=previous
    specs=list(candidates.values())
    rec=Recorder(output,machine,specs,{'mode':'topology14_oracle','repeats':repeats,'seed':140022},references=True)
    stamp(rec,space,{'fixed_indices':space.fixed,'symmetry_pairs':symmetry,
                     'scope':'all269 fixed masks x2 backends, dynamic16/outputdynamic32,row1; plus symmetry checks and previous best'})
    anchor=control(machine)
    for backend in ('openmp','pthread'): rec.measure(replace(anchor,backend=backend),'warmup',0,0)
    fixed_names={c.name for c in fixed}
    print(f'{rec.output}: {len(fixed)} fixed cells + {len(symmetry)} symmetry checks + previous best, through14 CPUs',flush=True)
    for rep in range(1,repeats+1):
        jobs=[(c,None) for c in specs]+[(anchor,name) for name in rec.refs]
        random.Random(140022+rep).shuffle(jobs)
        rec.measure(anchor,'anchor',rep,0)
        for order,(c,ref) in enumerate(jobs,1):
            kind='reference' if ref else ('fixed14' if c.name in fixed_names else ('symmetry' if c.name in symmetry else 'prior_best'))
            rec.measure(c,kind,rep,order,ref)
            if order%16==0 or order==len(jobs): rec.measure(anchor,'anchor',rep,order)
            if order%64==0 or order==len(jobs): print(f'oracle block{rep}/{repeats}: {order}/{len(jobs)}',flush=True)
    rec.close()


def search(output,budget=112,seeds=2):
    if budget!=112: raise ValueError('The registered beam14 plan uses112 queries (14+14x7)')
    machine=Machine.discover();space=Space(machine)
    policies={(method,seed):Policy(space,method,seed) for method in METHODS for seed in range(seeds)}
    rec=Recorder(output,machine,[],{'mode':'topology14_search','repeats':1,'budget':budget,'seeds':seeds,'methods':METHODS})
    stamp(rec,space,{'scope':'entire118800 canonical joint/kernel candidates; adaptive proposals, no oracle timings',
                     'candidate_budget_per_policy':budget,'control_every_candidates':16,
                     'randomization':'shuffle method/seed within each query round; sequential inference'})
    anchor=control(machine)
    for backend in ('openmp','pthread'): rec.measure(replace(anchor,backend=backend),'warmup',0,0)
    rec.measure(anchor,'anchor',1,0)
    fields=['round','method','seed','index','name','layer_count','output_count','reason','sequence',
            'ms_per_token','incumbent_index','incumbent_name','incumbent_ms_per_token','policy_cpu_s']
    measured={};total=0
    with (rec.output/'search_trace.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for step in range(1,budget+1):
            jobs=list(policies);random.Random(140100+step).shuffle(jobs)
            prepared={}
            for method,seed in jobs:
                before=time.process_time();index=policies[method,seed].ask()
                prepared[method,seed]=(index,time.process_time()-before)
            for method,seed in jobs:
                policy=policies[method,seed]
                index,ask_cpu=prepared[method,seed]
                c=space.candidate(index)
                row=rec.measure(c,'search',1,total+1)
                before=time.process_time();policy.tell(index,math.log(row['ms_per_token']));tell_cpu=time.process_time()-before
                best=policy.recommend();measured[c.name]=c
                writer.writerow({'round':step,'method':method,'seed':seed,'index':index,'name':c.name,
                                 'layer_count':len(c.layer),'output_count':len(c.output),'reason':policy.reason,'sequence':row['sequence'],
                                 'ms_per_token':row['ms_per_token'],'incumbent_index':best,'incumbent_name':space.candidate(best).name,
                                 'incumbent_ms_per_token':math.exp(policy.observed[best]),'policy_cpu_s':ask_cpu+tell_cpu})
                file.flush();total+=1
                if total%16==0: rec.measure(anchor,'anchor',1,total)
            if step%14==0: print(f'actual search round{step}/{budget}: {total} candidate measurements, all methods reached14 CPUs',flush=True)
        if total%16: rec.measure(anchor,'anchor',1,total)
    plan=json.loads((rec.output/'plan.json').read_text());plan['candidates']=[asdict(c) for c in measured.values()]
    dump(rec.output/'plan.json',plan)
    winners=[]
    for (method,seed),policy in policies.items():
        ranked=sorted(policy.observed,key=policy.observed.get)
        winners.append({'method':method,'seed':seed,'queries':len(policy.observed),
                        'layer_sizes':sorted({sum(space.layer_key(i)) for i in policy.observed}),
                        'output_sizes':sorted({sum(k[1] for k in space.keys[i//8]) for i in policy.observed}),
                        'top3':[asdict(space.candidate(i)) for i in ranked[:3]],
                        'incumbent_observed_ms_per_token':math.exp(policy.observed[ranked[0]])})
    dump(rec.output/'winners.json',winners);rec.close()


def confirm(source,output,repeats=8):
    selection=json.loads((source/'selection.json').read_text())
    specs=[Candidate.from_dict(c) for c in selection['candidates']]
    machine=Machine.discover();space=Space(machine)
    rec=Recorder(output,machine,specs,{'mode':'topology14_confirm','repeats':repeats,'source':str(source),
                                     'selection_sha256':sha(source/'selection.json')},references=True)
    stamp(rec,space,{'scope':'independent validation of preselected finalists; all14 boundaries retained'})
    anchor=control(machine)
    for backend in ('openmp','pthread'): rec.measure(replace(anchor,backend=backend),'warmup',0,0)
    for rep in range(1,repeats+1):
        jobs=[(c,None) for c in specs]+[(anchor,name) for name in rec.refs]
        random.Random(141000+rep).shuffle(jobs);rec.measure(anchor,'anchor',rep,0)
        for order,(c,ref) in enumerate(jobs,1): rec.measure(c,'reference' if ref else 'confirmation',rep,order,ref)
        rec.measure(anchor,'anchor',rep,len(jobs))
        print(f'confirmation block{rep}/{repeats}: {len(jobs)} cells',flush=True)
    rec.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('oracle','search','confirm'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--repeats',type=int)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--budget',type=int,default=112)
    args=parser.parse_args()
    if args.mode=='oracle': oracle(args.output,args.repeats or 3)
    elif args.mode=='search': search(args.output,args.budget,args.seeds)
    elif args.source: confirm(args.source,args.output,args.repeats or 8)
    else: parser.error('confirm requires --source')


if __name__=='__main__': main()
