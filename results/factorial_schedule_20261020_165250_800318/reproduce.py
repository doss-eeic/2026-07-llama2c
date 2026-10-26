from pathlib import Path
import sys,csv,random,json,hashlib,statistics
from datetime import datetime
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.benchmark_factorial import candidates
from cpuprofiler.benchmark_distribute import run
root=ROOT/'results'/datetime.now().strftime('factorial_schedule_%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
base=candidates(root);configs={}
for layout in ('P4E4','P2E8'):
 for backend in ('openmp','pthread'):
  c=base[f'{layout}_{backend}_d1_q1_f1_r1'][0]
  for layer in (0,1):
   for output in (0,1):
    name=f'{layout}_{backend}_layer{layer}_out{output}'
    configs[name]=replace(c,omp_schedule='dynamic,16' if layer else 'static',output_dynamic=bool(output))
rows=[];expected=None
for rep in range(1,6):
 names=list(configs);random.Random(900+rep).shuffle(names)
 for order,name in enumerate(names,1):
  c=configs[name];dest=root/name/f'run_{rep:02d}'
  result=run(replace(c,output=dest,build=False))
  digest=hashlib.sha256((dest/'stdout.txt').read_bytes()).hexdigest()
  if expected is None:expected=digest
  assert digest==expected
  rows.append(dict(name=name,replicate=rep,order=order,**result))
  with (root/'trials.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('schedule decomposition',rep,'/5',root,flush=True)
summary=[]
for name in configs:
 r=[r for r in rows if r['name']==name]
 summary.append(dict(name=name,runs=len(r),mean_tokens_per_second=statistics.mean(v['tokens_per_second'] for v in r),mean_ms_per_token=statistics.mean(v['ms_per_token'] for v in r)))
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
for r in summary:print(r,flush=True)
(root/'reproduce.py').write_text(Path(__file__).read_text().replace('ROOT=Path(__file__).resolve().parents[2]', 'ROOT=Path(__file__).resolve().parents[2]'))
(root/'provenance.json').write_text(json.dumps({'screening':'factorial_20261020_164542_137621','binaries':'same screening binaries','repeat':5,'seed':900,'fusion':'both on','output_block':4,'output_chunk_physical_rows':32},indent=2)+'\n')
