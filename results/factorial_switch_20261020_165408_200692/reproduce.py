from pathlib import Path
import sys,csv,random,json,hashlib,statistics
from datetime import datetime
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.benchmark_factorial import candidates
from cpuprofiler.benchmark_cores import ALL14
from cpuprofiler.benchmark_distribute import run
root=ROOT/'results'/datetime.now().strftime('factorial_switch_%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
base=candidates(root);configs={}
for backend in ('openmp','pthread'):
 c=base[f'P4E4_{backend}_d1_q1_f1_r0'][0]
 configs[f'{backend}_fixed8']=c
 configs[f'{backend}_switch8_14']=replace(c,cpus=ALL14,output_threads=14)
 configs[f'{backend}_fixed14']=replace(c,cpus=ALL14,threads=14,output_threads=14)
rows=[];expected=None
for rep in range(1,11):
 names=list(configs);random.Random(1200+rep).shuffle(names)
 for order,name in enumerate(names,1):
  c=configs[name];dest=root/name/f'run_{rep:02d}'
  result=run(replace(c,output=dest,build=False))
  digest=hashlib.sha256((dest/'stdout.txt').read_bytes()).hexdigest()
  if expected is None:expected=digest
  assert digest==expected
  rows.append(dict(name=name,replicate=rep,order=order,**result))
  with (root/'trials.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('switch comparison',rep,'/10',root,flush=True)
summary=[]
for name in configs:
 r=[r for r in rows if r['name']==name]
 summary.append(dict(name=name,runs=len(r),mean_tokens_per_second=statistics.mean(v['tokens_per_second'] for v in r),mean_ms_per_token=statistics.mean(v['ms_per_token'] for v in r)))
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
for r in summary:print(r,flush=True)
(root/'reproduce.py').write_text(Path(__file__).read_text().replace('ROOT=Path(__file__).resolve().parents[2]', 'ROOT=Path(__file__).resolve().parents[2]'))
(root/'provenance.json').write_text(json.dumps({'screening':'factorial_20261020_164542_137621','binaries':'same screening binaries','repeat':10,'seed':1200,'fusion':'QKV and FFN on','dynamic':'layer16, output32 physical rows','output_block':1},indent=2)+'\n')
