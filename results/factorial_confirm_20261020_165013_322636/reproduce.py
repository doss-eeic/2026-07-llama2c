from pathlib import Path
import sys,csv,random,json,hashlib,statistics
from dataclasses import replace
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.benchmark_factorial import candidates
from cpuprofiler.benchmark_distribute import run
root=ROOT/'results'/datetime.now().strftime('factorial_confirm_%Y%m%d_%H%M%S_%f')
root.mkdir(parents=True)
base=candidates(root)
configs={}
for layout in ('P4E4','P2E8'):
 for backend in ('openmp','pthread'):
  c=base[f'{layout}_{backend}_d1_q1_f1_r1'][0]
  for chunk in (32,128): configs[f'{layout}_{backend}_row4_chunk{chunk}']=replace(c,output_chunk=chunk)
for name in ('P4E4_openmp_d0_q1_f1_r0','switch_pthread_8_14_d1','switch_openmp_8_14_d1',
             'legacy_parallel_P2E8','legacy_upstream_P4E4'):
 configs[name]=base[name][0]
# The exact binaries from the screening are reused; all settings are runtime options.
expected=None;rows=[]
for rep in range(1,11):
 names=list(configs);random.Random(500+rep).shuffle(names)
 for order,name in enumerate(names,1):
  c=configs[name]
  result=run(replace(c,output=root/name/f'run_{rep:02d}',build=False))
  digest=hashlib.sha256((root/name/f'run_{rep:02d}'/'stdout.txt').read_bytes()).hexdigest()
  if expected is None:expected=digest
  assert digest==expected
  rows.append(dict(name=name,replicate=rep,order=order,**result))
  with (root/'trials.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('confirmation',rep,'/10',root,flush=True)
summary=[]
for name in configs:
 r=[r for r in rows if r['name']==name]
 summary.append(dict(name=name,runs=len(r),mean_tokens_per_second=statistics.mean(v['tokens_per_second'] for v in r),mean_ms_per_token=statistics.mean(v['ms_per_token'] for v in r)))
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
for r in sorted(summary,key=lambda r:r['mean_ms_per_token']):print(r,flush=True)
(root/'reproduce.py').write_text(Path(__file__).read_text().replace('ROOT=Path(__file__).resolve().parents[2]', 'ROOT=Path(__file__).resolve().parents[2]'))
(root/'provenance.json').write_text(json.dumps({'screening':'factorial_20261020_164542_137621','binaries':'same binaries as screening; no recompilation','seed':500,'repeat':10},indent=2)+'\n')
