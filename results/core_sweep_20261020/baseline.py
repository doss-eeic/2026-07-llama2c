from pathlib import Path
import sys,csv,random,statistics,json
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.benchmark_distribute import run
from cpuprofiler.benchmarkconfig import BenchmarkConfig
OUT=Path(__file__).resolve().parent/'baseline'
layouts={'P2':(0,2),'P4':(0,1,2,3),'E8':tuple(range(4,12)),
 'P2_E4':(0,2,4,5,8,9),'P4_E4':(0,1,2,3,4,5,8,9),
 'P2_E8':(0,2,*range(4,12)),'P4_E8':tuple(range(12)),
 'P2_E8_L2':(0,2,*range(4,14)),'all14':tuple(range(14))}
configs={name:BenchmarkConfig(cpus, len(cpus),OUT/name,binary=ROOT/'build/core-baseline',backend='upstream') for name,cpus in layouts.items()}
jobs=[(name,i) for i in range(3) for name in configs];random.Random(42).shuffle(jobs)
rows=[];built=False
for name,i in jobs:
 c=configs[name];r=run(replace(c,output=c.output/f'run_{i+1:02d}',build=not built));built=True
 rows.append(dict(config=name,run=i+1,cpus=','.join(map(str,c.cpus)),**r))
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print(name,i+1,round(r['tokens_per_second'],2),flush=True)
for name in configs:
 values=[r['tokens_per_second'] for r in rows if r['config']==name]
 print(name,round(statistics.mean(values),2),flush=True)
