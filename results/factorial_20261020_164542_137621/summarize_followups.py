"""Compute paired contrasts for the follow-up experiments; positive means saved time."""
import csv
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.analyze_factorial import interval,write_csv
RESULTS=ROOT/'results'
OUT=Path(__file__).resolve().parent

def load(name):
 with (RESULTS/name/'trials.csv').open() as f:rows=list(csv.DictReader(f))
 return {(r['name'],int(r['replicate'])):float(r['ms_per_token']) for r in rows}

def contrast(label,values,phase):
 mean,lo,hi=interval(values)
 return {'phase':phase,'contrast':label,'saved_ms_per_token':mean,'ci95_low':lo,'ci95_high':hi,'blocks':len(values)}

rows=[]
c=load('factorial_confirm_20261020_165013_322636')
for layout in ('P4E4','P2E8'):
 for chunk in (32,128):
  vals=[c[f'{layout}_pthread_row4_chunk{chunk}',i]-c[f'{layout}_openmp_row4_chunk{chunk}',i] for i in range(1,11)]
  rows.append(contrast(f'OpenMP faster than pthread: {layout}, row4, chunk{chunk}',vals,'confirmation'))
s=load('factorial_switch_20261020_165408_200692')
savings={}
for backend in ('openmp','pthread'):
 savings[backend]=[s[f'{backend}_fixed8',i]-s[f'{backend}_switch8_14',i] for i in range(1,11)]
 rows.append(contrast(f'{backend}: fixed8 -> switch8_14',savings[backend],'switch'))
 rows.append(contrast(f'{backend}: fixed14 -> switch8_14',
                      [s[f'{backend}_fixed14',i]-s[f'{backend}_switch8_14',i] for i in range(1,11)],'switch'))
rows.append(contrast('Extra switching benefit of pthread over OpenMP',
                     [a-b for a,b in zip(savings['pthread'],savings['openmp'])],'switch'))
s=load('factorial_schedule_20261020_165250_800318')
for layout in ('P4E4','P2E8'):
 for backend in ('openmp','pthread'):
  times=lambda i,l,o:s[f'{layout}_{backend}_layer{l}_out{o}',i]
  layer=[statistics.mean([times(i,0,o)-times(i,1,o) for o in (0,1)]) for i in range(1,6)]
  output=[statistics.mean([times(i,l,0)-times(i,l,1) for l in (0,1)]) for i in range(1,6)]
  interaction=[times(i,1,0)+times(i,0,1)-times(i,1,1)-times(i,0,0) for i in range(1,6)]
  for title,vals in [('layer dynamic',layer),('output dynamic',output),('layer x output synergy',interaction)]:
   rows.append(contrast(f'{layout} {backend}: {title}',vals,'stage_schedules'))
write_csv(OUT/'followup_contrasts.csv',rows)
for r in rows:print(r)
with (OUT/'trials.csv').open() as f:trials=list(csv.DictReader(f))
blocks=[{'replicate':i,'mean_ms_per_token':statistics.mean(float(r['ms_per_token']) for r in trials if int(r['replicate'])==i)} for i in range(1,6)]
write_csv(OUT/'block_means.csv',blocks)
print('Block means:',blocks)
s=load('factorial_20261020_164542_137621')
extra=[]
for layout in ('P4E4','P2E8'):
 for backend in ('openmp','pthread'):
  def t(rep,q,f):return s[f'{layout}_{backend}_d0_q{q}_f{f}_r0',rep]
  values=[t(i,1,0)+t(i,0,1)-t(i,1,1)-t(i,0,0) for i in range(1,6)]
  extra.append(contrast(f'{layout} {backend}: QKV x FFN, static, row1',values,'screening_conditional'))
write_csv(OUT/'conditional_interactions.csv',extra)
print('Conditional QKV/FFN:',extra)
