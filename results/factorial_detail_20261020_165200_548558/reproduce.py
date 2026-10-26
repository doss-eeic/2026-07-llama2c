from pathlib import Path
from datetime import datetime
from dataclasses import replace
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.benchmark_factorial import candidates
from cpuprofiler.benchmark_cores import run_suite
root=ROOT/'results'/datetime.now().strftime('factorial_detail_%Y%m%d_%H%M%S_%f')
base=candidates(root)
names=['P2E8_openmp_d0_q0_f0_r0','P2E8_pthread_d0_q0_f0_r0',
       'P2E8_openmp_d1_q1_f1_r1','P2E8_pthread_d1_q1_f1_r1',
       'switch_pthread_8_14_d1','switch_openmp_8_14_d1']
configs={name:replace(base[name][0],profile=True,binary=ROOT/'build'/('factorial-detail-'+base[name][0].backend)) for name in names}
print(root,flush=True)
run_suite(configs,root,repeat=3)
(root/'reproduce.py').write_text(Path(__file__).read_text().replace('ROOT=Path(__file__).resolve().parents[2]', 'ROOT=Path(__file__).resolve().parents[2]'))
