from pathlib import Path
import sys
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpuprofiler.benchmark_cores import *
root=ROOT/'results'/datetime.now().strftime('cores_confirm_%Y%m%d_%H%M%S_%f')
base=BenchmarkConfig(P4E4,8,root,binary=ROOT/'build/parallel-run',backend='parallel',output_block=1)
configs={
'upstream_P4E4':replace(base,backend='upstream',binary=ROOT/'build/core-upstream'),
'upstream_all14':replace(base,backend='upstream',binary=ROOT/'build/core-upstream',cpus=ALL14,threads=14),
'optimized_P4E4':base,
'optimized_P2E8_row1':replace(base,cpus=P2E8,threads=10),
'optimized_P2E8_row4':replace(base,cpus=P2E8,threads=10,output_block=4),
'optimized_P2E8_guided':replace(base,cpus=P2E8,threads=10,omp_schedule='guided,16'),
'optimized_all14':replace(base,cpus=ALL14,threads=14),
'optimized_noSMT12':replace(base,cpus=(*P2E8,12,13),threads=12),
}
print(root,flush=True)
run_suite(configs,root,repeat=10)
(root/'reproduce.py').write_text(Path(__file__).read_text())
