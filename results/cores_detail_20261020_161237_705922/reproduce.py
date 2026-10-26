from pathlib import Path
import sys
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpuprofiler.benchmark_cores import *
root=ROOT/'results'/datetime.now().strftime('cores_detail_%Y%m%d_%H%M%S_%f')
base=BenchmarkConfig(P4E4,8,root,binary=ROOT/'build/parallel-detail',backend='parallel',profile=True,output_block=1)
configs={
'upstream_P4E4':replace(base,backend='upstream',binary=ROOT/'build/core-upstream-detail'),
'upstream_all14':replace(base,backend='upstream',binary=ROOT/'build/core-upstream-detail',cpus=ALL14,threads=14),
'optimized_P4E4':base,
'optimized_P2E8':replace(base,cpus=P2E8,threads=10,output_block=4),
'optimized_all14':replace(base,cpus=ALL14,threads=14),
}
print(root,flush=True)
run_suite(configs,root,repeat=5)
(root/'reproduce.py').write_text(Path(__file__).read_text())
