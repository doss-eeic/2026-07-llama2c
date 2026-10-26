"""Recheck the recorded result set without running any inference benchmarks."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cpuprofiler.analyze_topology import load
from cpuprofiler.benchmark_detail import parse_profile
from cpuprofiler.benchmark_topology import dump, sha


def verify():
    phases=('adoption','online_raw','online_normalized','confirm','detail')
    results=[]
    digests=set()
    profiles=0
    for phase in phases:
        directory=ROOT/'results'/f'topology_{phase}_20261022'
        machine,candidates,rows,measured=load(directory)
        specs={c.name:c for c in candidates}
        for row in measured:
            if row['name'] in specs:
                c=specs[row['name']]
                config=json.loads((directory/row['path']/'config.json').read_text())
                assert tuple(config['cpus'])==c.cpus
                assert config['threads']==len(c.layer)
                assert config['output_threads']==len(c.output)
                assert config['backend']==c.backend
                assert config['output_dynamic']==bool(c.output_dynamic)
                assert config['output_block']==(4 if c.row4 else 1)
                assert config['fuse_qkv']==bool(c.qkv) and config['fuse_ffn']==bool(c.ffn)
        for row in rows:
            digests.add(row['stdout_sha256'])
            path=directory/row['path']/'profile.jsonl.gz'
            if path.exists():
                with gzip.open(path,'rt') as stream:
                    parsed=parse_profile(stream.read())
                assert parsed['end']['forward_calls']==222
                assert len(parsed['tokens'])==222
                profiles+=1
        plan=json.loads((directory/'plan.json').read_text())
        for backend in ('openmp','pthread'):
            binary=directory/'builds'/(backend+'-detail' if plan['profile'] else backend)
            expected=json.loads((directory/'builds'/(backend+'-sources')/'binary.json').read_text())
            assert sha(binary)==expected['sha256']
        for name in ('upstream','qkvffn_branch'):
            build=directory/'builds'/name
            if build.exists():
                prov=json.loads((build/'provenance.json').read_text())
                assert sha(build/'run')==prov['sha256']
                for source in (['run.c','src/fused.h'] if name=='qkvffn_branch' else ['run.c']):
                    original=subprocess.check_output(['git','show',f'{prov["revision"]}:{source}'],cwd=ROOT)
                    assert (build/source).read_bytes()==original
        results.append({'phase':phase,'measurements':len(rows),'comparisons':len(measured),
                        'trials_sha256':sha(directory/'trials.csv')})
    assert len(digests)==1
    assert sum(r['measurements'] for r in results)==2062
    assert profiles==29
    test_log=(Path(__file__).parent/'tests.log').read_text()
    assert 'Ran 16 tests' in test_log and test_log.rstrip().endswith('OK')
    outputs={name:sha(Path(__file__).parent/name) for name in (
        'fusion_contrasts.csv','search_summary.csv','raw_search_summary.csv','best_config.json','README.md')}
    result={'phases':results,'total_measurements':2062,'profiles_parsed':profiles,
            'all_generation_sha256':next(iter(digests)),'runtime_configs_match_catalog':True,
            'built_binary_hashes_verified':True,'reference_sources_match_pinned_git_revisions':True,
            'tests':16,'report_outputs_sha256':outputs,'verification_script_sha256':sha(Path(__file__))}
    dump(Path(__file__).parent/'final_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': verify()
