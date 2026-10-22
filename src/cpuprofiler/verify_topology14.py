"""Audit saved topology14 results without running inference or searching again."""
import argparse
import json
from pathlib import Path
import subprocess

from .analyze_topology14 import read_csv,validate
from .benchmark_topology import dump,sha
from .benchmarkconfig import ROOT


def verify(oracle,search,confirm):
    digests=set();phases=[]
    for root in (oracle,search,confirm):
        space,plan,rows,measured=validate(root)
        digests.update(r['stdout_sha256'] for r in rows)
        env=json.loads((root/'environment.json').read_text())
        if env['model_sha256']!=sha(ROOT/'build/stories15M.bin') or env['tokenizer_sha256']!=sha(ROOT/'tokenizer.bin'):
            raise ValueError('Model/tokenizer digest differs')
        if json.loads((root/'space_keys.json').read_text())!=[[list(pair) for pair in key] for key in space.keys]:
            raise ValueError('Saved full-space catalog differs from reconstructed space')
        for row in rows:
            directory=root/row['path']
            for name in ('config.json','summary.json','stdout.txt','stderr.log','state.json','runtime.json'):
                if not (directory/name).is_file(): raise ValueError(f'Missing raw artifact: {directory/name}')
            state=json.loads((directory/'state.json').read_text())
            runtime=json.loads((directory/'runtime.json').read_text())
            if not state['before']['monotonic_s']<=row['start_s']<row['end_s']<=state['after']['monotonic_s']:
                raise ValueError('State snapshots do not bracket execution')
            if not runtime: raise ValueError('Missing recorded runtime settings')
        for ref in plan['reference_names']:
            build=root/'builds'/ref;prov=json.loads((build/'provenance.json').read_text())
            if sha(build/'run')!=prov['sha256']: raise ValueError('Reference binary digest differs')
            for name in (['run.c','src/fused.h'] if ref=='qkvffn_branch' else ['run.c']):
                pinned=subprocess.check_output(['git','show',f'{prov["revision"]}:{name}'],cwd=ROOT)
                if pinned!=(build/name).read_bytes(): raise ValueError('Reference source differs from pinned revision')
        phases.append({'directory':root.name,'measurements':len(rows),'comparison_measurements':len(measured),
                       'trials_sha256':sha(root/'trials.csv')})
    if len(digests)!=1: raise ValueError('Generation differs across phases')
    _,plan,_,_=validate(oracle)
    if (plan['canonical_fixed_masks'],plan['canonical_layer_output_pairs'],plan['adoption_count'])!=(269,14850,118800):
        raise ValueError('Search space differs from registered plan')
    space,_,_,omeasured=validate(oracle)
    fixed_expected={space.candidate(i).name for i in space.fixed}
    for rep in range(1,4):
        actual=[r['name'] for r in omeasured if r['kind']=='fixed14' and r['replicate']==rep]
        if len(actual)!=538 or set(actual)!=fixed_expected: raise ValueError('Fixed canonical oracle is incomplete')
    _,cplan,_,_=validate(confirm)
    full14=[c for c in cplan['candidates'] if len(c['layer'])==14]
    if len(full14)!=16 or cplan['args']['repeats']!=8: raise ValueError('Full14 independent boundary validation incomplete')
    trace=read_csv(search/'search_trace.csv')
    actual=[]
    for method in sorted({r['method'] for r in trace}):
        for seed in sorted({int(r['seed']) for r in trace}):
            rs=[r for r in trace if r['method']==method and int(r['seed'])==seed]
            sizes=sorted({int(r['layer_count']) for r in rs})
            if sizes!=list(range(1,15)) or len(rs)!=112: raise ValueError('Required depth not reached')
            if method=='beam14':
                if any(sum(r['reason']==f'beam_depth_{n}' and int(r['layer_count'])==n for r in rs)!=7 for n in range(1,15)):
                    raise ValueError('Beam did not finish all14 growth depths')
            actual.append({'method':method,'seed':seed,'queries':len(rs),'layer_sizes':sizes})
    metrics=json.loads((oracle/'report_metrics.json').read_text())
    if sum(p['measurements'] for p in phases)!=sum(p['total'] for p in metrics['costs']):
        raise ValueError('Report measurement count mismatch')
    tests=(oracle/'tests.log').read_text()
    if 'Ran 22 tests' not in tests or not tests.rstrip().endswith('OK'): raise ValueError('Expected tests have not passed')
    result={'phases':phases,'total_measurements':sum(p['measurements'] for p in phases),
            'generation_sha256':next(iter(digests)),'actual_search_coverage':actual,
            'all_full14_depths_verified':True,'reference_sources_and_binaries_verified':True,
            'all269_fixed_classes_x2_backends_x3_blocks_verified':True,
            'independent_full14_variants':16,'independent_full14_blocks':8,
            'model_and_tokenizer_verified':True,'full_joint_oracle_claimed':False,
            'all_raw_artifacts_and_bracketing_state_snapshots_verified':True,'tests_passed':22,
            'report_sha256':sha(oracle/'README.md'),'verification_script_sha256':sha(Path(__file__))}
    dump(oracle/'final_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle',type=Path);parser.add_argument('search',type=Path);parser.add_argument('confirm',type=Path)
    args=parser.parse_args();verify(args.oracle,args.search,args.confirm)


if __name__=='__main__': main()
