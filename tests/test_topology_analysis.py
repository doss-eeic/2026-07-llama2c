"""Known control drift and paired effect algebra must survive saved-file parsing."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict, replace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpuprofiler.analyze_topology import load, contrasts
from cpuprofiler.topology_search import Candidate


class TopologyAnalysisTest(unittest.TestCase):
    def fixture(self,root):
        c = Candidate((0,),(0,),'openmp',1,1,0)
        (root/'machine.json').write_text(json.dumps({'performance':[[0,1]],'efficient_l2':[[2,3]],'low_power':[]}))
        (root/'plan.json').write_text(json.dumps({'candidates':[asdict(c)],'profile':True,'args':{'repeats':3,'mode':'confirm'}}))
        digest = hashlib.sha256(b'same text').hexdigest()
        rows = []
        for rep in (1,2,3):
            for kind,mid,ms in (('anchor',1.,10.),('confirmation',3.,7.5),('anchor',5.,20.)):
                name = c.name if kind=='confirmation' else 'control'
                dest = root/f'run{len(rows)}'
                dest.mkdir()
                (dest/'stdout.txt').write_bytes(b'same text')
                (dest/'summary.json').write_text(json.dumps({'tokens_per_second':1000/ms}))
                rows.append(dict(sequence=len(rows)+1,name=name,kind=kind,replicate=rep,order=0,
                                 start_s=mid-.1+10*rep,end_s=mid+.1+10*rep,tokens_per_second=1000/ms,
                                 ms_per_token=ms,wall_s=.2,stdout_sha256=digest,path=dest.name))
        (root/'complete.json').write_text(json.dumps({'measurements':len(rows)}))
        (root/'sources.json').write_text('{}')
        with (root/'trials.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return rows

    def test_control_interpolation_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=self.fixture(root)
            _,_,_,measured=load(root)
            self.assertEqual(len(measured),3)
            self.assertTrue(all(r['relative_latency']==.5 for r in measured))
            rows[4]['replicate']=1
            with (root/'trials.csv').open('w') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Incomplete/duplicate'): load(root)

    def test_isolated_combination_synergy_signs(self):
        base=Candidate((0,),(0,),'pthread',1,1,0,0,0)
        cs=[base,replace(base,qkv=1),replace(base,ffn=1),replace(base,qkv=1,ffn=1)]
        measured=[{'name':c.name,'replicate':rep,'ms_per_token':t,'relative_latency':t/2}
                  for rep in (1,2,3) for c,t in zip(cs,(10.,9.,8.,6.5))]
        with tempfile.TemporaryDirectory() as directory:
            with patch('cpuprofiler.analyze_topology.interval',side_effect=lambda v:(sum(v)/len(v),min(v),max(v))):
                effects=contrasts(Path(directory),cs,measured)
        saved={e['treatment']:e['saved_ms_per_token'] for e in effects}
        self.assertEqual(saved,{'qkv_only':1.,'ffn_only':2.,'combined':3.5,'synergy':.5})


if __name__ == '__main__': unittest.main()
