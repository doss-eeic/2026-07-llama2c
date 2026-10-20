"""Known effects and interactions must be recovered without a throughput-scale mixup."""
import csv
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpuprofiler import analyze_factorial as analysis


class FactorialAnalysisTest(unittest.TestCase):
    def test_saved_time_and_interaction_signs(self):
        rows=[]
        for replicate in (1,2):
            for b,d,q,f,r in itertools.product((0,1),repeat=5):
                time=10-.3*b-.2*d-1*q-2*f-.1*r-.5*q*f+.4*b*d
                rows.append(dict(name=f'{b}{d}{q}{f}{r}',family='factorial',layout='test',
                                 backend='pthread' if b else 'openmp',dynamic=d,qkv=q,ffn=f,row4=r,
                                 replicate=replicate,ms_per_token=time,tokens_per_second=1000/time,
                                 stdout_sha256='same'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with (root/'trials.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with patch.object(analysis,'interval',side_effect=lambda x:(sum(x)/len(x),min(x),max(x))):
                _,effects,interactions=analysis.analyze(root)
            qkv=next(r for r in effects if r['scope']=='both' and r['factor']=='qkv')
            self.assertAlmostEqual(qkv['saved_ms_per_token'],1.25)
            synergy=next(r for r in interactions if r['scope']=='both' and
                         (r['factor_a'],r['factor_b'])==('qkv','ffn'))
            self.assertAlmostEqual(synergy['extra_saved_ms_per_token'],.5)
            redundant=next(r for r in interactions if r['scope']=='both' and
                           (r['factor_a'],r['factor_b'])==('backend','dynamic'))
            self.assertAlmostEqual(redundant['extra_saved_ms_per_token'],-.4)

    def test_constant_block_interval(self):
        self.assertEqual(analysis.interval([1,1,1,1,1]),(1,1,1))
