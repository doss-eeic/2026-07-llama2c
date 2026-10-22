"""Symmetry counting, full-depth reachability, and causal search algebra."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpuprofiler.topology14 import Policy,Ridge,Space
from cpuprofiler.topology_search import Candidate,Machine

MACHINE=Machine(((0,1),(2,3)),((4,5,6,7),(8,9,10,11)),(12,13))


class Topology14Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.space=Space(MACHINE)

    def test_all16383_fixed_subsets_map_to269_classes(self):
        classes=set()
        for bits in range(1,1<<14):
            cpus=tuple(i for i in range(14) if bits&(1<<i))
            c=Candidate(cpus,cpus,'openmp',1,1,0)
            classes.add(self.space.identify(c))
        self.assertEqual(classes,set(i for i in self.space.fixed if i%8==2))
        self.assertEqual(len(classes),269)
        self.assertEqual({sum(self.space.layer_key(i)) for i in classes},set(range(1,15)))

    def test_joint_classes_roundtrip_and_kernel_index(self):
        self.assertEqual(len(self.space.keys),14850)
        self.assertEqual(len(self.space),118800)
        for pair in range(len(self.space.keys)):
            index=8*pair
            c=self.space.candidate(index)
            self.assertEqual(self.space.identify(c),index)
            self.assertEqual(len(c.cpus),len(set(c.cpus)))
        for index in (0,1,2,4,7,len(self.space)-1):
            self.assertEqual(self.space.identify(self.space.candidate(index)),index)
            for neighbor in self.space.neighbors(index):
                self.assertTrue(0<=neighbor<len(self.space))
                self.assertNotEqual(neighbor,index)

    def test_no_hidden_scores_and_all14_depths(self):
        for method in ('random','beam14','hierarchical','model_counts','model'):
            a,b=[Policy(self.space,method,3,allowed=self.space.fixed) for _ in range(2)]
            for k in range(20):
                i,j=a.ask(),b.ask();self.assertEqual(i,j)
                value=.2*math.sin(i)
                a.tell(i,value);b.tell(j,value)
            self.assertEqual({sum(self.space.layer_key(i)) for i in a.prefix},set(range(1,15)))
            self.assertEqual(a.recommend(),min(a.observed,key=a.observed.get))
        beam=Policy(self.space,'beam14',2)
        for _ in range(112):
            i=beam.ask();beam.tell(i,math.log(1+sum(self.space.layer_key(i))))
        self.assertEqual(beam.beam_depth,15)
        self.assertEqual(len(beam.observed),112)

    def test_ridge_first_update_and_invalid_observation(self):
        x=(1.,.5,-.5);model=Ridge(3);model.tell(x,2.)
        denom=1+sum(a*a for a in x)
        for actual,v in zip(model.beta,x): self.assertAlmostEqual(actual,2*v/denom)
        p=Policy(self.space,'random',allowed=self.space.fixed)
        with self.assertRaises(ValueError): p.tell(p.ask(),float('nan'))

    def test_counts_ablation_does_not_leak_smt_or_l2_occupancy(self):
        for a,b in (((0,1),(0,2)),((4,5),(4,8))):
            i=self.space.identify(Candidate(a,a,'openmp',1,1,0))
            j=self.space.identify(Candidate(b,b,'openmp',1,1,0))
            self.assertNotEqual(self.space.features(i),self.space.features(j))
            self.assertEqual(self.space.counts_features(i),self.space.counts_features(j))

    def test_finite_posterior_cache_matches_direct_ridge(self):
        p=Policy(self.space,'model',allowed=self.space.fixed)
        for _ in range(4):
            i=p.ask();p.tell(i,math.sin(i))
        for i in self.space.fixed[::13]:
            direct=p.ridge.score(self.space.features(i))
            cached=p.finite_mean[i]-.05*math.sqrt(p.finite_variance[i])
            self.assertAlmostEqual(direct,cached)


if __name__=='__main__': unittest.main()
