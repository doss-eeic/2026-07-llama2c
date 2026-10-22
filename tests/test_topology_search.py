"""Search validity, online causality and posterior algebra, without performance assertions."""
from dataclasses import asdict
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpuprofiler.topology_search import Candidate, Machine, SearchPolicy, catalog, features, layouts
from cpuprofiler.benchmark_topology import ablations, config_for, control


MACHINE = Machine(((0,1),(2,3)), ((4,5,6,7),(8,9,10,11)), (12,13))


class TopologySearchTest(unittest.TestCase):
    def test_realistic_space_and_runtime_mapping(self):
        cs = catalog(MACHINE)
        self.assertEqual(len(cs),480)
        self.assertEqual(len(cs),len(set(cs)))
        masks = {l for l,o in layouts(MACHINE) if l == o}
        self.assertIn((0,2,1,3,4,5,6,7),masks)  # compact L2
        self.assertIn((0,2,1,3,4,5,8,9),masks)  # spread L2
        self.assertIn((0,2,4,5,6,7,8,9,10,11),masks)  # no SMT
        self.assertEqual(len(set(ablations(MACHINE))-set(cs)),48)
        for c in cs:
            cfg = config_for(c,Path('/tmp/search-test'))
            self.assertEqual(cfg.cpus[:cfg.threads],c.layer)
            self.assertEqual(set(cfg.cpus),set(c.output))
            self.assertEqual(cfg.output_threads,len(c.output))
            self.assertTrue(set(MACHINE.p_primary) <= set(c.layer))
            self.assertEqual(c,Candidate.from_dict(asdict(c)))
            self.assertTrue(all(math.isfinite(v) for v in features(c,MACHINE)))

    def test_no_lookahead_repeatability_and_no_duplicates(self):
        cs = catalog(MACHINE)
        for method in ('model','model_no_topology','beam','random'):
            a,b = [SearchPolicy(cs,MACHINE,method,9) for _ in range(2)]
            for k in range(30):
                i,j = a.ask(),b.ask()
                self.assertEqual(i,j)
                value = math.sin(i)*.1
                a.tell(i,value); b.tell(j,value)
            self.assertEqual(len(a.observed),30)
            self.assertEqual(a.recommend(),min(a.observed,key=a.observed.get))
            with self.assertRaises(ValueError): a.tell(i,0.)

    def test_recursive_ridge_first_observation_exact(self):
        cs = catalog(MACHINE)
        policy = SearchPolicy(cs,MACHINE,seed=3)
        i = policy.ask()
        x = policy.x[i]
        denom = 1+sum(v*v for v in x)
        y = -.2
        policy.tell(i,y)
        for j in (0,20,479):
            dot = sum(a*b for a,b in zip(policy.x[j],x))
            self.assertAlmostEqual(policy.mean[j],dot*y/denom)
            self.assertAlmostEqual(policy.variance[j],sum(v*v for v in policy.x[j])-dot*dot/denom)

    def test_invalid_masks_and_missing_observations(self):
        with self.assertRaises(ValueError): Candidate((0,),(1,),'openmp',1,1,0)
        with self.assertRaises(ValueError): Candidate((0,),(0,0),'openmp',1,1,0)
        with self.assertRaises(ValueError): SearchPolicy([],MACHINE)
        c = control(MACHINE)
        policy = SearchPolicy([c],MACHINE)
        with self.assertRaises(ValueError): policy.recommend()
        with self.assertRaises(ValueError): policy.tell(0,float('nan'))
        policy.tell(policy.ask(),0.)
        with self.assertRaises(StopIteration): policy.ask()


if __name__ == '__main__': unittest.main()
