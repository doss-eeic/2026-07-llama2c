"""Symmetry-reduced search from one logical CPU through all fourteen.

Local counts preserve SMT and L2 sharing. Equivalence is a hardware heuristic,
not a promise that distinct physical placements have identical measured speed.
"""
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
import random

from .topology_search import Candidate


def canonical(key):
    return (*sorted(key[:2],reverse=True),*sorted(key[2:4],reverse=True),key[4])


class Space:
    """Every nested layer/output set, modulo exchanges of equivalent cores.

    Pair entries are (layer count, output count) in two P physical cores, two
    E L2 clusters, and the LP-E L2 cluster. Layer is a subset of output.
    """
    def __init__(self,machine):
        self.machine=machine
        if ([len(g) for g in machine.performance]!=[2,2]
                or [len(g) for g in machine.efficient_l2]!=[4,4]
                or len(machine.low_power)!=2):
            raise ValueError('This 14-CPU space requires P(2xSMT2), E(2xL2 of4), LP-E2')
        self.capacities=(2,2,4,4,2)
        def local(n): return [(l,o) for l in range(n+1) for o in range(l,n+1)]
        self.keys=sorted(canonical((*p,*e,lp))
                         for p,e,lp in itertools.product(
                             itertools.combinations_with_replacement(local(2),2),
                             itertools.combinations_with_replacement(local(4),2),local(2))
                         if sum(x[0] for x in (*p,*e,lp))>0)
        self.position={key:i for i,key in enumerate(self.keys)}
        self.by_layer={}
        self.by_depth={n:[] for n in range(1,15)}
        self.fixed=[]
        for i,key in enumerate(self.keys):
            layer=tuple(p[0] for p in key)
            self.by_layer.setdefault(layer,[]).extend(range(8*i,8*i+8))
            self.by_depth[sum(layer)].extend(range(8*i,8*i+8))
            if all(l==o for l,o in key): self.fixed.extend((8*i+2,8*i+6))

    def __len__(self): return 8*len(self.keys)

    def layer_key(self,index): return tuple(p[0] for p in self.keys[index//8])

    def candidate(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        key=self.keys[index//8]
        m=self.machine
        def cpus(which):
            # Physical P representatives, then their siblings, then E and LP-E.
            p=tuple(g[0] for g,k in zip(m.performance,key[:2]) if k[which]>0)
            smt=tuple(g[1] for g,k in zip(m.performance,key[:2]) if k[which]>1)
            e=tuple(c for g,k in zip(m.efficient_l2,key[2:4]) for c in g[:k[which]])
            return (*p,*smt,*e,*m.low_power[:key[4][which]])
        bits=index%8
        return Candidate(cpus(0),cpus(1),'pthread' if bits//4 else 'openmp',(bits//2)%2,1,bits%2)

    def identify(self,c):
        if not c.qkv or not c.ffn or not c.output_dynamic: raise ValueError('Outside kernel space')
        layer,output=set(c.layer),set(c.output)
        groups=(*self.machine.performance,*self.machine.efficient_l2,self.machine.low_power)
        key=canonical(tuple((len(layer&set(g)),len(output&set(g))) for g in groups))
        return self.position[key]*8+4*int(c.backend=='pthread')+2*c.dynamic+c.row4

    def grow_layers(self,layer):
        for j,limit in enumerate(self.capacities):
            if layer[j]>=limit: continue
            ls=list(layer);ls[j]+=1
            ls=(*sorted(ls[:2],reverse=True),*sorted(ls[2:4],reverse=True),ls[4])
            yield ls

    def neighbors(self,index):
        key=self.keys[index//8]
        for bit in (1,2,4): yield index^bit
        for j,(l,o) in enumerate(key):
            for nl,no in ((l,o+1),(l+1,o),(l+1,o+1),(l,o-1),(l-1,o),(l-1,o-1)):
                if not 0<=nl<=no<=self.capacities[j]: continue
                proposed=list(key);proposed[j]=(nl,no)
                result=canonical(tuple(proposed))
                if result in self.position: yield 8*self.position[result]+index%8

    @lru_cache(maxsize=8192)
    def features(self,index):
        key=self.keys[index//8]
        def counts(col):
            p=[k[col] for k in key[:2]];e=[k[col] for k in key[2:4]];lp=key[4][col]
            return [sum(p)/4,sum(x>0 for x in p)/2,sum(max(0,x-1) for x in p)/2,
                    sum(x*x for x in p)/8,sum(e)/8,sum(x>0 for x in e)/2,
                    sum(x*x for x in e)/32,lp/2]
        ls,os=counts(0),counts(1)
        ln=sum(k[0] for k in key)/14;on=sum(k[1] for k in key)/14
        switch=int(ln!=on)
        bits=index%8;b=bits//4;d=(bits//2)%2;r=bits%2
        return (1.,*ls,*os,ln,on,ln*ln,on*on,switch,on-ln,b,d,r,
                b*d,b*r,d*r,b*switch,d*ls[2],d*ls[4],r*on,b*ls[4],
                b*ln,b*on,d*ln,r*switch,ls[4]*os[4],ls[7]*os[7])


class Ridge:
    def __init__(self,n):
        self.inverse=[[float(i==j) for j in range(n)] for i in range(n)]
        self.beta=[0.]*n

    def score(self,x):
        mean=sum(a*b for a,b in zip(x,self.beta))
        u=[sum(a*b for a,b in zip(row,x)) for row in self.inverse]
        variance=max(0.,sum(a*b for a,b in zip(x,u)))
        return mean-.05*math.sqrt(variance)

    def tell(self,x,y):
        u=[sum(a*b for a,b in zip(row,x)) for row in self.inverse]
        denom=1.+sum(a*b for a,b in zip(x,u))
        residual=y-sum(a*b for a,b in zip(x,self.beta))
        for i in range(len(x)):
            self.beta[i]+=u[i]*residual/denom
            for j in range(len(x)): self.inverse[i][j]-=u[i]*u[j]/denom


class Policy:
    """Equal-budget random, depth14 beam, hierarchical, and ridge-LCB policies.

    All receive only asked observations. Every policy first measures one fixed
    layout of each size 1..14. Beam then spends seven queries per depth1..14.
    Model scores at most256 proposals rather than118800 candidates every step.
    """
    def __init__(self,space,method,seed=0,allowed=None):
        if method not in ('random','beam14','hierarchical','model'): raise ValueError(method)
        self.space=space;self.method=method;self.rng=random.Random(seed)
        self.allowed=tuple(range(len(space))) if allowed is None else tuple(allowed)
        self.allowed_set=set(self.allowed)
        if len(self.allowed_set)!=len(self.allowed): raise ValueError('Duplicate candidates')
        self.depth={n:[i for i in space.by_depth[n] if i in self.allowed_set] for n in range(1,15)}
        fixed=[i for i in self.allowed if all(l==o for l,o in space.keys[i//8])]
        self.prefix=[]
        for n in range(1,15):
            choices=[i for i in fixed if sum(space.layer_key(i))==n]
            if not choices: raise ValueError('Every depth1..14 must be present')
            self.prefix.append(self.rng.choice(choices))
        self.fixed=tuple(fixed)
        self.observed={};self.reasons={};self.reason=''
        self.ridge=Ridge(len(space.features(self.allowed[0])))
        self.beam_depth=1;self.beam_count=0;self.beam_rows=[];self.parents=[(0,0,0,0,0)]

    def _random(self,choices=None):
        choices=self.allowed if choices is None else choices
        remaining=[i for i in choices if i not in self.observed]
        if not remaining: return None
        return self.rng.choice(remaining)

    def ask(self):
        if len(self.observed)==len(self.allowed): raise StopIteration
        k=len(self.observed)
        if k<14:
            self.reason='mandatory_depth_1_to_14'
            return self.prefix[k]
        if self.method=='random':
            self.reason='uniform_quotient'
            return self._random()
        if self.method=='beam14':
            if self.beam_depth<=14:
                target={child for p in self.parents for child in self.space.grow_layers(p)}
                pool=[i for layer in sorted(target) for i in self.space.by_layer.get(layer,())
                      if i in self.allowed_set and i not in self.observed]
                if not pool: pool=[i for i in self.depth[self.beam_depth] if i not in self.observed]
                self.reason=f'beam_depth_{self.beam_depth}'
                if pool: return self.rng.choice(pool)
                self.reason='beam_exhausted_uniform_fallback'
            return self._random()
        if self.method=='hierarchical':
            if k<56:
                self.reason='fixed_size_stratified'
                i=self._random([i for i in self.fixed if sum(self.space.layer_key(i))==1+k%14])
                if i is not None: return i
            ranked=sorted(self.observed,key=self.observed.get)
            layers=list(dict.fromkeys(self.space.layer_key(i) for i in ranked))
            for limit in (4,8,16,len(layers)):
                pool=[i for layer in layers[:limit] for i in self.space.by_layer[layer]
                      if i in self.allowed_set and i not in self.observed]
                if pool:
                    self.reason=f'refine_best_{limit}_layer_classes'
                    return self.rng.choice(pool)
            self.reason='hierarchical_uniform_fallback'
            return self._random()
        if k%5==4:
            self.reason='model_uniform_exploration'
            return self._random()
        remaining=[i for i in self.allowed if i not in self.observed]
        pool=set(self.rng.sample(remaining,min(128,len(remaining))))
        for best in sorted(self.observed,key=self.observed.get)[:4]:
            pool.update(i for i in self.space.neighbors(best) if i in self.allowed_set and i not in self.observed)
        if len(pool)>256: pool=set(self.rng.sample(sorted(pool),256))
        self.reason='ridge_lcb_global_and_graph_proposals'
        return min(sorted(pool),key=lambda i:self.ridge.score(self.space.features(i)))

    def tell(self,index,value):
        if index not in self.allowed_set or index in self.observed or not math.isfinite(value):
            raise ValueError('Need one finite, allowed observation per query')
        self.observed[index]=value;self.reasons[index]=self.reason
        if self.method=='model': self.ridge.tell(self.space.features(index),value)
        if self.method=='beam14' and len(self.observed)>14 and self.beam_depth<=14:
            if sum(self.space.layer_key(index))==self.beam_depth: self.beam_rows.append(index)
            self.beam_count+=1
            if self.beam_count==7:
                candidates=self.beam_rows or [i for i in self.observed if sum(self.space.layer_key(i))==self.beam_depth]
                self.parents=list(dict.fromkeys(self.space.layer_key(i) for i in sorted(candidates,key=self.observed.get)))[:3]
                self.beam_depth+=1;self.beam_count=0;self.beam_rows=[]

    def recommend(self):
        if not self.observed: raise ValueError('No observations')
        return min(self.observed,key=self.observed.get)
