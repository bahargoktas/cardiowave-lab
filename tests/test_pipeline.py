import numpy as np
from cardiowave.data import generate, group_split
from cardiowave.features import matrix
from cardiowave.train import run

def test_reproducible():
    a=generate(seed=5,subjects=6,windows_per_class=1,sample_rate=50,seconds=4)
    b=generate(seed=5,subjects=6,windows_per_class=1,sample_rate=50,seconds=4)
    assert np.array_equal(a[0],b[0])

def test_shapes():
    x,y,g=generate(seed=5,subjects=6,windows_per_class=1,sample_rate=50,seconds=4)
    assert x.shape==(18,200)
    assert matrix(x,50).shape==(18,14)

def test_disjoint_groups():
    _,_,g=generate(seed=5,subjects=6,windows_per_class=1,sample_rate=50,seconds=4)
    tr,va,te=group_split(g,5)
    assert not(set(g[tr])&set(g[va]))
    assert not(set(g[tr])&set(g[te]))
    assert not(set(g[va])&set(g[te]))

def test_end_to_end():
    result=run(seed=5,subjects=9,windows_per_class=1,sample_rate=50,seconds=4)
    assert 0<=result["test"]["clean"]["macro_f1"]<=1
    assert sum(result["split_samples"].values())==27
