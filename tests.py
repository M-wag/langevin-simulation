from main import *

def test_sample_eps():
    samples = sample_eps(N=10_000, seed=0)
    assert np.all(samples >= 5), "sample_eps, should only return samples above 5"
