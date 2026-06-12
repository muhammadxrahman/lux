import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csampler
from sampler_reference import sample_logits as py_sample

rng = np.random.default_rng(0)

# Use a realistic-sized vocab of random scores so the filters actually do work.
def random_logits(n=2000):
    return list(rng.normal(0, 3, size=n))

cases = []
for temp in [0.0, 0.5, 1.0, 1.3]:
    for top_k in [0, 1, 10, 50]:
        for top_p in [1.0, 0.9, 0.5]:
            cases.append((temp, top_k, top_p))

all_ok = True
for temp, top_k, top_p in cases:
    logits = random_logits()
    cpp = np.array(csampler.sample_logits(logits, temp, top_k, top_p))
    py = py_sample(logits, temperature=temp, top_k=top_k, top_p=top_p)
    if not np.allclose(cpp, py, atol=1e-9):
        all_ok = False
        print(f"MISMATCH temp={temp} top_k={top_k} top_p={top_p}")
        # show the few biggest differences for debugging
        diff = np.abs(cpp - py)
        worst = np.argsort(diff)[::-1][:5]
        for w in worst:
            print(f"  idx {w}: cpp={cpp[w]:.6e} py={py[w]:.6e}")

print("ALL MATCH" if all_ok else "SOME MISMATCHES, see above")