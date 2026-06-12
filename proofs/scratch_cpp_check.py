import numpy as np
import csampler                       # your freshly compiled C++ module
from sampler_reference import sample_logits as py_sample

# A small, fixed set of scores so we can eyeball it.
logits = [2.0, 1.0, 0.5, -3.0, 0.1]

for temp in [0.0, 0.5, 1.0, 2.0]:
    cpp = np.array(csampler.sample_logits(logits, temp))
    py = py_sample(logits, temperature=temp)   # top_k/top_p default to off
    same = np.allclose(cpp, py, atol=1e-9)
    print(f"temp={temp}: match={same}")
    if not same:
        print("  C++:", cpp)
        print("  Py :", py)