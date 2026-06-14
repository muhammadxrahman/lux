"""Microbenchmark: how much does the sampler cost per token, and is the C++
kernel actually worth it?

The sampler runs once per generated token, on the GPU scheduler thread, so its
per-call cost is a direct tax on tokens/sec. This compares three ways to get
from a logits vector to a probability vector at a realistic vocab size:

  1. numpy        - the pure-Python reference (sampler_reference.sample_logits)
  2. cpp_list     - the ORIGINAL hot path: numpy -> Python list -> C++ -> numpy
  3. cpp_zerocopy - the NEW hot path: numpy buffer -> C++ -> numpy

Run after ./build.sh:

    python3 bench/bench_sampler.py

No model download required.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csampler
from sampler_reference import sample_logits as py_sample

VOCAB = 128_256          # Llama 3.2 vocabulary size
ITERS = 300
WARMUP = 30
# A non-trivial setting so top-k and top-p both do real work.
TEMP, TOP_K, TOP_P = 0.8, 50, 0.95

rng = np.random.default_rng(0)
# float32 is what MLX hands the sampler in production.
logits_f32 = rng.normal(0, 3, size=VOCAB).astype(np.float32)


def run_numpy():
    return py_sample(logits_f32, temperature=TEMP, top_k=TOP_K, top_p=TOP_P)


def run_cpp_list():
    # Mirrors the original sampling.py: cast to f64, build a Python list,
    # cross into C++, rebuild a numpy array.
    flat = np.asarray(logits_f32, dtype=np.float64)
    return np.asarray(csampler.sample_logits(flat.tolist(), TEMP, TOP_K, TOP_P))


def run_cpp_zerocopy():
    return np.asarray(csampler.sample_logits_np(logits_f32, TEMP, TOP_K, TOP_P))


def time_it(fn):
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    return (time.perf_counter() - t0) / ITERS * 1e6  # microseconds/call


def main():
    # Sanity: all three must agree before timing means anything.
    a, b, c = run_numpy(), run_cpp_list(), run_cpp_zerocopy()
    assert np.allclose(a, b, atol=1e-9) and np.allclose(a, c, atol=1e-9), \
        "implementations disagree -- run proofs/check_sampler.py"

    results = {
        "numpy (reference)": time_it(run_numpy),
        "cpp_list (old hot path)": time_it(run_cpp_list),
        "cpp_zerocopy (new hot path)": time_it(run_cpp_zerocopy),
    }

    baseline = results["numpy (reference)"]
    print(f"vocab={VOCAB:,}  iters={ITERS}  temp={TEMP} top_k={TOP_K} top_p={TOP_P}\n")
    print(f"{'approach':<30}{'us/token':>12}{'vs numpy':>12}")
    print("-" * 54)
    for name, us in results.items():
        print(f"{name:<30}{us:>12.1f}{baseline / us:>11.2f}x")


if __name__ == "__main__":
    main()
