import numpy as np
import mlx.core as mx

import csampler  # compiled C++ module


def make_sampler(temperature: float, top_k: int, top_p: float, seed: int | None = None):
    """Build a sampler for mlx-lm's generation loop.

    This single sampler serves both code paths:
      * streaming (stream_generate) hands us logits shaped (1, vocab),
      * batched generation (BatchGenerator) calls a per-sequence sampler with
        the same (1, vocab) slice.

    Either way we return the chosen token id as an mx.array shaped (1,). The
    math runs in the verified C++ kernel (temperature/top-k/top-p), and the
    pick is the same cumulative-sum draw proved in sampler_reference.py.
    """
    rng = np.random.default_rng(seed)

    def sampler(logits: mx.array) -> mx.array:
        # logits arrive as (1, vocab); drop the leading row -> (vocab,).
        # np.asarray reads MLX's buffer directly, and sample_logits_np reads
        # that buffer in C++ -- no per-token Python list of vocab floats.
        flat = np.asarray(logits[0])

        # C++ does temperature / top-k / top-p, returns probabilities.
        probs = np.asarray(
            csampler.sample_logits_np(flat, temperature, top_k, top_p)
        )

        u = rng.random()
        token_id = int(np.searchsorted(np.cumsum(probs), u))

        # Hand back the shape mlx-lm gave us: a 1-element mx.array.
        return mx.array([token_id])

    return sampler
