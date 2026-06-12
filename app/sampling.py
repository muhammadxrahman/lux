import numpy as np
import mlx.core as mx

import csampler  # your compiled C++ module


def make_sampler(temperature: float, top_k: int, top_p: float, seed: int | None = None):
    """Build a sampler function for mlx-lm's generation loop.

    mlx-lm hands us logits shaped (1, vocab) and expects us to return the
    chosen token id as an mx.array shaped (1,). Inside, we use the C++
    sampler for the math and a weighted draw for the pick.
    """
    rng = np.random.default_rng(seed)

    def sampler(logits: mx.array) -> mx.array:
        # logits arrive as (1, vocab). Drop the batch row -> (vocab,),
        # move to CPU/numpy, then to a plain Python list for the C++ call.
        flat = np.array(logits[0], dtype=np.float64)

        # C++ does temperature / top-k / top-p, returns probabilities.
        probs = np.array(
            csampler.sample_logits(flat.tolist(), temperature, top_k, top_p)
        )

        # Weighted pick: same cumulative-sum draw we proved in the reference.
        u = rng.random()
        token_id = int(np.searchsorted(np.cumsum(probs), u))

        # Hand back the shape mlx-lm gave us: a 1-element mx.array.
        return mx.array([token_id])

    return sampler