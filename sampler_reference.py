import numpy as np


def _softmax(x):
    # Turn scores into probabilities that sum to 1.
    # We subtract the max first because exp() of a big number overflows;
    # subtracting a constant doesn't change the result, it just keeps the
    # numbers small and safe. This trick is called "numerical stability".
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """The deterministic math half. Scores in, probabilities out."""
    logits = np.asarray(logits, dtype=np.float64).copy()  # don't touch caller's data

    # Greedy special case: temperature 0 means "always pick the best token".
    # We express that as a probability of 1.0 on the single highest score.
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[int(np.argmax(logits))] = 1.0
        return probs

    # STEP 1 - temperature: divide every score.
    # smaller temp -> sharper (predictable), larger temp -> flatter (creative)
    logits = logits / temperature

    # STEP 2 - top-k: keep only the k highest scores, set the rest to -inf.
    # -inf becomes probability 0 after softmax, so those tokens can't be chosen.
    if 0 < top_k < logits.shape[0]:
        kth_best = np.partition(logits, -top_k)[-top_k]  # value of the k-th largest
        logits[logits < kth_best] = -np.inf

    probs = _softmax(logits)

    # STEP 3 - top-p: keep the fewest top tokens whose probabilities reach p,
    # drop the long tail, then renormalize so the survivors sum to 1 again.
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]            # token ids, most likely first
        running_total = np.cumsum(probs[order])
        how_many = int(np.searchsorted(running_total, top_p)) + 1
        keep = order[:how_many]
        trimmed = np.zeros_like(probs)
        trimmed[keep] = probs[keep]
        probs = trimmed / trimmed.sum()

    return probs


def pick(probs, u):
    """The dice-roll half. Given probabilities and one random number u in
    [0, 1), return a token. Same probs + same u always give the same token,
    which is exactly what makes this testable against the C++ version."""
    running_total = np.cumsum(probs)
    return int(np.searchsorted(running_total, u))