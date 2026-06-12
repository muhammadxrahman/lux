import numpy as np
import mlx.core as mx
from mlx_lm import load
from sampler_reference import sample_logits, pick

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

msgs = [{"role": "user", "content": "The capital of France is"}]
ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

# One forward pass. The model returns scores for every position; we want the
# scores for the NEXT token, which live at the last position.
out = model(mx.array([ids]))
logits = np.array(out[0, -1, :]).astype(np.float64)

print("number of scores (vocab size):", logits.shape[0])
best = int(np.argmax(logits))
print("greedy pick (highest score):", best, "->", repr(tokenizer.decode([best])))

# Run our math half, then our dice half.
probs = sample_logits(logits, temperature=0.7, top_k=40, top_p=0.95)
u = np.random.random()
chosen = pick(probs, u)
print("sampled pick:", chosen, "->", repr(tokenizer.decode([chosen])))

# Sanity: probabilities must sum to 1 (allowing tiny rounding error)
print("probabilities sum to:", round(float(probs.sum()), 6))