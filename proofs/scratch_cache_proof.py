import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(MODEL)

SYSTEM = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Always be polite and clear, and never make up information."
)

def full_ids(user_text):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

# --- robust prefix: longest common TOKEN prefix of two real requests ---
a = full_ids("What is the capital of France?")
b = full_ids("Name a primary color.")
L = 0
for x, y in zip(a, b):
    if x != y:
        break
    L += 1
prefix = a[:L]
print(f"shared token-prefix length: {L}  (reqA={len(a)}, reqB={len(b)})")
print("prefix decodes to:\n", repr(tokenizer.decode(prefix)), "\n")

def gen_plain(ids, max_tokens=40):
    out = ""
    for r in stream_generate(model, tokenizer, prompt=ids, max_tokens=max_tokens):
        out += r.text
    return out

def gen_cached(ids, max_tokens=40):
    cache = make_prompt_cache(model)
    # prefill: forward pass over the prefix populates the cache, generates nothing
    mx.eval(model(mx.array([prefix]), cache=cache))
    suffix = ids[L:]
    out = ""
    for r in stream_generate(
        model, tokenizer, prompt=suffix, max_tokens=max_tokens, prompt_cache=cache
    ):
        out += r.text
    return out

req = full_ids("What is the capital of France?")

# 1. determinism check: plain path must equal itself
p1 = gen_plain(req)
p2 = gen_plain(req)
print("DETERMINISTIC (plain==plain):", p1 == p2)
if p1 != p2:
    print("  -> sampling is stochastic; correctness test needs greedy. Stop here.")
    raise SystemExit

# 2. timing + correctness
t0 = time.time(); plain = gen_plain(req); t_plain = time.time() - t0
t0 = time.time(); cached = gen_cached(req); t_cached = time.time() - t0

print("\n--- PLAIN  ---\n", plain)
print("\n--- CACHED ---\n", cached)
print("\nIDENTICAL OUTPUT:", plain == cached)
print(f"plain prefill+gen: {t_plain:.2f}s | cached: {t_cached:.2f}s")