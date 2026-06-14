# Runs in container, on MLX's CPU backend.
# Goal: prove the real code path executes on Linux/CPU, not to be fast.
import mlx.core as mx
from mlx_lm import load, generate, stream_generate, batch_generate
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
print("loading model on CPU backend...")
model, tokenizer = load(MODEL)
print("loaded.")

msgs = [{"role": "user", "content": "Say hello in one sentence."}]
prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

# 1. plain generate (simplest path)
print("\n[1] generate:")
print(generate(model, tokenizer, prompt=prompt_str, max_tokens=20))

# 2. stream_generate WITH a custom sampler (your Route A path)
print("\n[2] stream_generate + custom sampler:")
def greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)
out = ""
for r in stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=20, sampler=greedy_sampler):
    out += r.text
print(out)

# 3. batch_generate (your scheduler path)
print("\n[3] batch_generate:")
resp = batch_generate(model, tokenizer, prompts=[prompt_ids, prompt_ids], max_tokens=20)
print(resp.texts)

# 4. prefix-cache prefill (your Phase 3 path) — the riskiest op on CPU
print("\n[4] prefix-cache prefill:")
cache = make_prompt_cache(model)
mx.eval(model(mx.array([prompt_ids]), cache=cache))
print("prefill OK, cache populated")

print("\nALL PATHS RAN ON CPU BACKEND")