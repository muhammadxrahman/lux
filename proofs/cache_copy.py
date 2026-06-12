import copy
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(MODEL)

SYSTEM = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Always be polite and clear, and never make up information."
)

def full_ids(user_text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

a = full_ids("What is the capital of France?")
b = full_ids("Name a primary color.")
L = 0
for x, y in zip(a, b):
    if x != y: break
    L += 1
prefix = a[:L]

# build and prefill ONE prefix cache
base = make_prompt_cache(model)
mx.eval(model(mx.array([prefix]), cache=base))

# per-prompt copies so the batch can't cross-contaminate
caches = [copy.deepcopy(base) for _ in range(2)]
suffixes = [a[L:], b[L:]]

resp = batch_generate(
    model, tokenizer, prompts=suffixes,
    prompt_caches=caches, max_tokens=40,
)
print("cached-batch texts:", resp.texts)

# compare against the plain no-cache batch on full prompts
plain = batch_generate(model, tokenizer, prompts=[a, b], max_tokens=40)
print("plain-batch texts: ", plain.texts)
print("MATCH:", resp.texts == plain.texts)