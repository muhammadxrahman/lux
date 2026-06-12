import inspect
import numpy as np
import mlx.core as mx
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# 1. Does stream_generate even mention a sampler parameter anywhere?
sig = inspect.signature(stream_generate)
print("stream_generate params:", list(sig.parameters))

# 2. Try passing a custom sampler and watch what it receives.
#    We log the type and shape of what mlx-lm hands us, then fall back
#    to argmax (greedy) so generation still runs.
seen = {}

def spy_sampler(logits):
    if "type" not in seen:
        seen["type"] = type(logits)
        seen["shape"] = getattr(logits, "shape", None)
        seen["ndim"] = getattr(logits, "ndim", None)
    # greedy pick so the run completes; return an mx.array token id
    tok = mx.argmax(logits, axis=-1)
    return tok

msgs = [{"role": "user", "content": "Say hello."}]
prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

try:
    out = ""
    for r in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=15, sampler=spy_sampler
    ):
        out += r.text
    print("\nsampler= ACCEPTED")
    print("logits type:", seen.get("type"))
    print("logits shape:", seen.get("shape"), "ndim:", seen.get("ndim"))
    print("output:", repr(out))
except TypeError as e:
    print("\nsampler= REJECTED ->", e)
    print("We'll need a different hook; paste this output.")