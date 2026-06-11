import inspect

# 1. Where does the cache helper live?
try:
    from mlx_lm.models.cache import make_prompt_cache
    print("make_prompt_cache: mlx_lm.models.cache  OK")
    print("  signature:", inspect.signature(make_prompt_cache))
except Exception as e:
    print("NOT in mlx_lm.models.cache ->", e)
    # fallbacks to probe
    import mlx_lm.models.cache as c
    print("  cache module contents:", [n for n in dir(c) if not n.startswith("_")])

# 2. What cache-related params do generate / stream_generate accept?
from mlx_lm import generate, stream_generate
print("\ngenerate params:")
print(" ", list(inspect.signature(generate).parameters))
print("stream_generate params:")
print(" ", list(inspect.signature(stream_generate).parameters))