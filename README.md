# Lux Server

A local LLM inference server for Apple Silicon. Serves quantized
Llama 3.2 models over an OpenAI-compatible HTTP API, with request
batching, a custom C++ sampling kernel, prefix caching, and
Prometheus metrics.

Built on MLX (Apple's array framework) for Metal-accelerated
inference on Mac.

## Features

- OpenAI-compatible API: works with the official `openai` client
  and any tool that speaks the `/v1/chat/completions` schema.
- Token streaming over Server-Sent Events.
- Concurrent request scheduler that groups simultaneous requests
  into batches to share GPU passes.
- System-prompt prefix caching: a shared system prompt is
  processed once and reused across requests.
- Custom sampler (temperature, top-k, top-p) implemented in C++
  and bound to Python via pybind11.
- Prometheus metrics endpoint for throughput, latency, batch size,
  and cache hit rate.

## Requirements

- macOS on Apple Silicon (M-series).
- Python 3.14 or newer.
- A C++ compiler (Xcode command line tools: `xcode-select --install`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the C++ sampler:

```bash
./build.sh
```

The model is downloaded automatically on first run from Hugging Face
(`mlx-community/Llama-3.2-3B-Instruct-4bit`, about 1.8 GB).

## Running

```bash
uvicorn app.main:app
```

The server loads the model on startup, then listens on
`http://localhost:8000`.

## Usage

Standard completion:

```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-3b","messages":[{"role":"user","content":"Hello"}],"max_tokens":40}'
```

Streaming:

```bash
curl -N -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-3b","messages":[{"role":"user","content":"Hello"}],"max_tokens":40,"stream":true}'
```

With the official OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="llama-3.2-3b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=40,
)
print(resp.choices[0].message.content)
```

## Endpoints

- `POST /v1/chat/completions` - chat completions, streaming and
  non-streaming.
- `GET /health` - readiness check, reports the loaded model.
- `GET /metrics` - Prometheus metrics.

## Configuration

Settings are environment variables with the `INFER_` prefix:

- `INFER_MODEL_PATH` - model repo or path.
- `INFER_MAX_CONCURRENT_SEQS` - max sequences decoding together (default 16).
- `INFER_PREFILL_BATCH_SIZE` - max sequences prefilling together (default 8).
- `INFER_SYSTEM_PROMPT` - system prompt to precompute and cache.
- `INFER_ENABLE_PREFIX_CACHE` - toggle prefix caching (default true).

## Architecture

Requests enter through a FastAPI layer that validates them against
the OpenAI schema. A single scheduler thread owns the model and all
GPU work (MLX requires GPU operations to stay on one thread).

The scheduler runs **continuous batching**: one long-lived
generator decodes every in-flight request together, one token per
step. Each step admits any newly-arrived requests (so they join the
running batch mid-flight) and advances every active sequence by one
token; finished sequences leave and free their slot immediately.
Streaming and non-streaming requests share this single path - they
differ only in delivery (a token queue vs. a future). Streaming is
no longer serialized: many streams decode concurrently.

The same verified C++ sampler drives every request. Each request
gets its own sampler instance (via the generator's per-sequence
samplers), so its `temperature`, `top_k` and `top_p` are honored
even while it shares decode steps with other requests.

The C++ sampler is verified against a Python reference
implementation across the full parameter space. The test runs in
CI on every push.

## Testing

```bash
python3 proofs/check_sampler.py
```

This builds nothing on its own; run `./build.sh` first. The same
test runs automatically in GitHub Actions. It verifies both C++
entry points (the list-based `sample_logits` and the zero-copy
`sample_logits_np` used in the hot path) against the reference.

Continuous batching has its own correctness proof:

```bash
python3 proofs/check_continuous_batch.py
```

It asserts that a request submitted mid-flight decodes in the same
steps as one already running (real batching, not serialization),
and that text streamed token-by-token is byte-identical to the same
request decoded as a single batch (the per-sequence incremental
detokenizer doesn't corrupt output).

## Benchmarks

```bash
python3 bench/bench_sampler.py     # sampler cost per token (no model needed)
python3 bench/bench_server.py      # end-to-end throughput (needs a running server)
```

**Sampler cost.** The sampler runs once per generated token on the
GPU scheduler thread, so its per-call cost is a direct tax on
throughput. At Llama 3.2's 128,256-token vocabulary
(`bench/bench_sampler.py`, Apple Silicon):

| approach                      | µs/token | vs NumPy |
|-------------------------------|---------:|---------:|
| NumPy reference               |   1857   |  1.00x   |
| C++ via Python list (old)     |   5771   |  0.32x   |
| C++ zero-copy (current)       |   1030   |  1.80x   |

The original C++ path was **3x slower than NumPy**: converting a
128k-element logit vector to a Python list per token cost more than
the math it saved. Reading the NumPy buffer directly in C++
(`sample_logits_np`) removed that overhead, making the C++ kernel a
genuine 1.8x win and 5.6x faster than the old path.

**Continuous batching.** Aggregate throughput scales with load
while per-request latency grows sub-linearly
(`bench/bench_server.py`, M-series, 64 tokens/request):

| concurrency | tokens/sec | p50 latency |
|------------:|-----------:|------------:|
| 1           |  88        | 0.62 s      |
| 2           | 140        | 0.78 s      |
| 4           | 196        | 1.10 s      |
| 8           | 231        | 1.91 s      |

Because streams share decode steps instead of running one at a
time, streaming time-to-first-token stays bounded as concurrent
streams pile up (p50): ~100 ms at 1 stream, ~530 ms at 8. Under the
old one-at-a-time model the 8th stream's first token waited for the
previous seven to finish.

## Deployment and portability

This server targets Apple Silicon and uses MLX's Metal backend for
acceleration, which is the intended performance path.

Linux containerization was investigated. MLX's CPU backend
generates and compiles its math kernels at runtime, and this fails
against the GCC version in current slim Python base images. The
investigation (Dockerfile and smoke test) is kept in `docker/`.

The clean path to portable Linux deployment is a swappable
llama.cpp/GGUF backend behind the existing engine interface. The
engine is structured to allow this; it is not yet implemented.

## Known limitations and future work

- No request cancellation yet: if a client disconnects mid-stream,
  the sequence keeps generating until it finishes. The generator
  supports removing a sequence by id, so this is a small follow-up.
- Prefix caching covers a shared system prompt, not arbitrary
  per-conversation history. A prefix-tree (radix) cache would reuse
  any shared lead across requests.
- A prefix-cache hit deep-copies the cached KV state per request.
  This grows with load; a cache pool would avoid the per-request copy.
- The C++ sampler runs on CPU and copies logits from the GPU each
  token. This is the main single-stream cost and the reason
  throughput climbs with batch width. A Metal-native sampler would
  remove the transfer.