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
- `INFER_MAX_BATCH_SIZE` - max requests per batch (default 8).
- `INFER_SYSTEM_PROMPT` - system prompt to precompute and cache.
- `INFER_ENABLE_PREFIX_CACHE` - toggle prefix caching (default true).

## Architecture

Requests enter through a FastAPI layer that validates them against
the OpenAI schema. A single scheduler thread owns the model and all
GPU work (MLX requires GPU operations to stay on one thread). The
scheduler drains a queue of pending requests, groups non-streaming
ones into a batch, and runs them together. Streaming requests run
one at a time through the sampler.

The same verified C++ sampler drives **both** paths. Batched
requests use one sampler instance per request (via
`BatchGenerator`'s per-sequence samplers), so each request's
`temperature`, `top_k` and `top_p` are honored even when requests
share a batch.

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

## Benchmarks

```bash
python3 bench/bench_sampler.py     # sampler cost per token (no model needed)
python3 bench/bench_server.py      # end-to-end throughput (needs a running server)
```

The sampler runs once per generated token on the GPU scheduler
thread, so its per-call cost is a direct tax on throughput. At
Llama 3.2's 128,256-token vocabulary (`bench/bench_sampler.py`,
Apple Silicon):

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

- Streaming requests are served one at a time, not batched.
  Batching streamed requests requires a lower-level generation API.
- Prefix caching covers a shared system prompt, not arbitrary
  per-conversation history.
- The C++ sampler runs on CPU and copies logits from the GPU each
  token. A Metal-native sampler would remove that transfer.