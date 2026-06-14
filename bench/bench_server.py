"""End-to-end server benchmark: throughput and latency vs concurrency.

Unlike bench_sampler.py, this needs a RUNNING server and the model loaded:

    uvicorn app.main:app          # in one terminal
    python3 bench/bench_server.py # in another

It fires N concurrent non-streaming requests (which the engine batches) and
reports aggregate tokens/sec and per-request latency, so you can see how
batching scales. Also measures time-to-first-token on the streaming path.
"""

import argparse
import threading
import time

from openai import OpenAI

PROMPT = "Explain what a hash map is in two sentences."


def one_completion(client, model, max_tokens):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    dt = time.perf_counter() - t0
    # usage may be absent; fall back to a rough token estimate from text.
    text = resp.choices[0].message.content or ""
    n_tokens = max(1, len(text.split()))
    return dt, n_tokens


def run_concurrency(client, model, concurrency, max_tokens):
    results = []
    lock = threading.Lock()

    def worker():
        dt, n = one_completion(client, model, max_tokens)
        with lock:
            results.append((dt, n))

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0

    total_tokens = sum(n for _, n in results)
    latencies = sorted(dt for dt, _ in results)
    p50 = latencies[len(latencies) // 2]
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "throughput_tok_s": total_tokens / wall,
        "p50_latency_s": p50,
    }


def measure_ttft(client, model, max_tokens):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )
    ttft = float("nan")
    # Drain the whole stream even after the first token: the server has no
    # disconnect-cancellation yet, so abandoning a stream leaves it generating
    # and would skew the next measurement.
    for event in stream:
        has_content = event.choices and event.choices[0].delta.content
        if has_content and ttft != ttft:  # ttft still NaN -> first token
            ttft = time.perf_counter() - t0
    return ttft


def measure_concurrent_stream_ttft(client, model, n, max_tokens):
    """Fire n streaming requests at once and record each one's TTFT.

    This is the headline continuous-batching result: before, streams were
    served strictly one at a time, so the k-th stream's first token waited for
    the previous k-1 to finish and TTFT grew with n. Now streams share decode
    steps, so TTFT should stay roughly flat as n rises.
    """
    ttfts = []
    lock = threading.Lock()

    def worker():
        dt = measure_ttft(client, model, max_tokens)
        with lock:
            ttfts.append(dt)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ttfts.sort()
    return {
        "n": n,
        "min_ms": ttfts[0] * 1000,
        "p50_ms": ttfts[len(ttfts) // 2] * 1000,
        "max_ms": ttfts[-1] * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="llama-3.2-3b")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="unused")

    ttft = measure_ttft(client, args.model, args.max_tokens)
    print(f"single-stream time-to-first-token: {ttft * 1000:.0f} ms\n")

    print("concurrent streaming TTFT (continuous batching keeps this flat):")
    print(f"{'streams':>8}{'min (ms)':>11}{'p50 (ms)':>11}{'max (ms)':>11}")
    print("-" * 41)
    for c in args.levels:
        r = measure_concurrent_stream_ttft(client, args.model, c, args.max_tokens)
        print(f"{r['n']:>8}{r['min_ms']:>11.0f}{r['p50_ms']:>11.0f}{r['max_ms']:>11.0f}")
    print()

    print(f"{'concurrency':>11}{'wall (s)':>10}{'tok/s':>10}{'p50 (s)':>10}")
    print("-" * 41)
    for c in args.levels:
        r = run_concurrency(client, args.model, c, args.max_tokens)
        print(f"{r['concurrency']:>11}{r['wall_s']:>10.2f}"
              f"{r['throughput_tok_s']:>10.1f}{r['p50_latency_s']:>10.2f}")


if __name__ == "__main__":
    main()
