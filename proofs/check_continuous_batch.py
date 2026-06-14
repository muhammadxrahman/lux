"""Proof that the engine does continuous batching.

Two independent claims, both checked against the live model:

  1. OVERLAP: a request submitted while another is mid-flight decodes in the
     SAME steps as the first one -- they share GPU passes instead of running
     one after another. (This is what was impossible before: streaming used to
     be served strictly one at a time.)

  2. CORRECTNESS: the text a streaming client receives token-by-token is
     byte-identical to decoding the same request as a single batch call. The
     per-sequence incremental detokenizer must not corrupt the output.

Run (downloads the model on first use):

    python3 proofs/check_continuous_batch.py
"""

import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import engine
from app.schemas import ChatMessage


def msgs(q):
    return [ChatMessage(role="user", content=q)]


def check_overlap():
    """Instrument the scheduler: record which uids decode in each step."""
    steps = []  # list of sets of uids that produced a token in a step

    # Wrap next_generated so we can snapshot per-step uid sets.
    real_next = engine._gen.next_generated

    def wrapped_next():
        resp = real_next()
        steps.append({r.uid for r in resp})
        return resp

    engine._gen.next_generated = wrapped_next
    try:
        long_p = msgs("Write a long, detailed paragraph about the ocean.")
        short_p = msgs("Name three fruits.")

        async def main():
            # Start a long request, then add a short one shortly after so it
            # joins while the long one is still going.
            long_task = asyncio.create_task(
                engine.submit_batch(long_p, max_tokens=100, temperature=0.0)
            )
            await asyncio.sleep(0.15)
            short_task = asyncio.create_task(
                engine.submit_batch(short_p, max_tokens=40, temperature=0.0)
            )
            return await asyncio.gather(long_task, short_task)

        asyncio.run(main())
    finally:
        engine._gen.next_generated = real_next

    overlap = sum(1 for s in steps if len(s) >= 2)
    print(f"  steps total: {len(steps)}, steps with >=2 sequences: {overlap}")
    assert overlap > 0, "no step decoded two requests together -> not batching"
    print("  OVERLAP OK")


def check_stream_matches_batch():
    prompt = msgs("Explain how a hash map works.")

    # Streamed (token-by-token) text.
    pieces = []

    def consume():
        for p in engine.submit_stream(prompt, max_tokens=80, temperature=0.0):
            pieces.append(p)

    t = threading.Thread(target=consume)
    t.start()
    t.join()
    streamed = "".join(pieces)

    # Same request, single batch decode (greedy -> deterministic).
    batched = asyncio.run(
        engine.submit_batch(prompt, max_tokens=80, temperature=0.0)
    )

    print(f"  streamed {len(pieces)} pieces, {len(streamed)} chars")
    assert streamed == batched, (
        "stream/batch text differ:\n"
        f"  stream: {streamed!r}\n  batch : {batched!r}"
    )
    print("  CORRECTNESS OK (streamed text == batched text)")


def main():
    engine.start()
    print("[1] overlap")
    check_overlap()
    print("[2] stream-vs-batch correctness")
    check_stream_matches_batch()
    print("\nALL CONTINUOUS-BATCH CHECKS PASSED")


if __name__ == "__main__":
    main()
