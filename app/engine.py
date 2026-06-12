import copy
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from mlx_lm import load, stream_generate, batch_generate

from app.config import settings
from app.schemas import ChatMessage

_SENTINEL = object()  # marks end-of-stream in the per-request queue


@dataclass
class _Job:
    kind: str  # "batch" or "stream"
    max_tokens: int
    prompt_ids: Optional[list[int]] = None          # batch jobs carry token IDs
    prompt_str: Optional[str] = None                # stream jobs carry a string
    loop: Optional[asyncio.AbstractEventLoop] = None
    future: Optional[asyncio.Future] = None
    out_queue: Optional[queue.Queue] = None


class Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="gpu-scheduler", daemon=True
        )
        self._prefix_ids: list[int] = []
        self._prefix_cache = None

    # --- lifecycle ---
    def start(self):
        self._thread.start()
        self._ready.wait()  # block until the model is loaded on the worker

    def _run(self):
        # Load on THIS thread so the model and all GPU work share one thread.
        self.model, self.tokenizer = load(settings.model_path)
        if settings.enable_prefix_cache:
            self._build_prefix_cache()
        self._ready.set()
        self._loop()  # never returns

    # --- prompt encoding ---
    def _encode_ids(self, messages: list[ChatMessage]) -> list[int]:
        as_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            as_dicts, add_generation_prompt=True
        )  # default tokenize=True -> token IDs, what batch_generate wants

    def _encode_str(self, messages: list[ChatMessage]) -> str:
        as_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            as_dicts, add_generation_prompt=True, tokenize=False
        )  # string, what stream_generate wants

    # --- public submit APIs ---
    async def submit_batch(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> str:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._jobs.put(
            _Job(
                kind="batch",
                max_tokens=max_tokens,
                prompt_ids=self._encode_ids(messages),
                loop=loop,
                future=fut,
            )
        )
        return await fut

    def submit_stream(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> Iterator[str]:
        out: queue.Queue = queue.Queue()
        self._jobs.put(
            _Job(
                kind="stream",
                max_tokens=max_tokens,
                prompt_str=self._encode_str(messages),
                out_queue=out,
            )
        )
        while True:
            item = out.get()
            if item is _SENTINEL:
                break
            yield item

    # --- scheduler loop (runs on the gpu thread) ---
    def _loop(self):
        while True:
            job = self._jobs.get()  # blocks until at least one job
            if job.kind == "stream":
                self._do_stream(job)
                continue

            batch = [job]
            while len(batch) < settings.max_batch_size:
                try:
                    nxt = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if nxt.kind == "stream":
                    # can't batch a stream job: flush what we have, then stream
                    self._do_batch(batch)
                    batch = []
                    self._do_stream(nxt)
                    break
                batch.append(nxt)
            if batch:
                self._do_batch(batch)

    def _do_batch(self, jobs: list[_Job]):
        prompts, caches, used = [], [], 0
        for j in jobs:
            suffix, hit = self._strip_prefix(j.prompt_ids)
            prompts.append(suffix)
            if hit:
                caches.append(copy.deepcopy(self._prefix_cache))
                used += 1
            else:
                caches.append(None)

        # batch_generate wants either no caches, or a cache per prompt.
        # If none hit, pass None; if any hit, we must supply a full list.
        if used == 0:
            batch_caches = None
        else:
            # fill the misses with fresh empty caches so lengths line up
            batch_caches = [
                c if c is not None else make_prompt_cache(self.model)
                for c in caches
            ]

        print(f"[scheduler] batch of {len(jobs)} | prefix hits: {used}")
        try:
            resp = batch_generate(
                self.model,
                self.tokenizer,
                prompts=prompts,
                prompt_caches=batch_caches,
                max_tokens=[j.max_tokens for j in jobs],
            )
            for j, text in zip(jobs, resp.texts):
                j.loop.call_soon_threadsafe(j.future.set_result, text)
        except Exception as exc:
            for j in jobs:
                j.loop.call_soon_threadsafe(j.future.set_exception, exc)

    def _do_stream(self, job: _Job):
        try:
            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt=job.prompt_str,
                max_tokens=job.max_tokens,
            ):
                job.out_queue.put(chunk.text)
        finally:
            job.out_queue.put(_SENTINEL)
            
            
    def _build_prefix_cache(self):
        # Tokenize the system block alone, opening the user turn, so the
        # cached prefix ends exactly where real requests' user content begins.
        msgs = [{"role": "system", "content": settings.system_prompt}]
        # add_generation_prompt=False here: we want the user-turn header,
        # not the assistant header. We append the user header explicitly by
        # tokenizing a sentinel request and taking the shared lead.
        probe_a = self._encode_ids(
            [ChatMessage(role="system", content=settings.system_prompt),
             ChatMessage(role="user", content="x")]
        )
        probe_b = self._encode_ids(
            [ChatMessage(role="system", content=settings.system_prompt),
             ChatMessage(role="user", content="y")]
        )
        L = 0
        for x, y in zip(probe_a, probe_b):
            if x != y:
                break
            L += 1
        self._prefix_ids = probe_a[:L]
        cache = make_prompt_cache(self.model)
        mx.eval(self.model(mx.array([self._prefix_ids]), cache=cache))
        self._prefix_cache = cache
        print(f"[prefix] cached {L} shared tokens")

    def _strip_prefix(self, ids: list[int]):
        # Returns (suffix, used_cache_bool). Only strips on an exact match.
        p = self._prefix_ids
        if (
            self._prefix_cache is not None
            and len(ids) > len(p)
            and ids[: len(p)] == p
        ):
            return ids[len(p):], True
        return ids, False


engine = Engine()