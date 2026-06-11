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

    # --- lifecycle ---
    def start(self):
        self._thread.start()
        self._ready.wait()  # block until the model is loaded on the worker

    def _run(self):
        # Load on THIS thread so the model and all GPU work share one thread.
        self.model, self.tokenizer = load(settings.model_path)
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
        print(f"[scheduler] running batch of {len(jobs)}")
        prompts = [j.prompt_ids for j in jobs]
        max_toks = [j.max_tokens for j in jobs]
        try:
            resp = batch_generate(
                self.model, self.tokenizer, prompts=prompts, max_tokens=max_toks
            )
            for j, text in zip(jobs, resp.texts):
                j.loop.call_soon_threadsafe(j.future.set_result, text)
        except Exception as exc:  # deliver the error to every waiter in the batch
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


engine = Engine()