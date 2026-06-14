from app.sampling import make_sampler
import copy
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

from mlx_lm import load
from mlx_lm.generate import BatchGenerator

from app.config import settings
from app.schemas import ChatMessage
import time
import logging
from app import metrics

log = logging.getLogger("engine")


_SENTINEL = object()  # marks end-of-stream in the per-request queue


@dataclass
class _Job:
    kind: str  # "batch" or "stream"
    max_tokens: int
    prompt_ids: list[int]
    loop: Optional[asyncio.AbstractEventLoop] = None  # batch jobs only
    future: Optional[asyncio.Future] = None           # batch jobs only
    out_queue: Optional[queue.Queue] = None           # stream jobs only
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


@dataclass
class _Active:
    """Per-sequence state for a request currently inside the generator."""
    job: _Job
    tokens: list[int] = field(default_factory=list)
    detok: object = None  # a fresh StreamingDetokenizer, for stream jobs


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
        # uid -> _Active. Only ever touched by the gpu-scheduler thread.
        self._active: dict[int, _Active] = {}
        self._gen: Optional[BatchGenerator] = None

    # --- lifecycle ---
    def start(self):
        self._thread.start()
        self._ready.wait()  # block until the model is loaded on the worker

    def _run(self):
        # Load on THIS thread so the model and all GPU work share one thread.
        self.model, self.tokenizer = load(settings.model_path)
        if settings.enable_prefix_cache:
            self._build_prefix_cache()
        self._gen = self._new_generator()
        self._ready.set()
        self._loop()  # never returns

    def _new_generator(self) -> BatchGenerator:
        return BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
            completion_batch_size=settings.max_concurrent_seqs,
            prefill_batch_size=settings.prefill_batch_size,
        )

    # --- prompt encoding ---
    def _encode_ids(self, messages: list[ChatMessage]) -> list[int]:
        as_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            as_dicts, add_generation_prompt=True
        )  # default tokenize=True -> token IDs, what insert() wants

    # --- public submit APIs ---
    async def submit_batch(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
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
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
        return await fut

    def submit_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        out: queue.Queue = queue.Queue()
        self._jobs.put(
            _Job(
                kind="stream",
                max_tokens=max_tokens,
                prompt_ids=self._encode_ids(messages),
                out_queue=out,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
        while True:
            item = out.get()
            if item is _SENTINEL:
                break
            yield item

    # --- scheduler loop (runs on the gpu thread) -----------------------------
    #
    # One long-lived BatchGenerator decodes every in-flight request together,
    # one token per step. Each iteration admits any newly-arrived jobs (so they
    # join mid-flight) and then advances every active sequence by one token.
    # Batch and stream jobs share this path; they differ only in how output is
    # delivered (a Future vs. a per-request queue).
    def _loop(self):
        while True:
            self._admit_ready()

            if not self._active:
                # Nothing running and nothing admitted -> block for the next
                # arrival instead of spinning, then loop to admit it.
                job = self._jobs.get()
                self._jobs.put(job)
                continue

            start = time.time()
            try:
                responses = self._gen.next_generated()
            except Exception:  # model-level failure: don't wedge the loop
                log.exception("decode step failed; failing %d jobs", len(self._active))
                for uid, active in list(self._active.items()):
                    self._fail(uid, active)
                self._gen = self._new_generator()
                continue
            metrics.GENERATION_SECONDS.observe(time.time() - start)

            if responses:
                metrics.BATCH_SIZE.observe(len(responses))
            for r in responses:
                self._handle_response(r)

            metrics.ACTIVE_SEQUENCES.set(len(self._active))
            metrics.QUEUE_DEPTH.set(self._jobs.qsize())

    def _admit_ready(self):
        """Insert every queued job into the running generator (non-blocking)."""
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            self._admit(job)

    def _admit(self, job: _Job):
        suffix, hit = self._strip_prefix(job.prompt_ids)
        if hit:
            caches = [copy.deepcopy(self._prefix_cache)]
            metrics.PREFIX_CACHE.labels(result="hit").inc()
        else:
            caches = None  # insert() will allocate a fresh cache
            metrics.PREFIX_CACHE.labels(result="miss").inc()

        # Same verified sampler the whole server uses; one per request so each
        # request's temperature/top_k/top_p is honored within the shared batch.
        sampler = make_sampler(job.temperature, job.top_k, job.top_p)
        uid = self._gen.insert(
            [suffix],
            max_tokens=[job.max_tokens],
            caches=caches,
            samplers=[sampler],
        )[0]

        active = _Active(job=job)
        if job.kind == "stream":
            # Each stream needs its own incremental detokenizer; the property
            # hands back a fresh instance per access.
            active.detok = self.tokenizer.detokenizer
            active.detok.reset()
        self._active[uid] = active

        metrics.REQUESTS.labels(mode=job.kind).inc()
        metrics.INFLIGHT.inc()

    def _handle_response(self, r):
        active = self._active.get(r.uid)
        if active is None:
            return  # already finished/removed
        metrics.TOKENS.inc()

        # The eos "stop" token is signalled but never emitted to the client.
        if r.finish_reason != "stop":
            active.tokens.append(r.token)
            if active.job.kind == "stream":
                active.detok.add_token(r.token)
                seg = active.detok.last_segment
                if seg:
                    active.job.out_queue.put(seg)

        if r.finish_reason is not None:
            self._finish(r.uid, active)

    def _finish(self, uid: int, active: _Active):
        job = active.job
        if job.kind == "stream":
            active.detok.finalize()
            seg = active.detok.last_segment
            if seg:
                job.out_queue.put(seg)
            job.out_queue.put(_SENTINEL)
        else:
            text = self.tokenizer.decode(active.tokens)
            job.loop.call_soon_threadsafe(job.future.set_result, text)
        self._active.pop(uid, None)
        metrics.INFLIGHT.dec()

    def _fail(self, uid: int, active: _Active):
        """End a request after a model-level error without crashing the loop."""
        job = active.job
        if job.kind == "stream":
            job.out_queue.put(_SENTINEL)
        else:
            exc = RuntimeError("generation failed")
            job.loop.call_soon_threadsafe(job.future.set_exception, exc)
        self._active.pop(uid, None)
        metrics.INFLIGHT.dec()

    # --- prefix cache --------------------------------------------------------
    def _build_prefix_cache(self):
        # Tokenize the system block alone, opening the user turn, so the
        # cached prefix ends exactly where real requests' user content begins.
        # We find the shared lead by tokenizing two sentinel requests and
        # taking their common prefix (the assistant header differs per content).
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
