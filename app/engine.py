from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

from mlx_lm import load, generate, stream_generate

from app.config import settings
from app.schemas import ChatMessage


class Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # Exactly one worker => all GPU work runs on the same thread.
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

    def load(self):
        # Load ON the gpu worker so the model lives on the same thread
        # that will later run generation.
        self._gpu.submit(self._load_impl).result()

    def _load_impl(self):
        self.model, self.tokenizer = load(settings.model_path)

    def _format(self, messages: list[ChatMessage]) -> str:
        as_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            as_dicts, add_generation_prompt=True, tokenize=False
        )

    def complete(self, messages: list[ChatMessage], max_tokens: int) -> str:
        prompt = self._format(messages)
        return self._gpu.submit(
            self._complete_impl, prompt, max_tokens
        ).result()

    def _complete_impl(self, prompt: str, max_tokens: int) -> str:
        return generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens
        )

    def stream(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> Iterator[str]:
        prompt = self._format(messages)
        # A thread-safe handoff: the gpu worker produces tokens and puts
        # them on a queue; the caller's thread consumes them.
        import queue

        q: queue.Queue[Optional[str]] = queue.Queue()
        _SENTINEL = None

        def produce():
            try:
                for chunk in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                ):
                    q.put(chunk.text)
            finally:
                q.put(_SENTINEL)

        self._gpu.submit(produce)

        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield item


engine = Engine()