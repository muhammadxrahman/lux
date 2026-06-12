from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

SYSTEM = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Always be polite and clear, and never make up information."
)

def ask(with_system, q):
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": SYSTEM})
    msgs.append({"role": "user", "content": q})
    r = client.chat.completions.create(
        model="llama-3.2-3b", messages=msgs, max_tokens=40
    )
    return r.choices[0].message.content

import concurrent.futures as cf
jobs = [
    (True, "What is the capital of France?"),
    (True, "Name a primary color."),
    (False, "What is 2+2?"),          # a miss, no system prompt
    (True, "What is the largest planet?"),
]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    out = list(ex.map(lambda a: ask(*a), jobs))
for (s, q), o in zip(jobs, out):
    print(f"[{'HIT ' if s else 'MISS'}] {q} -> {o}")