import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")


def ask(i):
    r = client.chat.completions.create(
        model="llama-3.2-3b",
        messages=[{"role": "user", "content": f"In one short sentence, give me fact number {i}."}],
        max_tokens=40,
    )
    return r.choices[0].message.content


start = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    results = list(ex.map(ask, range(6)))
elapsed = time.time() - start

for r in results:
    print("-", r)
print(f"\n6 concurrent requests in {elapsed:.2f}s")