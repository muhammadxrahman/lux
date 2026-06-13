from openai import OpenAI
import concurrent.futures as cf
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

SYSTEM = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Always be polite and clear, and never make up information."
)

def ask(q):
    r = client.chat.completions.create(
        model="x",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": q}],
        max_tokens=30,
    )
    return r.choices[0].message.content

qs = ["Capital of Japan?", "Largest ocean?", "Speed of light?", "Tallest mountain?"]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for a in ex.map(ask, qs):
        print("-", a)