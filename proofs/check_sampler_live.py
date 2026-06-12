from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

def gen(temp, top_p, top_k):
    out = ""
    s = client.chat.completions.create(
        model="x",
        messages=[{"role": "user", "content": "Write one short sentence about the sea."}],
        max_tokens=30, temperature=temp, top_p=top_p, stream=True,
        extra_body={"top_k": top_k},
    )
    for e in s:
        if e.choices[0].delta.content:
            out += e.choices[0].delta.content
    return out

print("greedy   :", gen(0.0, 1.0, 0))
print("greedy   :", gen(0.0, 1.0, 0), "(should match the line above exactly)")
print("creative :", gen(1.2, 0.95, 50))
print("creative :", gen(1.2, 0.95, 50), "(should differ from the line above)")