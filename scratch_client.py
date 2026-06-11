from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="llama-3.2-3b",
    messages=[{"role": "user", "content": "Say hi in one sentence."}],
    max_tokens=40,
)
print(resp.choices[0].message.content)

print("--- streaming ---")
stream = client.chat.completions.create(
    model="llama-3.2-3b",
    messages=[{"role": "user", "content": "Count to five."}],
    max_tokens=40,
    stream=True,
)
for event in stream:
    piece = event.choices[0].delta.content
    if piece:
        print(piece, end="", flush=True)
print()