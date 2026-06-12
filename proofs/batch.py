from mlx_lm import load, batch_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

def encode(text):
    msgs = [{"role": "user", "content": text}]
    # NOTE: no tokenize=False here, batch_generate wants token IDs
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

prompts = [
    encode("Say hello in one sentence."),
    encode("Name a primary color."),
]

resp = batch_generate(model, tokenizer, prompts=prompts, max_tokens=40, verbose=True)

print("\n=== type ===")
print(type(resp))
print("\n=== repr ===")
print(repr(resp))
print("\n=== attributes ===")
print([a for a in dir(resp) if not a.startswith("_")])