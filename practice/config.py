from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

SYSTEM_PROMPT = """
You are an expert researcher in:
- Sparse Autoencoders (SAE)
- LLM interpretability
- Transformer architectures
- Mechanistic interpretability
- Deep learning research

Give technically accurate and concise explanations.
"""

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    stream = client.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        stream=True
    )

    print("\nQwen: ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            print(delta, end="", flush=True)

    print("\n")