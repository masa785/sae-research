import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=DEVICE
)

prompt = "Sparse autoencoders learn interpretable features."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(DEVICE)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden = outputs.hidden_states[-1]

print("\n===== Token Activations =====\n")

for i, token in enumerate(tokens):

    activation = hidden[0, i]

    magnitude = torch.norm(activation).item()

    print(f"Token {i}")
    print(f"Text: {token}")
    print(f"Activation Magnitude: {magnitude:.4f}")
    print("-" * 40)