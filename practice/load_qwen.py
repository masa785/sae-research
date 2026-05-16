import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#モデルはQwen2.5-1.5B-Instructを使用します。
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="mps"
)

print("Model loaded!")

prompt = "Explain sparse autoencoders simply."

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nResponse:")
print(response)