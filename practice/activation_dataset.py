import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================
# Settings
# =====================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"

# どのlayerを使うか
TARGET_LAYER = 20

# 保存先
SAVE_PATH = "activation_dataset.pt"

# =====================================
# Prompts
# =====================================

prompts = [
    "Sparse autoencoders learn interpretable features.",
    "Transformers process tokens using attention.",
    "Neural networks contain distributed representations.",
    "Polysemantic neurons activate for multiple concepts.",
    "Mechanistic interpretability studies internal computations.",
    "Large language models generate text autoregressively.",
    "Attention heads can specialize in different behaviors.",
    "Hidden states encode contextual information.",
    "Feature superposition occurs in compressed representations.",
    "Sparse features may improve interpretability."
]

# =====================================
# Load Model
# =====================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=DEVICE
)

print("Model loaded!")

# =====================================
# Collect Activations
# =====================================

all_activations = []
all_tokens = []

for idx, prompt in enumerate(prompts):

    print(f"\n[{idx+1}/{len(prompts)}]")
    print(f"Prompt: {prompt}")

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    # Select target layer
    hidden = outputs.hidden_states[TARGET_LAYER]

    # hidden shape:
    # [batch, tokens, hidden_dim]

    hidden = hidden[0]

    # Move to CPU
    hidden = hidden.cpu()

    # Save token-wise activations
    for token, activation in zip(tokens, hidden):

        all_tokens.append(token)
        all_activations.append(activation)

# =====================================
# Stack Activations
# =====================================

activation_tensor = torch.stack(all_activations)

print("\n===== Dataset Info =====")

print("Activation tensor shape:")
print(activation_tensor.shape)

print("\nExample token:")
print(all_tokens[0])

print("\nExample activation:")
print(activation_tensor[0][:10])

# =====================================
# Save Dataset
# =====================================

save_data = {
    "layer": TARGET_LAYER,
    "tokens": all_tokens,
    "activations": activation_tensor
}

torch.save(save_data, SAVE_PATH)

print(f"\nSaved dataset to: {SAVE_PATH}")

# =====================================
# Load Check
# =====================================

loaded = torch.load(SAVE_PATH)

print("\n===== Load Check =====")

print("Layer:")
print(loaded["layer"])

print("\nDataset shape:")
print(loaded["activations"].shape)

print("\nFirst token:")
print(loaded["tokens"][0])

print("\nDone.")