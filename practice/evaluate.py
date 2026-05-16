import torch

from data.model import SparseAutoencoder

# =====================================
# Settings
# =====================================

DATA_PATH = "activation_dataset.pt"
MODEL_PATH = "sae.pt"

INPUT_DIM = 1536
LATENT_DIM = 4096

DEVICE = "mps"

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

data = torch.load(DATA_PATH)

activations = data["activations"].float()
tokens = data["tokens"]

print("Dataset shape:")
print(activations.shape)

# =====================================
# Load SAE
# =====================================

print("\nLoading SAE model...")

model = SparseAutoencoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM
).to(DEVICE).float()

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

print("Model loaded!")

# =====================================
# Forward Pass
# =====================================

x = activations.to(DEVICE)

with torch.no_grad():

    x_hat, z = model(x)

z = z.cpu()

# =====================================
# Sparsity Statistics
# =====================================

print("\n===== Sparsity Statistics =====")

# 非ゼロfeature数
nonzero = (z > 0).float()

active_counts = nonzero.sum(dim=1)

print("\nAverage active features:")
print(active_counts.mean().item())

print("\nMin active features:")
print(active_counts.min().item())

print("\nMax active features:")
print(active_counts.max().item())

# =====================================
# Top Activating Feature
# =====================================

print("\n===== Top Activating Features =====")

feature_strength = z.mean(dim=0)

topk = torch.topk(feature_strength, k=10)

for rank, (idx, value) in enumerate(
    zip(topk.indices, topk.values)
):

    print(f"\nRank {rank+1}")

    print(f"Feature ID: {idx.item()}")

    print(f"Mean Activation: {value.item():.4f}")

# =====================================
# Token-wise Strongest Feature
# =====================================

print("\n===== Token-wise Strongest Feature =====")

for i in range(min(15, len(tokens))):

    token = tokens[i]

    token_z = z[i]

    top_feature = torch.argmax(token_z)

    top_value = token_z[top_feature]

    print("\n------------------")

    print(f"Token: {token}")

    print(f"Top Feature: {top_feature.item()}")

    print(f"Activation: {top_value.item():.4f}")

print("\nDone.")