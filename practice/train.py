import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from data.model import SparseAutoencoder
from data.loss import sae_loss

# =====================================
# Settings
# =====================================

DATA_PATH = "activation_dataset.pt"

INPUT_DIM = 1536
LATENT_DIM = 4096

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3

DEVICE = "mps"

SAVE_PATH = "sae.pt"

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

data = torch.load(DATA_PATH)

activations = data["activations"].float()

print("Dataset shape:")
print(activations.shape)

dataset = TensorDataset(activations)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =====================================
# Model
# =====================================

model = SparseAutoencoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM
).to(DEVICE)

optimizer = Adam(
    model.parameters(),
    lr=LR
)

print("\nModel initialized!")

# =====================================
# Training Loop
# =====================================

for epoch in range(EPOCHS):

    total_loss_sum = 0
    recon_loss_sum = 0
    sparse_loss_sum = 0

    for batch in loader:

        x = batch[0].to(DEVICE)

        # Forward
        x_hat, z = model(x)

        # Loss
        losses = sae_loss(
            x,
            x_hat,
            z
        )

        total_loss = losses["total_loss"]

        # Backprop
        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        # Logging
        total_loss_sum += total_loss.item()
        recon_loss_sum += losses["recon_loss"].item()
        sparse_loss_sum += losses["sparse_loss"].item()

    # Epoch Log
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    print(f"Total Loss: {total_loss_sum:.4f}")
    print(f"Recon Loss: {recon_loss_sum:.4f}")
    print(f"Sparse Loss: {sparse_loss_sum:.4f}")

# =====================================
# Save Model
# =====================================

torch.save(
    model.state_dict(),
    SAVE_PATH
)

print(f"\nModel saved to: {SAVE_PATH}")

print("\nTraining complete.")