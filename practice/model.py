import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim=1536,
        latent_dim=4096
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Linear(
            input_dim,
            latent_dim
        )

        # Decoder
        self.decoder = nn.Linear(
            latent_dim,
            input_dim
        )

    def forward(self, x):

        # Sparse latent
        z = F.relu(
            self.encoder(x)
        )

        # Reconstruction
        x_hat = self.decoder(z)

        return x_hat, z