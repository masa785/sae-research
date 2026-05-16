import torch
import torch.nn.functional as F


def sae_loss(
    x,
    x_hat,
    z,
    lambda_sparse=1e-3
):

    # Reconstruction loss
    recon_loss = F.mse_loss(
        x_hat,
        x
    )

    # Sparsity loss
    sparse_loss = torch.mean(
        torch.abs(z)
    )

    # Total
    total_loss = (
        recon_loss
        + lambda_sparse * sparse_loss
    )

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "sparse_loss": sparse_loss
    }