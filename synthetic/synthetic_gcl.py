import numpy as np
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_seed, compute_mcc, create_matrix_figure
from tqdm import tqdm
import wandb
import json
import matplotlib.pyplot as plt
from datetime import datetime

def topk_sparsify(z, topk):
    if topk is None or topk <= 0 or topk >= z.shape[-1]:
        return z
    _, idx = torch.topk(torch.abs(z), k=topk, dim=-1)
    mask = torch.zeros_like(z)
    mask.scatter_(-1, idx, 1.0)
    return z * mask


def generate_orthonormal_U(n_dim, h_dim):
    q, _ = np.linalg.qr(np.random.randn(n_dim, h_dim))
    return q[:, :h_dim].astype(np.float32)


def nonlinear_observation(H, A, nonlinear_scale=0.25):
    """
    Smooth nonlinear observation map f_H(h).
    H: (..., h_dim)
    A: (x_dim, h_dim)
    """
    linear = np.matmul(H, A.T)
    nonlinear = nonlinear_scale * np.tanh(linear)
    return linear + nonlinear


def generate_synthetic_data(
    B: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    U: np.ndarray,
    num_samples: int = 1024,
    noise_type: str = "laplace",
    noise_scale: float = 1.0,
    length: int = 1,
    num_domains: int = 4,
    nonlinear_scale: float = 0.25,
):
    """
    Generate synthetic temporal data for Overcomplete Temporal Causal GCL-SAE.

    True causal coordinates:
        h_t in R^h_dim

    Overcomplete sparse representation:
        z_t = U h_t in R^z_dim, z_dim >> h_dim

    Observation:
        x_t = f_H(h_t)

    Dynamics:
        (I - M) h_t = B h_{t-1} + eps_t

    Noise:
        eps_t depends on auxiliary variable u_t.
    """
    h_dim = B.shape[1]
    z_dim = U.shape[0]

    # Domain-dependent Laplace location/scale.
    domain_mu = np.linspace(-0.3, 0.3, num_domains).reshape(num_domains, 1)
    domain_mu = np.repeat(domain_mu, h_dim, axis=1).astype(np.float32)
    domain_scale = np.linspace(0.6, 1.4, num_domains).reshape(num_domains, 1)
    domain_scale = np.repeat(domain_scale, h_dim, axis=1).astype(np.float32)

    h_0 = np.random.uniform(0, 1, (num_samples, h_dim)).astype(np.float32)
    H = [h_0]
    U_labels = []

    h_l = h_0
    I_M_inv = np.linalg.inv(np.eye(h_dim, dtype=np.float32) - M.astype(np.float32))

    eps_last = None
    u_last = None

    for t in range(length):
        u_t = np.random.randint(0, num_domains, size=(num_samples,))
        mu_t = domain_mu[u_t]
        scale_t = domain_scale[u_t] * noise_scale

        if noise_type == "normal":
            noise = np.random.normal(mu_t, scale_t).astype(np.float32)
        elif noise_type == "laplace":
            noise = np.random.laplace(mu_t, scale_t).astype(np.float32)
        else:
            raise ValueError("Unsupported noise type. Choose 'normal' or 'laplace'.")

        h_hist = np.dot(h_l, B.T)
        h_t = np.dot(h_hist + noise, I_M_inv.T).astype(np.float32)

        H.append(h_t)
        U_labels.append(u_t)
        h_l = h_t

        eps_last = noise
        u_last = u_t

    H = np.array(H).transpose((1, 0, 2)).astype(np.float32)       # [batch, length+1, h_dim]
    Z = np.matmul(H, U.T).astype(np.float32)                      # [batch, length+1, z_dim]
    X = nonlinear_observation(H, A, nonlinear_scale).astype(np.float32)

    return {
        "H": H,
        "Z": Z,
        "X": X,
        "u": u_last.astype(np.int64),
        "eps": eps_last.astype(np.float32),
    }

# %%
# Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class OvercompleteTemporalCausalGCLSAE(nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim,
        U,
        num_domains,
        hidden_dim=64,
        topk=3,
        M_mask="off_diag_perm",
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.topk = topk
        self.M_mask = M_mask

        # In this first synthetic experiment, U is fixed to the true subspace basis.
        # Later, this can be made learnable with an orthogonality penalty.
        self.register_buffer("U", torch.tensor(U, dtype=torch.float32))

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

        self.B = nn.Parameter(torch.randn(h_dim, h_dim))
        self.M = nn.Parameter(torch.randn(h_dim, h_dim))

        self.u_embedding = nn.Embedding(num_domains, hidden_dim)
        self.gcl_param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * h_dim),
        )

    def encode_z(self, x):
        z = self.encoder(x)
        z = topk_sparsify(z, self.topk)
        return z

    def z_to_h(self, z):
        # U has shape [z_dim, h_dim], columns orthonormal.
        # h = U^T z.
        return torch.matmul(z, self.U)

    def encode(self, x):
        z_hat = self.encode_z(x)
        h_hat = self.z_to_h(z_hat)
        return z_hat, h_hat

    def decode(self, h):
        return self.decoder(h)

    def get_M(self):
        if self.M_mask == "tril":
            return torch.tril(self.M, diagonal=-1)
        elif self.M_mask == "off_diag":
            return self.M * (1 - torch.eye(self.h_dim, device=self.M.device))
        elif self.M_mask == "off_diag_perm":
            no_self_loop_M = self.M * (1 - torch.eye(self.h_dim, device=self.M.device))
            _, permutation = torch.sort(no_self_loop_M.abs().sum(dim=1))
            perm_tril = torch.tril(no_self_loop_M[permutation][:, permutation])
            inverse_permutation = torch.zeros_like(permutation)
            inverse_permutation[permutation] = torch.arange(self.h_dim, device=self.M.device)
            inv_perm_tril = perm_tril[inverse_permutation][:, inverse_permutation]
            return inv_perm_tril
        raise ValueError("Invalid M_mask value. Choose 'tril', 'off_diag', or 'off_diag_perm'.")

    def estimate_prior(self, h):
        h_t0 = h[:, 0, :]
        h_t1 = h[:, 1, :]

        I_M = torch.eye(self.h_dim, device=self.M.device) - self.get_M()
        h_t1_I_M_T = torch.matmul(h_t1, I_M.T)
        eps = h_t1_I_M_T - torch.matmul(h_t0, self.B.T)
        return eps

    def gcl_score(self, eps, u):
        """
        r(eps, u) = sum_i a_i(u) eps_i + b_i(u) |eps_i|

        This matches the fixed sufficient-statistic version:
            T_i(eps_i) = (eps_i, |eps_i|)
        """
        emb = self.u_embedding(u)
        params = self.gcl_param_net(emb)
        a, b = params[:, :self.h_dim], params[:, self.h_dim:]
        return torch.sum(a * eps + b * torch.abs(eps), dim=-1)

    def gcl_loss(self, eps, u):
        pos_score = self.gcl_score(eps, u)

        perm = torch.randperm(u.shape[0], device=u.device)
        u_neg = u[perm]
        neg_score = self.gcl_score(eps, u_neg)

        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def forward(self, x, u):
        batch_size, period, x_dim = x.shape

        z_hat_list = []
        h_hat_list = []
        x_hat_list = []

        for t in range(period):
            z_hat_t, h_hat_t = self.encode(x[:, t, :])
            x_hat_t = self.decode(h_hat_t)

            z_hat_list.append(z_hat_t)
            h_hat_list.append(h_hat_t)
            x_hat_list.append(x_hat_t)

        z_hat = torch.stack(z_hat_list, dim=1)
        h_hat = torch.stack(h_hat_list, dim=1)
        x_hat = torch.stack(x_hat_list, dim=1)

        eps_hat = self.estimate_prior(h_hat)
        gcl_loss = self.gcl_loss(eps_hat, u)

        return {
            "x_hat": x_hat,
            "z_hat": z_hat,
            "h_hat": h_hat,
            "eps_hat": eps_hat,
            "gcl_loss": gcl_loss,
        }

# %%

x_dim = 3
h_dim = 3
z_dim = 12
length = 1
w_inst = 0.2
lr = 8e-3
wd = 6e-4
total_steps = 50_000
batch_size = 1024
noise_type = "laplace"
noise_scale = 1.0
log_interval = 100
M_sparsity_loss_coeff = 1e-5
B_sparsity_loss_coeff = 1e-8
kl_div_coeff = 1.0
gcl_loss_coeff = 0.1
z_sparsity_loss_coeff = 1e-4
seed = 44
M_mask = "off_diag_perm"
topk = 3
hidden_dim = 64
num_domains = 8
nonlinear_scale = 0.25

set_seed(seed)

B = np.array(
    [
        [0.4, 0.6, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)
M = np.array(
    [
        [0, 0, 0],
        [w_inst, 0, 0],
        [0, w_inst, 0],
    ],
    dtype=np.float32,
)
A = special_ortho_group.rvs(3).astype(np.float32)
U = generate_orthonormal_U(z_dim, h_dim)

wandb_run = wandb.init(
    project="overcomplete-temporal-causal-gcl-sae",
    config={
        "x_dim": x_dim,
        "h_dim": h_dim,
        "z_dim": z_dim,
        "lr": lr,
        "wd": wd,
        "B": json.dumps(B.tolist()),
        "M": json.dumps(M.tolist()),
        "A": json.dumps(A.tolist()),
        "U": json.dumps(U.tolist()),
        "length": length,
        "w_inst": w_inst,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "noise_type": noise_type,
        "noise_scale": noise_scale,
        "log_interval": log_interval,
        "kl_div_coeff": kl_div_coeff,
        "gcl_loss_coeff": gcl_loss_coeff,
        "z_sparsity_loss_coeff": z_sparsity_loss_coeff,
        "M_sparsity_loss_coeff": M_sparsity_loss_coeff,
        "B_sparsity_loss_coeff": B_sparsity_loss_coeff,
        "M_mask": M_mask,
        "topk": topk,
        "hidden_dim": hidden_dim,
        "num_domains": num_domains,
        "nonlinear_scale": nonlinear_scale,
        "seed": seed,
    },
)

model = OvercompleteTemporalCausalGCLSAE(
    x_dim=x_dim,
    h_dim=h_dim,
    z_dim=z_dim,
    U=U,
    num_domains=num_domains,
    hidden_dim=hidden_dim,
    topk=topk,
    M_mask=M_mask,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

with tqdm(total=total_steps) as pbar:
    for step in range(total_steps):
        try:
            batch = generate_synthetic_data(
                B=B,
                M=M,
                A=A,
                U=U,
                num_samples=batch_size,
                noise_type=noise_type,
                noise_scale=noise_scale,
                length=length,
                num_domains=num_domains,
                nonlinear_scale=nonlinear_scale,
            )
            X_batch, H_batch, Z_batch = batch["X"], batch["H"], batch["Z"]
            U_batch = batch["u"]

            optimizer.zero_grad()

            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            U_batch = torch.tensor(U_batch, dtype=torch.long).to(device)

            outputs = model(X_batch, U_batch)

            Z_hat_batch = outputs["z_hat"]
            H_hat_batch = outputs["h_hat"]
            X_hat_batch = outputs["x_hat"]
            eps_hat_batch = outputs["eps_hat"]

            kl_div = torch.abs(eps_hat_batch).mean()
            recon_loss = F.mse_loss(X_hat_batch, X_batch)
            B_sparsity_loss = torch.abs(model.B).sum()
            M_sparsity_loss = torch.abs(model.get_M()).sum()
            z_sparsity_loss = torch.abs(Z_hat_batch).mean()
            gcl_loss = outputs["gcl_loss"]

            loss = recon_loss + \
                kl_div * kl_div_coeff + \
                M_sparsity_loss * M_sparsity_loss_coeff + \
                B_sparsity_loss * B_sparsity_loss_coeff + \
                z_sparsity_loss * z_sparsity_loss_coeff + \
                gcl_loss * gcl_loss_coeff

            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                h_flat = H_batch.reshape(-1, h_dim).T
                h_hat_flat = H_hat_batch.detach().cpu().numpy().reshape(-1, h_dim).T

                mcc_dict = compute_mcc(h_flat, h_hat_flat, dict_size=h_dim, return_dict=True)
                mcc = mcc_dict["mcc"]
                cc = mcc_dict["cc"]

                wandb_run.log({
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_div": kl_div.item(),
                    "gcl_loss": gcl_loss.item(),
                    "z_sparsity_loss": z_sparsity_loss.item(),
                    "B_sparsity_loss": B_sparsity_loss.item(),
                    "M_sparsity_loss": M_sparsity_loss.item(),
                    "mcc": mcc,
                }, step=step)

                B_est = model.B.detach().cpu().numpy()
                B_est_permutated = B_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"], :]

                M_est = model.get_M().detach().cpu().numpy()
                M_est_permutated = M_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"], :]

                B_err = np.abs(np.abs(B_est_permutated) - B).sum()
                M_err = np.abs(np.abs(M_est_permutated) - M).sum()

                wandb_run.log({
                    "B_err": B_err,
                    "M_err": M_err,
                    "err": B_err + M_err,
                }, step=step)

                if step % (log_interval * 10) == 0:
                    B_fig = create_matrix_figure(
                        B_est,
                        title="B matrix",
                        vmin=0,
                        vmax=1,
                    )
                    B_fig_permutated = create_matrix_figure(
                        B_est_permutated,
                        title="B matrix (permutated)",
                        vmin=0,
                        vmax=1,
                    )

                    M_fig = create_matrix_figure(
                        M_est,
                        title="M matrix",
                        vmin=0,
                        vmax=1,
                    )

                    M_fig_permutated = create_matrix_figure(
                        M_est_permutated,
                        title="M matrix (permutated)",
                        vmin=0,
                        vmax=1,
                    )

                    CC_fig = create_matrix_figure(
                        cc,
                        title="CC matrix",
                        vmin=0,
                        vmax=1,
                    )

                    wandb_run.log({"valid/B_matrix": wandb.Image(B_fig)}, step=step)
                    wandb_run.log({"valid/B_matrix_permutated": wandb.Image(B_fig_permutated)}, step=step)
                    wandb_run.log({"valid/CC_matrix": wandb.Image(CC_fig)}, step=step)
                    wandb_run.log({"valid/M_matrix": wandb.Image(M_fig)}, step=step)
                    wandb_run.log({"valid/M_matrix_permutated": wandb.Image(M_fig_permutated)}, step=step)

                    plt.close(B_fig)
                    plt.close(CC_fig)
                    plt.close(M_fig)
                    plt.close(B_fig_permutated)
                    plt.close(M_fig_permutated)

                pbar.set_postfix({"MCC": mcc, "Loss": loss.item()})

            pbar.update(1)

        except Exception as e:
            print(f"An error occurred: {e}")
            wandb_run.finish()
            break

wandb_run.finish()