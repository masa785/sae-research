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

# ============================================================
# 修正点サマリ（v2: 理論的整合性の修正版）
# ============================================================
# [v1から継続]
# - num_domains: 4 -> 8 (A3条件: rank = h_dim*k+1 = 7 以上)
# - domain_mu_range: 0.3 -> 0.05 (kl_divとの矛盾緩和)
# - 学習後のLiNGAM後処理（診断機能つき）
#
# [v2 新規修正 1] NOTEARS非巡回性制約に切り替え
#   問題: `tril`は学習開始時点で「インデックス順=真の因果順序」を
#   仮定してしまい、この誤った制約のもとでeps_hatが歪む。
#   歪んだeps_hatに後からLiNGAMをかけても手遅れ（今回の実験結果が示す通り）。
#   対策: 学習中は M_mask="off_diag"（順序を仮定しない自由な行列）とし、
#   NOTEARSの微分可能な非巡回性ペナルティ h(M)=tr(exp(M∘M))-h_dim を
#   損失に追加する。因果順序の発見は学習後のLiNGAMに完全に委ねる。
#
# [v2 新規修正 2] アンカーサポート正則化（A5Block条件の明示的な強制）
#   理論のA5Block条件は「Mの各列は他の列と共有されない一意なサポート
#   インデックスを持つ」ことを要求する。これはCorollary 1の
#   成分ごと同定可能性の前提だが、これまでのコードには一切実装
#   されていなかった。exclusivityペナルティとして明示的に追加する。
#   Bについても対応する条件（対角成分の非ゼロ性）を追加する。
#
# [v2 新規修正 3] 損失重みのウォームアップ・スケジューリング
#   問題: GCL損失・NOTEARS損失・アンカー損失は「eps_hatがある程度
#   意味のある残差になっている」ことを前提にしているが、学習初期は
#   z_hat, h_hatがランダムに近くeps_hatも無意味。この状態で重い制約を
#   かけると誤った局所解に早期収束するリスクがある。
#   対策: 再構成損失を先に十分収束させ、その後GCL・NOTEARS・アンカー
#   損失を線形にウォームアップして導入する。
# ============================================================


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
    num_domains: int = 8,
    nonlinear_scale: float = 0.25,
    domain_mu_range: float = 0.05,
):
    h_dim = B.shape[1]
    z_dim = U.shape[0]

    required_domains = h_dim * 2 + 1
    if num_domains < required_domains:
        print(
            f"[WARNING] num_domains={num_domains} は A3条件 "
            f"(rank = h_dim*k+1 = {required_domains}) を満たしていません。"
        )

    domain_mu = np.linspace(-domain_mu_range, domain_mu_range, num_domains).reshape(num_domains, 1)
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

    H = np.array(H).transpose((1, 0, 2)).astype(np.float32)
    Z = np.matmul(H, U.T).astype(np.float32)
    X = nonlinear_observation(H, A, nonlinear_scale).astype(np.float32)

    return {
        "H": H, "Z": Z, "X": X,
        "u": u_last.astype(np.int64),
        "eps": eps_last.astype(np.float32),
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# [v2修正2] アンカーサポート正則化・DAG制約関数
# ============================================================

def notears_acyclicity_penalty(M):
    """
    NOTEARS非巡回性ペナルティ (Zheng et al., 2018)
    h(M) = tr(exp(M∘M)) - h_dim
    h(M) = 0 のときのみ M はDAG（非巡回）である。
    M∘M（要素ごと二乗）を使うことで符号に依存せず、常に非負の
    「エッジの存在強度」として扱える。
    """
    h_dim = M.shape[0]
    M_squared = M * M
    exp_M = torch.matrix_exp(M_squared)
    return torch.trace(exp_M) - h_dim


def anchor_support_loss(M, margin=0.1):
    """
    A5Block条件（アンカーサポート）の明示的な正則化。

    「Mの各列 l は、他の列と共有されない一意なサポートインデックス k_l
    を持つ」という理論の仮定を、以下のexclusivityマージン損失で促す。

    exclusivity[i, l] = |M[i,l]| - sum_{l' != l} |M[i,l']|

    各列 l について、少なくとも1つの行 i で exclusivity が margin を
    超えることを要求する（ヒンジ損失）。
    """
    h_dim = M.shape[0]
    abs_M = M.abs()
    row_sum = abs_M.sum(dim=1, keepdim=True)          # (h_dim, 1)
    other_sum = row_sum - abs_M                         # (h_dim, h_dim)
    exclusivity = abs_M - other_sum                      # (h_dim, h_dim)
    max_exclusivity_per_col = exclusivity.max(dim=0).values  # (h_dim,)
    return F.relu(margin - max_exclusivity_per_col).mean()


def diagonal_persistence_loss(B, margin=0.1):
    """
    Bにおけるアンカー条件の対応物。
    Corollary 1のB版は「B_tauの対角成分が非ゼロであること」を要求する
    （論文Discussion: "the counterpart of Corollary 1 requires nonzero
    diagonal entries in B_tau"）。
    L1スパース正則化がB全体にかかるため、対角成分まで潰れてしまう
    ことを防ぐために追加する。
    """
    diag = torch.diagonal(B).abs()
    return F.relu(margin - diag).mean()


# ============================================================
# [v2修正3] 損失重みウォームアップ・スケジューラ
# ============================================================

class LossWeightScheduler:
    """
    再構成損失を先に安定させてから、GCL・NOTEARS・アンカー損失を
    線形にウォームアップして導入するスケジューラ。

    問題の背景: 学習初期はz_hat, h_hatがほぼランダムでeps_hatも
    無意味な値になる。この状態でGCL損失やNOTEARS制約を強くかけると、
    意味のない残差構造に対して誤った因果順序やDAG構造を「学習」して
    しまい、局所解に固まるリスクがある。
    """

    def __init__(self, target_coeffs: dict, warmup_start_step: dict, warmup_end_step: dict):
        self.target_coeffs = target_coeffs
        self.warmup_start = warmup_start_step
        self.warmup_end = warmup_end_step

    def get(self, name: str, step: int) -> float:
        target = self.target_coeffs[name]
        start = self.warmup_start.get(name, 0)
        end = self.warmup_end.get(name, 0)

        if step < start:
            return 0.0
        if end <= start:
            return target
        frac = min(1.0, (step - start) / (end - start))
        return target * frac


# ============================================================
# モデル定義
# ============================================================

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
        M_mask="off_diag",   # [v2修正1] tril/off_diag_perm ではなく off_diag をデフォルトに
        M_scale=2.0,          # [v2修正1補足] NOTEARS安定化のためのMの値域制限
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.topk = topk
        self.M_mask = M_mask
        self.M_scale = M_scale

        self.register_buffer("U", torch.tensor(U, dtype=torch.float32))

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

        self.B = nn.Parameter(torch.randn(h_dim, h_dim) * 0.1)
        self.M = nn.Parameter(torch.randn(h_dim, h_dim) * 0.1)

        self.u_embedding = nn.Embedding(num_domains, hidden_dim)
        self.gcl_param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * h_dim),
        )

    def encode_z(self, x):
        z = self.encoder(x)
        z = topk_sparsify(z, self.topk)
        return z

    def z_to_h(self, z):
        return torch.matmul(z, self.U)

    def encode(self, x):
        z_hat = self.encode_z(x)
        h_hat = self.z_to_h(z_hat)
        return z_hat, h_hat

    def decode(self, h):
        return self.decoder(h)

    def get_M(self):
        """
        [v2修正1] デフォルトはoff_diag（順序を仮定しない自由な行列）。
        NOTEARSペナルティの数値安定性のため、tanhでMの値域を
        [-M_scale, M_scale] に制限する。
        tril/off_diag_permはレガシーオプションとして残すが、
        NOTEARS運用時はoff_diag推奨。
        """
        M_bounded = torch.tanh(self.M) * self.M_scale
        eye = torch.eye(self.h_dim, device=self.M.device)

        if self.M_mask == "off_diag":
            return M_bounded * (1 - eye)
        elif self.M_mask == "tril":
            return torch.tril(M_bounded, diagonal=-1)
        elif self.M_mask == "off_diag_perm":
            no_self_loop_M = M_bounded * (1 - eye)
            _, permutation = torch.sort(no_self_loop_M.abs().sum(dim=1))
            perm_tril = torch.tril(no_self_loop_M[permutation][:, permutation])
            inverse_permutation = torch.zeros_like(permutation)
            inverse_permutation[permutation] = torch.arange(self.h_dim, device=self.M.device)
            return perm_tril[inverse_permutation][:, inverse_permutation]
        raise ValueError("Invalid M_mask value. Choose 'off_diag' (recommended), 'tril', or 'off_diag_perm'.")

    def estimate_prior(self, h):
        h_t0 = h[:, 0, :]
        h_t1 = h[:, 1, :]
        I_M = torch.eye(self.h_dim, device=self.M.device) - self.get_M()
        h_t1_I_M_T = torch.matmul(h_t1, I_M.T)
        eps = h_t1_I_M_T - torch.matmul(h_t0, self.B.T)
        return eps

    def gcl_score(self, eps, u):
        emb = self.u_embedding(u)
        params = self.gcl_param_net(emb)
        a, b = params[:, :self.h_dim], params[:, self.h_dim:]
        return torch.sum(a * eps + b * torch.abs(eps), dim=-1)

    def gcl_loss(self, eps, u):
        pos_score = self.gcl_score(eps, u)

        # [修正] 自己参照（同ドメイン）を確実に除外する負例サンプリング
        # 単純な randperm は n_domains が小さいほど「負例」のうち
        # 実際は同じドメイン（誤ラベル）である割合が高くなる
        # (n_domains=2で約50%, n_domains=8で約12.5%の誤ラベル)。
        # これがGCL損失が学習しない根本原因だったため、自己参照を
        # 明示的に除外するリトライループを追加する。
        batch_size = u.shape[0]
        u_neg = u.clone()
        still_same = torch.ones(batch_size, dtype=torch.bool, device=u.device)
        max_retries = 10
        for _ in range(max_retries):
            if not still_same.any():
                break
            perm = torch.randperm(batch_size, device=u.device)
            candidate = u[perm]
            update_mask = still_same & (candidate != u)
            u_neg[update_mask] = candidate[update_mask]
            still_same = (u_neg == u)

        neg_score = self.gcl_score(eps, u_neg)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def forward(self, x, u):
        batch_size, period, x_dim = x.shape
        z_hat_list, h_hat_list, x_hat_list = [], [], []

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
            "x_hat": x_hat, "z_hat": z_hat, "h_hat": h_hat,
            "eps_hat": eps_hat, "gcl_loss": gcl_loss,
        }


# ============================================================
# ハイパーパラメータ
# ============================================================

x_dim = 3
h_dim = 3
z_dim = 12
length = 1
w_inst = 0.2
lr = 8e-3
wd = 6e-4
total_steps = 50_000          # [推奨] 50,000 -> 150,000 (前回MCC未収束のため延長)
batch_size = 1024
noise_type = "laplace"
noise_scale = 1.0
log_interval = 100
seed = 44
topk = 3
hidden_dim = 64
num_domains = 8
nonlinear_scale = 0.25
domain_mu_range = 5.0

M_mask = "off_diag"            # [v2修正1] NOTEARS運用のため off_diag に変更
M_scale = 2.0

# --- 損失の目標係数（ウォームアップ後の最終値） ---
target_coeffs = {
    "kl_div": 0.1,              # 1.0 -> 0.1 （ドメイン依存平均との衝突緩和、v1から継続）
    "M_sparsity": 1e-5,
    "B_sparsity": 1e-8,
    "z_sparsity": 1e-4,
    "gcl": 0.5,
    "notears": 1.0,              # [v2新規] NOTEARS非巡回性の最終目標係数
    "anchor_M": 0.3,              # [v2新規] Mのアンカーサポート正則化
    "anchor_B": 0.1,              # [v2新規] Bの対角持続性正則化
    "scale": 0.0,
}

# --- ウォームアップスケジュール（ステップ数）---
# 再構成を先に安定させてから因果構造系の損失を導入する
warmup_start = {
    "kl_div": 0,
    "M_sparsity": 0,
    "B_sparsity": 0,
    "z_sparsity": 0,
    "gcl": 5_000,        # 再構成がある程度進んでから開始
    "notears": 5_000,
    "anchor_M": 10_000,   # NOTEARSがある程度効いてから開始
    "anchor_B": 10_000,
    "scale": 0,
}
warmup_end = {
    "kl_div": 0,
    "M_sparsity": 0,
    "B_sparsity": 0,
    "z_sparsity": 0,
    "gcl": 20_000,
    "notears": 20_000,
    "anchor_M": 30_000,
    "anchor_B": 30_000,
    "scale": 0,
}

anchor_margin = 0.1
diag_margin = 0.1

lingam_eval_batches = 20
lingam_eval_samples = lingam_eval_batches * batch_size
mcc_threshold = 0.9
independence_threshold = 0.15

set_seed(seed)

B = np.array([[0.4, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
M = np.array([[0, 0, 0], [w_inst, 0, 0], [0, w_inst, 0]], dtype=np.float32)
A = special_ortho_group.rvs(3).astype(np.float32)
U = generate_orthonormal_U(z_dim, h_dim)

loss_scheduler = LossWeightScheduler(target_coeffs, warmup_start, warmup_end)

wandb_run = wandb.init(
    project="overcomplete-temporal-causal-gcl-sae",
    config={
        "x_dim": x_dim, "h_dim": h_dim, "z_dim": z_dim, "lr": lr, "wd": wd,
        "B": json.dumps(B.tolist()), "M": json.dumps(M.tolist()),
        "A": json.dumps(A.tolist()), "U": json.dumps(U.tolist()),
        "length": length, "w_inst": w_inst, "batch_size": batch_size,
        "total_steps": total_steps, "noise_type": noise_type,
        "noise_scale": noise_scale, "log_interval": log_interval,
        "target_coeffs": target_coeffs, "warmup_start": warmup_start,
        "warmup_end": warmup_end, "anchor_margin": anchor_margin,
        "diag_margin": diag_margin, "M_mask": M_mask, "M_scale": M_scale,
        "topk": topk, "hidden_dim": hidden_dim, "num_domains": num_domains,
        "nonlinear_scale": nonlinear_scale, "domain_mu_range": domain_mu_range,
        "seed": seed, "fix_version": "v2_notears_anchor_warmup",
    },
)

model = OvercompleteTemporalCausalGCLSAE(
    x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, U=U, num_domains=num_domains,
    hidden_dim=hidden_dim, topk=topk, M_mask=M_mask, M_scale=M_scale,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

with tqdm(total=total_steps) as pbar:
    for step in range(total_steps):
        try:
            batch = generate_synthetic_data(
                B=B, M=M, A=A, U=U, num_samples=batch_size,
                noise_type=noise_type, noise_scale=noise_scale, length=length,
                num_domains=num_domains, nonlinear_scale=nonlinear_scale,
                domain_mu_range=domain_mu_range,
            )
            X_batch, H_batch = batch["X"], batch["H"]
            U_batch = batch["u"]

            optimizer.zero_grad()

            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            U_batch = torch.tensor(U_batch, dtype=torch.long).to(device)

            outputs = model(X_batch, U_batch)
            Z_hat_batch = outputs["z_hat"]
            H_hat_batch = outputs["h_hat"]
            X_hat_batch = outputs["x_hat"]
            eps_hat_batch = outputs["eps_hat"]

            # --- 各損失の計算 ---
            recon_loss = F.mse_loss(X_hat_batch, X_batch)
            kl_div = torch.abs(eps_hat_batch).mean()
            M_current = model.get_M()
            B_sparsity_loss = torch.abs(model.B).sum()
            M_sparsity_loss = torch.abs(M_current).sum()
            z_sparsity_loss = torch.abs(Z_hat_batch).mean()
            gcl_loss = outputs["gcl_loss"]

            # [v2修正1] NOTEARS非巡回性ペナルティ
            notears_loss = notears_acyclicity_penalty(M_current)

            # [v2修正2] アンカーサポート正則化
            anchor_M_loss = anchor_support_loss(M_current, margin=anchor_margin)
            anchor_B_loss = diagonal_persistence_loss(model.B, margin=diag_margin)

            # [追加] スケール正規化ロス (仮定A5Block: 正規化制約)
            # eps_hat_batch は (batch_size, h_dim) なのでそのまま分散を計算
            eps_var = torch.var(eps_hat_batch, dim=0)  # 各次元の分散 (h_dim,)
            scale_loss = torch.mean((eps_var - 1.0) ** 2)

            # [v2修正3] ウォームアップされた係数を取得
            c_kl = loss_scheduler.get("kl_div", step)
            c_Msp = loss_scheduler.get("M_sparsity", step)
            c_Bsp = loss_scheduler.get("B_sparsity", step)
            c_zsp = loss_scheduler.get("z_sparsity", step)
            c_gcl = loss_scheduler.get("gcl", step)
            c_notears = loss_scheduler.get("notears", step)
            c_anchor_M = loss_scheduler.get("anchor_M", step)
            c_anchor_B = loss_scheduler.get("anchor_B", step)
            c_scale = loss_scheduler.get("scale", step)

            loss = (
                recon_loss
                + c_kl * kl_div
                + c_Msp * M_sparsity_loss
                + c_Bsp * B_sparsity_loss
                + c_zsp * z_sparsity_loss
                + c_gcl * gcl_loss
                + c_notears * notears_loss
                + c_anchor_M * anchor_M_loss
                + c_anchor_B * anchor_B_loss
                + c_scale * scale_loss
            )

            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                h_flat = H_batch.reshape(-1, h_dim).T
                h_hat_flat = H_hat_batch.detach().cpu().numpy().reshape(-1, h_dim).T
                mcc_dict = compute_mcc(h_flat, h_hat_flat, dict_size=h_dim, return_dict=True)
                mcc = mcc_dict["mcc"]
                cc = mcc_dict["cc"]

                wandb_run.log({
                    "loss": loss.item(), "recon_loss": recon_loss.item(),
                    "kl_div": kl_div.item(), "gcl_loss": gcl_loss.item(),
                    "notears_loss": notears_loss.item(),
                    "anchor_M_loss": anchor_M_loss.item(),
                    "anchor_B_loss": anchor_B_loss.item(),
                    "scale_loss": scale_loss.item(),
                    "z_sparsity_loss": z_sparsity_loss.item(),
                    "B_sparsity_loss": B_sparsity_loss.item(),
                    "M_sparsity_loss": M_sparsity_loss.item(),
                    "mcc": mcc,
                    "coeff/gcl": c_gcl, "coeff/notears": c_notears,
                    "coeff/anchor_M": c_anchor_M, "coeff/anchor_B": c_anchor_B,
                    "coeff/scale": c_scale,
                }, step=step)

                B_est = model.B.detach().cpu().numpy()
                B_est_permutated = B_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"], :]
                M_est = model.get_M().detach().cpu().numpy()
                M_est_permutated = M_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"], :]

                B_err = np.abs(np.abs(B_est_permutated) - B).sum()
                M_err = np.abs(np.abs(M_est_permutated) - M).sum()
                wandb_run.log({"B_err": B_err, "M_err": M_err, "err": B_err + M_err}, step=step)

                if step % (log_interval * 10) == 0:
                    B_fig = create_matrix_figure(B_est, title="B matrix", vmin=0, vmax=1)
                    B_fig_p = create_matrix_figure(B_est_permutated, title="B matrix (permutated)", vmin=0, vmax=1)
                    M_fig = create_matrix_figure(M_est, title="M matrix", vmin=0, vmax=1)
                    M_fig_p = create_matrix_figure(M_est_permutated, title="M matrix (permutated)", vmin=0, vmax=1)
                    CC_fig = create_matrix_figure(cc, title="CC matrix", vmin=0, vmax=1)

                    wandb_run.log({
                        "valid/B_matrix": wandb.Image(B_fig),
                        "valid/B_matrix_permutated": wandb.Image(B_fig_p),
                        "valid/CC_matrix": wandb.Image(CC_fig),
                        "valid/M_matrix": wandb.Image(M_fig),
                        "valid/M_matrix_permutated": wandb.Image(M_fig_p),
                    }, step=step)

                    plt.close(B_fig); plt.close(CC_fig); plt.close(M_fig)
                    plt.close(B_fig_p); plt.close(M_fig_p)

                pbar.set_postfix({"MCC": mcc, "Loss": loss.item()})

            pbar.update(1)

        except Exception as e:
            print(f"An error occurred: {e}")
            wandb_run.finish()
            raise


# ============================================================
# 学習後のLiNGAM後処理
# ============================================================
# ★★★ 重要な設計修正 ★★★
# 【誤り】eps_hat（残差ノイズ）にLiNGAMをかけていた。
#   理論のA1条件が明示する通り、eps_tは定義上「成分間が互いに独立」
#   なノイズであり、M行列による因果構造はeps_tには一切含まれない。
#   オラクルテスト(test1_oracle_lingam.py)で、真のepsにLiNGAMを
#   かけると隣接行列が全てゼロになる（=因果構造が見つからない）
#   ことが確認された。
#
# 【正しい設計】h_hat（潜在変数そのもの）にLiNGAMをかける。
#   (I - M) h_t = B h_{t-1} + eps_t
#   ⇔ h_t = M h_t + (B h_{t-1} + eps_t)
#   これはLiNGAMの標準形 X = BX + e と一致する（X=h_t）。
#   同一オラクルテストで、真のh_tにLiNGAMをかけると因果順序
#   [0,1,2]が正確に復元されることを確認済み。
# ============================================================
print("\n" + "=" * 60)
print("Post-processing: LiNGAM causal order estimation")
print("=" * 60)

try:
    from lingam import DirectLiNGAM
except ImportError:
    print("[ERROR] `pip install lingam` を実行してください。")
    DirectLiNGAM = None

if DirectLiNGAM is not None:
    model.eval()
    h_hat_collected = []       # [修正] eps_collected -> h_hat_collected
    eps_collected_diag = []    # 独立性診断用にeps_hatも別途保持
    with torch.no_grad():
        for _ in range(lingam_eval_batches):
            eval_batch = generate_synthetic_data(
                B=B, M=M, A=A, U=U, num_samples=batch_size,
                noise_type=noise_type, noise_scale=noise_scale, length=length,
                num_domains=num_domains, nonlinear_scale=nonlinear_scale,
                domain_mu_range=domain_mu_range,
            )
            X_eval = torch.tensor(eval_batch["X"], dtype=torch.float32).to(device)
            U_eval = torch.tensor(eval_batch["u"], dtype=torch.long).to(device)
            eval_outputs = model(X_eval, U_eval)
            # [修正] LiNGAMにかけるのは h_hat の t=1 時点（構造方程式が
            # 適用された後のh）。h_hatは (batch, period, h_dim) なので
            # t=1（period内の2番目、length=1なので最後の時刻）を使う。
            h_hat_collected.append(eval_outputs["h_hat"][:, 1, :].cpu().numpy())
            eps_collected_diag.append(eval_outputs["eps_hat"].cpu().numpy())

    h_hat_all = np.concatenate(h_hat_collected, axis=0)
    eps_all_diag = np.concatenate(eps_collected_diag, axis=0)
    print(f"Collected h_hat for LiNGAM: shape={h_hat_all.shape}")

    # 独立性診断は引き続きeps_hatに対して行う（こちらは正しい用途）
    corr_matrix = np.corrcoef(eps_all_diag.T)
    off_diag_corr = np.abs(corr_matrix - np.eye(h_dim)).sum() / (h_dim * (h_dim - 1))
    print(f"\n[診断] 推定ノイズ(eps_hat)間の平均絶対相関（独立性チェック用）: {off_diag_corr:.4f}")
    lingam_reliable = off_diag_corr < independence_threshold
    if not lingam_reliable:
        print(f"[WARNING] ノイズ独立性が閾値{independence_threshold}を超過。エンコーダの学習が不十分な可能性。")

    # [修正] LiNGAMはh_hatに適用する
    lingam_model = DirectLiNGAM()
    lingam_model.fit(h_hat_all)
    causal_order = lingam_model.causal_order_
    print(f"Estimated causal order (LiNGAM on h_hat): {causal_order}")
    print(f"LiNGAM推定隣接行列（M_hatの直接推定値、参考）:\n{lingam_model.adjacency_matrix_}")

    B_est_raw = model.B.detach().cpu().numpy()
    M_est_raw = model.get_M().detach().cpu().numpy()
    B_lingam_corrected = B_est_raw[causal_order][:, causal_order]
    M_lingam_corrected = M_est_raw[causal_order][:, causal_order]

    print(f"\nTrue B:\n{B}")
    print(f"LiNGAM-corrected B (abs):\n{np.abs(B_lingam_corrected)}")
    B_err_lingam = np.abs(np.abs(B_lingam_corrected) - B).sum()
    print(f"B error (LiNGAM corrected): {B_err_lingam:.4f}")

    print(f"\nTrue M:\n{M}")
    print(f"LiNGAM-corrected M (abs):\n{np.abs(M_lingam_corrected)}")
    M_err_lingam = np.abs(np.abs(M_lingam_corrected) - M).sum()
    print(f"M error (LiNGAM corrected): {M_err_lingam:.4f}")

    eval_batch_final = generate_synthetic_data(
        B=B, M=M, A=A, U=U, num_samples=5000,
        noise_type=noise_type, noise_scale=noise_scale, length=length,
        num_domains=num_domains, nonlinear_scale=nonlinear_scale,
        domain_mu_range=domain_mu_range,
    )
    X_final = torch.tensor(eval_batch_final["X"], dtype=torch.float32).to(device)
    U_final = torch.tensor(eval_batch_final["u"], dtype=torch.long).to(device)
    with torch.no_grad():
        final_outputs = model(X_final, U_final)
    H_final = eval_batch_final["H"]
    H_hat_final = final_outputs["h_hat"].cpu().numpy()

    h_flat_final = H_final.reshape(-1, h_dim).T
    h_hat_flat_final = H_hat_final.reshape(-1, h_dim).T
    mcc_dict_final = compute_mcc(h_flat_final, h_hat_flat_final, dict_size=h_dim, return_dict=True)
    print(f"\nFinal MCC: {mcc_dict_final['mcc']:.4f}")

    B_mcc_corrected = B_est_raw[:, mcc_dict_final["col_ind"]][mcc_dict_final["col_ind"], :]
    M_mcc_corrected = M_est_raw[:, mcc_dict_final["col_ind"]][mcc_dict_final["col_ind"], :]
    B_err_mcc = np.abs(np.abs(B_mcc_corrected) - B).sum()
    M_err_mcc = np.abs(np.abs(M_mcc_corrected) - M).sum()

    print("\n" + "-" * 60)
    print("比較: MCCベース置換補正 vs LiNGAMベース置換補正")
    print("-" * 60)
    print(f"{'':20s} {'B_err':>10s} {'M_err':>10s} {'合計':>10s}")
    print(f"{'MCCベース':20s} {B_err_mcc:>10.4f} {M_err_mcc:>10.4f} {B_err_mcc+M_err_mcc:>10.4f}")
    print(f"{'LiNGAMベース':20s} {B_err_lingam:>10.4f} {M_err_lingam:>10.4f} {B_err_lingam+M_err_lingam:>10.4f}")

    mcc_reliable = mcc_dict_final["mcc"] >= mcc_threshold
    print("\n" + "-" * 60)
    print("推奨判定")
    print("-" * 60)
    print(f"MCC = {mcc_dict_final['mcc']:.4f} (閾値{mcc_threshold}: {'達成' if mcc_reliable else '未達成'})")
    print(f"独立性 off_diag_corr = {off_diag_corr:.4f} (閾値{independence_threshold}: {'良好' if lingam_reliable else '不十分'})")

    if mcc_reliable and lingam_reliable:
        recommended = "LiNGAMベース"
        recommended_B, recommended_M = B_lingam_corrected, M_lingam_corrected
    else:
        recommended = "MCCベース"
        recommended_B, recommended_M = B_mcc_corrected, M_mcc_corrected
        print("[提案] 指標未達成のため、MCCベースの補正を採用することを推奨します。")

    print(f"\n>>> 推奨する補正方法: {recommended}")

    B_lingam_fig = create_matrix_figure(np.abs(B_lingam_corrected), title="B matrix (LiNGAM corrected)", vmin=0, vmax=1)
    M_lingam_fig = create_matrix_figure(np.abs(M_lingam_corrected), title="M matrix (LiNGAM corrected)", vmin=0, vmax=1)

    wandb_run_post = wandb.init(
        project="overcomplete-temporal-causal-gcl-sae",
        name=f"lingam_postprocess_{wandb_run.id}",
        config={"parent_run": wandb_run.id, "causal_order": [int(x) for x in causal_order]},
    )
    wandb_run_post.log({
        "postprocess/B_matrix_lingam_corrected": wandb.Image(B_lingam_fig),
        "postprocess/M_matrix_lingam_corrected": wandb.Image(M_lingam_fig),
        "postprocess/noise_independence_off_diag_corr": off_diag_corr,
        "postprocess/lingam_reliable": lingam_reliable,
        "postprocess/mcc_reliable": mcc_reliable,
        "postprocess/recommended_method": recommended,
        "postprocess/B_err_lingam": B_err_lingam,
        "postprocess/M_err_lingam": M_err_lingam,
        "postprocess/B_err_mcc": B_err_mcc,
        "postprocess/M_err_mcc": M_err_mcc,
        "postprocess/final_mcc": mcc_dict_final["mcc"],
    })
    wandb_run_post.finish()
    plt.close(B_lingam_fig); plt.close(M_lingam_fig)
    print("\nLiNGAM後処理が完了しました。")

wandb_run.finish()