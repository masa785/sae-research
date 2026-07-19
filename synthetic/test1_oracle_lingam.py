"""
テスト1: オラクルテスト（真のh_tにLiNGAM）※重要な設計修正あり
============================================================
【重要】当初の設計ミスについて:
当初、eps（残差ノイズ）に直接LiNGAMをかける設計にしていましたが、
これは理論的に誤りでした。

理論のA1条件が明示する通り、eps_tは定義上「成分間が互いに独立」
なノイズです。M行列による因果構造は、eps_tが生成される前の
h_t自体の中に埋め込まれています。eps_tには最初から見つけるべき
因果構造が存在しないため、LiNGAMをかけても意味がありません
（隣接行列が全てゼロになるのはこのため）。

正しい設計:
    (I - M) h_t = B h_{t-1} + eps_t
    ⇔ h_t = M h_t + (B h_{t-1} + eps_t)
    ⇔ h_t = M h_t + y_t   (y_t := B h_{t-1} + eps_t を "外生項" とみなす)

これはLiNGAMの標準形 X = BX + e と完全に一致します（X=h_t, e=y_t）。
したがってLiNGAMは h_t のサンプル集合（多数の独立試行）に対して
直接適用すべきです。

実行方法:
    python test1_oracle_lingam.py
============================================================
"""

import numpy as np
from lingam import DirectLiNGAM
from scipy.stats import special_ortho_group


def generate_orthonormal_U(n_dim, h_dim):
    q, _ = np.linalg.qr(np.random.randn(n_dim, h_dim))
    return q[:, :h_dim].astype(np.float32)


def nonlinear_observation(H, A, nonlinear_scale=0.25):
    linear = np.matmul(H, A.T)
    nonlinear = nonlinear_scale * np.tanh(linear)
    return linear + nonlinear


def generate_synthetic_data(B, M, A, U, num_samples=1024, noise_type="laplace",
                              noise_scale=1.0, length=1, num_domains=8,
                              nonlinear_scale=0.25, domain_mu_range=0.05):
    h_dim = B.shape[1]
    domain_mu = np.linspace(-domain_mu_range, domain_mu_range, num_domains).reshape(num_domains, 1)
    domain_mu = np.repeat(domain_mu, h_dim, axis=1).astype(np.float32)
    domain_scale = np.linspace(0.6, 1.4, num_domains).reshape(num_domains, 1)
    domain_scale = np.repeat(domain_scale, h_dim, axis=1).astype(np.float32)

    h_0 = np.random.uniform(0, 1, (num_samples, h_dim)).astype(np.float32)
    H = [h_0]
    h_l = h_0
    I_M_inv = np.linalg.inv(np.eye(h_dim, dtype=np.float32) - M.astype(np.float32))

    eps_last, u_last = None, None
    for t in range(length):
        u_t = np.random.randint(0, num_domains, size=(num_samples,))
        mu_t = domain_mu[u_t]
        scale_t = domain_scale[u_t] * noise_scale
        noise = np.random.laplace(mu_t, scale_t).astype(np.float32)
        h_hist = np.dot(h_l, B.T)
        h_t = np.dot(h_hist + noise, I_M_inv.T).astype(np.float32)
        H.append(h_t)
        h_l = h_t
        eps_last, u_last = noise, u_t

    H = np.array(H).transpose((1, 0, 2)).astype(np.float32)
    Z = np.matmul(H, U.T).astype(np.float32)
    X = nonlinear_observation(H, A, nonlinear_scale).astype(np.float32)

    return {"H": H, "Z": Z, "X": X, "u": u_last.astype(np.int64), "eps": eps_last.astype(np.float32)}


if __name__ == "__main__":
    np.random.seed(44)

    w_inst = 0.2
    h_dim = 3
    z_dim = 12
    num_domains = 8
    domain_mu_range = 0.05

    B = np.array([[0.4, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    M = np.array([[0, 0, 0], [w_inst, 0, 0], [0, w_inst, 0]], dtype=np.float32)
    A = special_ortho_group.rvs(3).astype(np.float32)
    U = generate_orthonormal_U(z_dim, h_dim)

    print("=" * 60)
    print("テスト1(修正版): 真のh_tにLiNGAM")
    print("=" * 60)
    print(f"\n真のB:\n{B}")
    print(f"\n真のM:\n{M}")
    print("\n期待される因果順序: [0, 1, 2]")

    n_samples = 50_000
    true_data = generate_synthetic_data(
        B=B, M=M, A=A, U=U, num_samples=n_samples,
        noise_type="laplace", noise_scale=1.0, length=1,
        num_domains=num_domains, domain_mu_range=domain_mu_range,
    )

    # [修正] eps ではなく H[:, 1, :] (t=1時点のh、つまり構造方程式が
    # 適用された後のh_t) にLiNGAMをかける
    h_t_samples = true_data["H"][:, 1, :]  # (n_samples, h_dim)
    print(f"\n収集した真のh_tサンプル数: {h_t_samples.shape}")
    print(f"h_tの平均（各次元）: {h_t_samples.mean(axis=0)}")
    print(f"h_tの標準偏差（各次元）: {h_t_samples.std(axis=0)}")

    print("\n--- LiNGAM実行中（対象: h_t）---")
    lingam_oracle = DirectLiNGAM()
    lingam_oracle.fit(h_t_samples)
    causal_order = lingam_oracle.causal_order_
    adjacency = lingam_oracle.adjacency_matrix_

    print(f"\n推定された因果順序: {causal_order}")
    print(f"期待される因果順序: [0, 1, 2]")

    is_correct = (list(causal_order) == [0, 1, 2])
    print(f"\n{'✓ 一致' if is_correct else '✗ 不一致'}")

    print(f"\nLiNGAMが推定した隣接行列（h_t間の因果構造）:")
    print(adjacency)
    print(f"\n真のM（比較用）:")
    print(M)

    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)
    if is_correct:
        print(
            "✓ 真のh_tからは正しい因果順序が復元できました。\n"
            "  → LiNGAM・補題2自体は今回のデータ設定において機能します。\n"
            "  → これまでの実装で eps_hat にLiNGAMをかけていたのは設計ミスでした。\n"
            "  → 修正版では、学習後の後処理を h_hat（またはB*h_{t-1}を除いた残差では"
            "なく h_hat そのもの）に対してLiNGAMをかける形に変更する必要があります。"
        )
    else:
        print(
            "✗ 真のh_tからでも正しい因果順序が復元できませんでした。\n"
            "  → h_{t-1}由来の項（B h_{t-1}）が十分に非ガウス的でない、\n"
            "    あるいはLiNGAMの他の適用条件が満たされていない可能性があります。\n"
            "  → domain_mu_range, noise_scale, num_domainsを変えて再実行してください。"
        )

