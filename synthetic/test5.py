"""
テスト5: 負例サンプリング修正を前提とした num_domains の系統的検証
============================================================
背景:
    テスト4で、負例サンプリングの誤ラベルバグ（自己参照が除外されて
    いなかった）を修正した結果、n_domains=2の極端ケースは劇的に改善
    したが、n_domains=8では依然としてほぼ学習が進まなかった。

    考えられる原因は2つ:
      (a) 単純に学習不足（ステップ数・学習率が8クラス識別には
          足りていない）
      (b) num_domains自体が本質的にGCLの最適化を難しくしている
          （A3条件が要求する多様性と、最適化のしやすさが対立する）

目的:
    num_domains × total_steps の組み合わせを系統的に振り、
    どこまでステップ数を増やせば学習が進むかを特定する。
    A3条件（num_domains >= h_dim*k+1 = 7）を満たす最小限の
    ドメイン数（7または8）で、実用的なステップ数内に学習が
    収束するかを確認する。

実行方法:
    python test5_num_domains_sweep.py
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class GCLScorer(nn.Module):
    def __init__(self, h_dim, num_domains, hidden_dim=64):
        super().__init__()
        self.h_dim = h_dim
        self.u_embedding = nn.Embedding(num_domains, hidden_dim)
        self.gcl_param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * h_dim),
        )

    def gcl_score(self, eps, u):
        emb = self.u_embedding(u)
        params = self.gcl_param_net(emb)
        a, b = params[:, :self.h_dim], params[:, self.h_dim:]
        return torch.sum(a * eps + b * torch.abs(eps), dim=-1)

    def gcl_loss(self, eps, u):
        pos_score = self.gcl_score(eps, u)

        # [修正] 自己参照（同ドメイン）を確実に除外する負例サンプリング
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


def run_sweep_case(num_domains, total_steps, h_dim=3, batch_size=1024,
                     scale_min=0.6, scale_max=1.4, mu_range=0.5,
                     lr=1e-2, hidden_dim=64, seed=0, log_every=None):
    """
    num_domains, total_steps を変えて gcl_loss の収束を測定。
    scale/muは実データ相当の設定をデフォルトにしている。
    """
    torch.manual_seed(seed)
    model = GCLScorer(h_dim=h_dim, num_domains=num_domains, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    domain_mu = torch.linspace(-mu_range, mu_range, num_domains).unsqueeze(1).repeat(1, h_dim)
    domain_scale = torch.linspace(scale_min, scale_max, num_domains).unsqueeze(1).repeat(1, h_dim)

    losses = []
    t0 = time.time()
    for step in range(total_steps):
        u = torch.randint(0, num_domains, (batch_size,))
        mu = domain_mu[u]
        scale = domain_scale[u]
        eps = torch.distributions.Laplace(loc=mu, scale=scale).sample()

        optimizer.zero_grad()
        loss = model.gcl_loss(eps, u)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if log_every and step % log_every == 0:
            recent = np.mean(losses[-log_every:]) if len(losses) >= log_every else np.mean(losses)
            print(f"    step={step:6d}  gcl_loss(直近平均)={recent:.4f}")

    elapsed = time.time() - t0
    initial = np.mean(losses[:100])
    final = np.mean(losses[-100:])
    chance = 2 * np.log(2)

    return {
        "num_domains": num_domains, "total_steps": total_steps,
        "initial": initial, "final": final, "chance": chance,
        "improved": final < chance * 0.7, "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("テスト5: num_domains × total_steps の系統的検証")
    print("=" * 70)
    print("設定: domain_scale=[0.6,1.4], domain_mu_range=5.0 (診断Aで有効性を確認済み)")
    print("=" * 70)

    # A3条件: num_domains >= h_dim*k+1 = 3*2+1 = 7
    # 7未満は理論上の識別可能性を満たさないため対象から除外し、
    # 7以上の範囲でステップ数との関係を調べる。
    domain_candidates = [7, 8, 12, 16]
    step_candidates = [3_000, 10_000, 30_000]

    # [確定] 診断A(mu_range sweep)で mu_range=5.0 が学習を成功させる
    # ことを確認済みのため、ここではmu_range=5.0を固定して
    # num_domains x total_stepsの効果を見る
    MU_RANGE = 5.0

    results = []
    for nd in domain_candidates:
        for steps in step_candidates:
            print(f"\n--- num_domains={nd}, total_steps={steps} を実行中 ---")
            r = run_sweep_case(num_domains=nd, total_steps=steps, mu_range=MU_RANGE, log_every=None)
            results.append(r)
            status = "✓ 学習成功" if r["improved"] else "✗ 未収束"
            print(f"    初期={r['initial']:.4f} 最終={r['final']:.4f} "
                  f"(chance={r['chance']:.4f}) [{status}] 所要時間={r['elapsed_sec']:.1f}秒")

    # ------------------------------------------------------------
    # 結果まとめ
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("結果まとめ（表形式）")
    print("=" * 70)
    print(f"{'num_domains':>12s} | " + " | ".join(f"steps={s:>7d}" for s in step_candidates))
    print("-" * 70)
    for nd in domain_candidates:
        row = [r for r in results if r["num_domains"] == nd]
        row_str = " | ".join(
            f"{'✓' if r['improved'] else '✗'} final={r['final']:.3f}"
            for r in sorted(row, key=lambda x: x["total_steps"])
        )
        print(f"{nd:>12d} | {row_str}")

    print("\n" + "=" * 70)
    print("判定")
    print("=" * 70)

    # A3条件を満たす最小のnum_domains(=7)で、実用的なステップ数(<=30000)
    # で学習が成功するかを最優先でチェック
    nd7_results = [r for r in results if r["num_domains"] == 7]
    nd7_success_steps = [r["total_steps"] for r in nd7_results if r["improved"]]

    if nd7_success_steps:
        min_steps = min(nd7_success_steps)
        print(
            f"✓ num_domains=7（A3条件の最小値）で、{min_steps}ステップから学習が成功しています。\n"
            f"  → synthetic_gcl_v3_fixed.py の total_steps を {min_steps}以上に設定し、\n"
            f"    num_domains=7または8のまま、負例サンプリング修正を適用して再実行してください。"
        )
    else:
        print(
            "✗ num_domains=7では、テストした範囲のステップ数(最大30,000)でも学習が\n"
            "  成功しませんでした。\n"
            "  → より多くのステップ数が必要か、hidden_dim/学習率などアーキテクチャ側の\n"
            "    調整が必要です。以下のいずれかを試してください:\n"
            "    1. total_steps を50,000以上に増やす\n"
            "    2. hidden_dim を64->128に増やす\n"
            "    3. GCL部分だけ学習率を上げる（gcl_param_net用に別のoptimizerを用意）\n"
            "    4. A3条件を満たしつつドメイン数を絞れないか再検討する\n"
            "       （例: domain_mu と domain_scale を独立に変化させる設計にし、\n"
            "       各々のドメイン数を減らして組み合わせでnum_domains相当を稼ぐ）"
        )

    # より大きいnum_domainsで改善する場合、逆に大きい方が良い可能性も報告
    best = max(results, key=lambda r: (r["improved"], -r["final"]))
    print(
        f"\n参考: 全条件中で最も良い結果は "
        f"num_domains={best['num_domains']}, total_steps={best['total_steps']} "
        f"(final_loss={best['final']:.4f})"
    )