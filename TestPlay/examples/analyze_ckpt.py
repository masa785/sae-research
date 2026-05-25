import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a Linear IDOL checkpoint and export instantaneous/time-delayed relations."
    )
    parser.add_argument("--ckpt", required=True, help="Path to a .ckp file saved by TestPlay/examples/main.py")
    parser.add_argument("--out-dir", default=None, help="Directory for analysis outputs. Defaults to <run_dir>/analysis")
    parser.add_argument("--top-edges", type=int, default=200, help="Number of strongest edges to export")
    parser.add_argument("--heatmap-features", type=int, default=80, help="Number of high-strength features to include in heatmaps")
    parser.add_argument("--include-self", action="store_true", help="Keep self edges i -> i in exported relations")
    parser.add_argument(
        "--m-diagonal",
        type=int,
        default=1,
        help="Triangular mask diagonal for M. The training forward pass uses torch.tril(M, diagonal=1).",
    )
    return parser.parse_args()


def default_out_dir(ckpt_path):
    ckpt_dir = Path(ckpt_path).resolve().parent
    if ckpt_dir.name == "ckps":
        return ckpt_dir.parent / "analysis"
    return ckpt_dir / "analysis"


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict in {path}, got {type(state)}")
    required = {"M", "F_enc", "F_dec"}
    missing = required - set(state)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")
    return state


def sorted_b_keys(state):
    keys = [key for key in state if key.startswith("Bs.")]
    if not keys:
        raise KeyError("Checkpoint has no Bs.* lag matrices")
    return sorted(keys, key=lambda key: int(key.split(".")[1]))


def maybe_mask_self(abs_matrix, include_self):
    if include_self:
        return abs_matrix
    masked = abs_matrix.clone()
    diag_n = min(masked.shape)
    masked[torch.arange(diag_n), torch.arange(diag_n)] = -torch.inf
    return masked


def top_edges_from_matrix(matrix, top_k, relation_type, include_self, lag=None):
    signed = matrix.detach().cpu().float()
    abs_values = maybe_mask_self(signed.abs(), include_self)
    flat = abs_values.reshape(-1)
    valid_count = int(torch.isfinite(flat).sum().item())
    k = min(top_k, valid_count)
    if k <= 0:
        return []

    values, indices = torch.topk(flat, k=k)
    z_dim = signed.shape[0]
    rows = []
    for rank, (abs_value, flat_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        target = flat_idx // z_dim
        source = flat_idx % z_dim
        signed_value = float(signed[target, source].item())
        row = {
            "rank": rank,
            "relation_type": relation_type,
            "source_feature": source,
            "target_feature": target,
            "weight": signed_value,
            "abs_weight": float(abs_value),
        }
        if lag is not None:
            row["lag"] = lag
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_feature_summary(path, rows):
    fieldnames = [
        "feature",
        "encoder_norm",
        "decoder_norm",
        "instant_in_abs",
        "instant_out_abs",
        "delayed_in_abs",
        "delayed_out_abs",
        "total_relation_abs",
    ]
    write_csv(path, rows, fieldnames)


def plot_heatmap(matrix, feature_ids, title, path):
    if not feature_ids:
        return
    sub = matrix[np.ix_(feature_ids, feature_ids)]
    vmax = float(np.max(np.abs(sub))) if sub.size else 0.0
    if vmax == 0:
        vmax = 1.0

    fig_size = max(6, min(18, len(feature_ids) * 0.18))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sub, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("source feature")
    ax.set_ylabel("target feature")

    tick_step = max(1, len(feature_ids) // 12)
    tick_positions = list(range(0, len(feature_ids), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(feature_ids[i]) for i in tick_positions], rotation=90)
    ax.set_yticklabels([str(feature_ids[i]) for i in tick_positions])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_lag_strengths(lag_rows, path):
    lags = [row["lag"] for row in lag_rows]
    sums = [row["abs_sum"] for row in lag_rows]
    maxes = [row["abs_max"] for row in lag_rows]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(lags, sums, color="#4C78A8", alpha=0.75)
    ax1.set_xlabel("lag")
    ax1.set_ylabel("sum abs(B_lag)")

    ax2 = ax1.twinx()
    ax2.plot(lags, maxes, color="#F58518", marker="o", linewidth=1.5)
    ax2.set_ylabel("max abs(B_lag)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_checkpoint(ckpt_path)
    b_keys = sorted_b_keys(state)
    m_raw = state["M"].detach().cpu().float()
    m_effective = torch.tril(m_raw, diagonal=args.m_diagonal)
    z_dim = m_effective.shape[0]

    b_max_abs = torch.zeros((z_dim, z_dim), dtype=torch.float32)
    b_max_signed = torch.zeros((z_dim, z_dim), dtype=torch.float32)
    b_best_lag = torch.zeros((z_dim, z_dim), dtype=torch.int16)
    lag_strength_rows = []
    all_lag_top_edges = []

    for lag, key in enumerate(b_keys, start=1):
        b_lag = state[key].detach().cpu().float()
        b_abs = b_lag.abs()
        update = b_abs > b_max_abs
        b_max_abs[update] = b_abs[update]
        b_max_signed[update] = b_lag[update]
        b_best_lag[update] = lag
        lag_strength_rows.append(
            {
                "lag": lag,
                "abs_sum": float(b_abs.sum().item()),
                "abs_mean": float(b_abs.mean().item()),
                "abs_max": float(b_abs.max().item()),
            }
        )
        all_lag_top_edges.extend(
            top_edges_from_matrix(
                b_lag,
                args.top_edges,
                relation_type="time_delayed_lag",
                include_self=args.include_self,
                lag=lag,
            )
        )

    instant_edges = top_edges_from_matrix(
        m_effective,
        args.top_edges,
        relation_type="instantaneous",
        include_self=args.include_self,
    )
    delayed_edges = top_edges_from_matrix(
        b_max_signed,
        args.top_edges,
        relation_type="time_delayed_max_over_lags",
        include_self=args.include_self,
    )
    for row in delayed_edges:
        target = row["target_feature"]
        source = row["source_feature"]
        row["best_lag"] = int(b_best_lag[target, source].item())

    all_lag_top_edges = sorted(all_lag_top_edges, key=lambda row: row["abs_weight"], reverse=True)[: args.top_edges]
    for rank, row in enumerate(all_lag_top_edges, start=1):
        row["rank"] = rank

    write_csv(
        out_dir / "instantaneous_edges.csv",
        instant_edges,
        ["rank", "relation_type", "source_feature", "target_feature", "weight", "abs_weight"],
    )
    write_csv(
        out_dir / "time_delayed_edges_max_over_lags.csv",
        delayed_edges,
        ["rank", "relation_type", "source_feature", "target_feature", "best_lag", "weight", "abs_weight"],
    )
    write_csv(
        out_dir / "time_delayed_edges_by_lag.csv",
        all_lag_top_edges,
        ["rank", "relation_type", "lag", "source_feature", "target_feature", "weight", "abs_weight"],
    )
    write_csv(
        out_dir / "lag_strengths.csv",
        lag_strength_rows,
        ["lag", "abs_sum", "abs_mean", "abs_max"],
    )

    f_enc = state["F_enc"].detach().cpu().float()
    f_dec = state["F_dec"].detach().cpu().float()
    instant_abs = m_effective.abs()
    delayed_abs = b_max_abs
    total_strength = (
        instant_abs.sum(dim=0)
        + instant_abs.sum(dim=1)
        + delayed_abs.sum(dim=0)
        + delayed_abs.sum(dim=1)
    )
    feature_rows = []
    for feature in range(z_dim):
        feature_rows.append(
            {
                "feature": feature,
                "encoder_norm": float(f_enc[:, feature].norm().item()),
                "decoder_norm": float(f_dec[feature, :].norm().item()),
                "instant_in_abs": float(instant_abs[feature, :].sum().item()),
                "instant_out_abs": float(instant_abs[:, feature].sum().item()),
                "delayed_in_abs": float(delayed_abs[feature, :].sum().item()),
                "delayed_out_abs": float(delayed_abs[:, feature].sum().item()),
                "total_relation_abs": float(total_strength[feature].item()),
            }
        )
    feature_rows = sorted(feature_rows, key=lambda row: row["total_relation_abs"], reverse=True)
    write_feature_summary(out_dir / "feature_summary.csv", feature_rows)

    heatmap_features = [row["feature"] for row in feature_rows[: args.heatmap_features]]
    plot_heatmap(
        m_effective.numpy(),
        heatmap_features,
        "Instantaneous relation M (top relation-strength features)",
        out_dir / "instantaneous_M_heatmap.png",
    )
    plot_heatmap(
        b_max_signed.numpy(),
        heatmap_features,
        "Time-delayed relation max_lag B_lag (top relation-strength features)",
        out_dir / "time_delayed_Bmax_heatmap.png",
    )
    plot_lag_strengths(lag_strength_rows, out_dir / "lag_strengths.png")

    summary = {
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
        "z_dim": z_dim,
        "tau": len(b_keys),
        "m_diagonal": args.m_diagonal,
        "include_self": args.include_self,
        "top_edges": args.top_edges,
        "heatmap_features": args.heatmap_features,
        "outputs": [
            "instantaneous_edges.csv",
            "time_delayed_edges_max_over_lags.csv",
            "time_delayed_edges_by_lag.csv",
            "lag_strengths.csv",
            "feature_summary.csv",
            "instantaneous_M_heatmap.png",
            "time_delayed_Bmax_heatmap.png",
            "lag_strengths.png",
        ],
        "edge_direction": "matrix[target_feature, source_feature], matching Z_target += W[target, source] * Z_source",
    }
    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
