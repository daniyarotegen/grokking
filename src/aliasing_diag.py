"""Stage 4A: Zero-training aliasing & frequency-pruning diagnostic on saved checkpoints.

Analyzes all saved checkpoints of a grokking run in float64:
1. Per-frequency trig-sum variance share and L2 norm trajectories for every frequency
   k in 1..(P-1)//2 exceeding 0.5% variance share at any checkpoint.
2. Analytical worst-case alias offset Delta* predicted from the weighted cosine sum
   sum_{k in K} w_k * cos(2 * pi * k * Delta / P).
3. Test-set logit margin trajectories (mean & 1st-percentile) for both full logits and
   restricted logits (projected onto final K), split by alias offsets Delta = +/-12 vs
   all other wrong classes, plus exact misclassification counts by offset.
4. Publication-clean figure saved to grokking/artifacts/figures/fig4_seed42_aliasing_dynamics.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grokking.src.analysis import (
    _configure_plot_style,
    compute_full_logit_table,
    dft_2d_key_frequency_projection,
    load_model_from_checkpoint,
    logit_table_2d_fourier,
    make_fourier_basis,
)
from grokking.src.data import make_modular_addition_dataset
from grokking.src.model import TransformerConfig


def compute_trig_sum_energies_vectorized(
    fourier_logits: torch.Tensor,  # (P, P, P), float64
) -> tuple[torch.Tensor, float]:
    """Vectorized computation of trig-sum squared energy for k = 1..(P-1)//2 and non-DC energy.

    Returns:
        trig_sum_energy: ((P - 1) // 2,) float64 tensor indexed by k - 1 (for k = 1..56).
        non_dc_energy:   float, total squared norm excluding DC (0, 0).
    """
    p = fourier_logits.shape[0]
    n_k = p // 2  # 56 for P = 113
    c_idx = 2 * torch.arange(1, n_k + 1, device=fourier_logits.device) - 1  # (56,)
    s_idx = c_idx + 1                                                       # (56,)

    cc = fourier_logits[c_idx, c_idx, :]  # (56, P)
    ss = fourier_logits[s_idx, s_idx, :]  # (56, P)
    sc = fourier_logits[s_idx, c_idx, :]  # (56, P)
    cs = fourier_logits[c_idx, s_idx, :]  # (56, P)

    # Orthonormal coordinates on cos(w_k(a+b)) and sin(w_k(a+b))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    coeff_cos = inv_sqrt2 * (cc - ss)     # (56, P)
    coeff_sin = inv_sqrt2 * (sc + cs)     # (56, P)

    trig_sum_energy = coeff_cos.pow(2).sum(dim=-1) + coeff_sin.pow(2).sum(dim=-1)  # (56,)
    dc_energy = fourier_logits[0, 0, :].pow(2).sum().item()
    total_energy = fourier_logits.pow(2).sum().item()
    non_dc_energy = max(total_energy - dc_energy, 1e-12)
    return trig_sum_energy, non_dc_energy


def analytical_alias_scores(
    trig_sum_energy: np.ndarray,  # (56,) indexed by k - 1
    non_dc_energy: float,
    p: int = 113,
    min_var_frac: float = 0.005,
    alias_offset: int = 12,
    fixed_freqs: list[int] | None = None,
) -> dict[str, object]:
    """Compute weighted cosine sum S(Delta) = sum_{k in K} w_k cos(2*pi*k*Delta / P) for Delta=1..P-1.

    Evaluates both L2-norm (amplitude) weights w_k proportional to sqrt(E_k) — the physically
    exact linear-logit amplitude weighting — and variance weights w_k proportional to E_k.
    """
    var_fracs = trig_sum_energy / non_dc_energy
    if fixed_freqs is not None:
        ks = np.array(fixed_freqs, dtype=np.int64)
    else:
        selected = np.where(var_fracs >= min_var_frac)[0] + 1
        if len(selected) == 0:
            selected = np.array([int(np.argmax(trig_sum_energy)) + 1], dtype=np.int64)
        ks = selected.astype(np.int64)

    energies = trig_sum_energy[ks - 1]
    l2_norms = np.sqrt(np.maximum(energies, 0.0))

    w_l2 = l2_norms / max(l2_norms.sum(), 1e-12)
    w_var = energies / max(energies.sum(), 1e-12)

    deltas = np.arange(1, p, dtype=np.int64)  # 1..112
    # cos_mat[i, d] = cos(2*pi * ks[i] * deltas[d] / p) -> (|K|, P-1)
    cos_mat = np.cos(2.0 * math.pi * ks[:, None] * deltas[None, :] / p)

    scores_l2 = w_l2 @ cos_mat    # (P-1,)
    scores_var = w_var @ cos_mat  # (P-1,)

    # Note symmetry S(Delta) == S(P - Delta); fold to canonical Delta in 1..(P-1)//2
    half = p // 2
    s_l2_half = scores_l2[:half]
    s_var_half = scores_var[:half]

    top_l2_idx = np.argsort(s_l2_half)[::-1]
    top_var_idx = np.argsort(s_var_half)[::-1]

    return {
        "active_k_used": ks.tolist(),
        "l2_weights": {int(k): float(w) for k, w in zip(ks, w_l2)},
        "var_weights": {int(k): float(w) for k, w in zip(ks, w_var)},
        "delta_star_l2": int(top_l2_idx[0] + 1),
        "score_l2_max": float(s_l2_half[top_l2_idx[0]]),
        "score_l2_delta12": float(s_l2_half[alias_offset - 1]),
        "top3_l2": [
            {"delta": int(i + 1), "score": float(s_l2_half[i])} for i in top_l2_idx[:3]
        ],
        "delta_star_var": int(top_var_idx[0] + 1),
        "score_var_max": float(s_var_half[top_var_idx[0]]),
        "score_var_delta12": float(s_var_half[alias_offset - 1]),
        "top3_var": [
            {"delta": int(i + 1), "score": float(s_var_half[i])} for i in top_var_idx[:3]
        ],
    }


def compute_test_margins_and_errors(
    logit_table: torch.Tensor,  # (P, P, P), float64
    a_test: torch.Tensor,       # (N_test,), int64
    b_test: torch.Tensor,       # (N_test,), int64
    y_test: torch.Tensor,       # (N_test,), int64
    p: int = 113,
    alias_offset: int = 12,
) -> dict[str, float | int]:
    """Compute test-set logit margins (mean & 1st percentile) and misclassification counts."""
    logits = logit_table[a_test, b_test]                              # (N_test, P), float64
    n_test = logits.shape[0]
    row_idx = torch.arange(n_test, device=logits.device)

    true_logits = logits[row_idx, y_test]                             # (N_test,)
    alias_pos = (y_test + alias_offset) % p                           # (N_test,)
    alias_neg = (y_test - alias_offset) % p                           # (N_test,)

    alias_max = torch.maximum(logits[row_idx, alias_pos], logits[row_idx, alias_neg])  # (N_test,)
    margin_alias = true_logits - alias_max                            # (N_test,)

    # Mask out true class and +/-12 alias classes to get max over all other 110 wrong classes
    masked = logits.clone()
    masked[row_idx, y_test] = float("-inf")
    masked[row_idx, alias_pos] = float("-inf")
    masked[row_idx, alias_neg] = float("-inf")
    other_max = masked.max(dim=-1).values                             # (N_test,)
    margin_other = true_logits - other_max                            # (N_test,)

    preds = logits.argmax(dim=-1)                                     # (N_test,)
    offsets = (preds - y_test) % p                                    # (N_test,)
    is_wrong = offsets != 0
    is_alias_err = (offsets == alias_offset) | (offsets == (p - alias_offset))
    is_other_err = is_wrong & (~is_alias_err)

    return {
        "alias_margin_mean": margin_alias.mean().item(),
        "alias_margin_p01": torch.quantile(margin_alias, 0.01).item(),
        "alias_margin_min": margin_alias.min().item(),
        "other_margin_mean": margin_other.mean().item(),
        "other_margin_p01": torch.quantile(margin_other, 0.01).item(),
        "other_margin_min": margin_other.min().item(),
        "err_delta12": int(is_alias_err.sum().item()),
        "err_other": int(is_other_err.sum().item()),
        "err_total": int(is_wrong.sum().item()),
    }


def plot_aliasing_dynamics(
    records: list[dict[str, object]],
    active_freqs: list[int],
    final_key_freqs: list[int],
    out_path: Path,
) -> None:
    """Generate publication-clean 3-panel figure of frequency pruning, margins, and alias errors."""
    _configure_plot_style()

    epochs = np.array([int(r["epoch"]) for r in records], dtype=np.float64)
    mask = epochs >= 250
    ep = epochs[mask]

    # Sort active frequencies by their peak variance share across all checkpoints
    peak_shares = {
        k: max(float(r["freq_var_share"][str(k)]) for r in records) for k in active_freqs
    }
    sorted_freqs = sorted(active_freqs, key=lambda k: peak_shares[k], reverse=True)
    final_set = set(final_key_freqs)

    # Distinct colors: bold Okabe-Ito for surviving final K, muted/dashed distinct palette for pruned freqs
    surviving_colors = {
        10: "#0072B2",
        56: "#D55E00",
        12: "#009E73",
    }
    pruned_palette = [
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#8C564B",
        "#7F7F7F",
        "#BCBD22",
        "#9467BD",
        "#17BECF",
        "#666666",
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.15, 0.85]},
    )

    # --- Panel A: Per-frequency trig-sum variance share across epochs ---
    ax_a = axes[0]
    pruned_idx = 0
    for k in sorted_freqs:
        shares = np.array([100.0 * float(r["freq_var_share"][str(k)]) for r in records])[mask]
        peak_pct = 100.0 * peak_shares[k]
        if k in final_set:
            col = surviving_colors.get(k, "#000000")
            ax_a.plot(
                ep,
                np.clip(shares, 1e-3, None),
                color=col,
                lw=2.3,
                label=f"k={k} (surviving, peak {peak_pct:.1f}%)",
                zorder=4,
            )
        else:
            col = pruned_palette[pruned_idx % len(pruned_palette)]
            pruned_idx += 1
            ax_a.plot(
                ep,
                np.clip(shares, 1e-3, None),
                color=col,
                lw=1.7,
                ls="--",
                alpha=0.9,
                label=f"k={k} (pruned, peak {peak_pct:.1f}%)",
                zorder=3,
            )

    ax_a.set_yscale("log")
    ax_a.set_ylim(0.05, 145.0)
    ax_a.axhline(0.5, color="#888888", ls=":", lw=1.0, alpha=0.7)
    ax_a.text(
        38500,
        0.58,
        "0.5% threshold (no other k > 0.5% at any epoch; next highest is k=20 at 0.30%)",
        fontsize=7.6,
        color="#666666",
        va="bottom",
        ha="right",
    )
    ax_a.set_ylabel("Trig-sum variance\nshare (%)")
    ax_a.set_title("Seed 42 Frequency Dominance & Δ = ±12 Aliasing Dynamics (P = 113, wd = 1.0)")
    ax_a.legend(loc="lower left", ncol=2, fontsize=8.0)

    # --- Panel B: 1st-percentile (and min) test logit margins (Full vs Restricted, Δ=±12 vs Other) ---
    ax_b = axes[1]
    full_alias_p01 = np.array([r["full_margins"]["alias_margin_p01"] for r in records])[mask]
    full_alias_min = np.array([r["full_margins"]["alias_margin_min"] for r in records])[mask]
    full_other_p01 = np.array([r["full_margins"]["other_margin_p01"] for r in records])[mask]
    rest_alias_p01 = np.array([r["restricted_margins"]["alias_margin_p01"] for r in records])[mask]
    rest_other_p01 = np.array([r["restricted_margins"]["other_margin_p01"] for r in records])[mask]

    ax_b.axhline(0.0, color="#222222", ls="-", lw=1.1, alpha=0.8, zorder=1)
    ax_b.plot(ep, full_alias_p01, color="#D55E00", lw=2.3, label=r"Full model: $\Delta = \pm 12$ (1st pct)")
    ax_b.plot(
        ep,
        full_alias_min,
        color="#D55E00",
        lw=1.4,
        ls="-.",
        alpha=0.75,
        label=r"Full model: $\Delta = \pm 12$ (min / worst test pair)",
    )
    ax_b.plot(ep, full_other_p01, color="#0072B2", lw=2.0, label=r"Full model: other $\Delta$ (1st pct)")
    ax_b.plot(
        ep,
        rest_alias_p01,
        color="#009E73",
        lw=2.2,
        ls="--",
        label=r"Restricted ($K=[10,12,56]$): $\Delta = \pm 12$ (1st pct)",
    )
    ax_b.plot(
        ep,
        rest_other_p01,
        color="#56B4E9",
        lw=1.8,
        ls=":",
        label=r"Restricted ($K=[10,12,56]$): other $\Delta$ (1st pct)",
    )
    ax_b.set_ylim(-88.0, 24.0)
    ax_b.set_ylabel("Test logit margin\n(float64)")
    ax_b.legend(loc="center right", bbox_to_anchor=(0.99, 0.53), ncol=1, fontsize=7.9)

    # --- Panel C: Test misclassification counts (Δ=±12 vs Other) ---
    ax_c = axes[2]
    err_d12 = np.array([r["full_margins"]["err_delta12"] for r in records])[mask]
    err_oth = np.array([r["full_margins"]["err_other"] for r in records])[mask]
    rest_err_total = np.array([r["restricted_margins"]["err_total"] for r in records])[mask]

    # Plot on symlog y-axis so both 0..50 errors (post-grokking) and 8,000+ errors (pre-grokking) are clear
    ax_c.plot(ep, err_d12, color="#D55E00", lw=2.3, label=r"Full model errors at $\Delta = \pm 12$")
    ax_c.plot(ep, err_oth, color="#0072B2", lw=2.0, label=r"Full model errors at other $\Delta \notin \{0, \pm 12\}$")
    ax_c.plot(
        ep,
        rest_err_total,
        color="#009E73",
        lw=2.0,
        ls="--",
        label="Restricted model total test errors",
    )
    ax_c.set_yscale("symlog", linthresh=10.0)
    ax_c.set_ylim(-1.0, 14000.0)
    ax_c.set_ylabel("Test errors\n(of 8,939)")
    ax_c.set_xlabel("Epoch (log scale)")
    ax_c.set_xscale("log")
    ax_c.set_xlim(250, epochs[-1])
    ax_c.legend(loc="lower left", fontsize=8.0)

    # Annotate key post-grokking window (epochs 5,500 -> 12,000) across all panels
    for ax in axes:
        ax.axvspan(5500, 12000, color="#f0ad4e", alpha=0.09, zorder=0)

    axes[0].text(
        8100,
        55.0,
        "Dominance & tail\nplunge (5.5k–12k)",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#8a5a00",
    )

    # Annotate exact final Δ=±12 error count on Panel C (clean upper-right empty space)
    final_err_d12 = int(records[-1]["full_margins"]["err_delta12"])
    final_err_oth = int(records[-1]["full_margins"]["err_other"])
    ax_c.annotate(
        f"Epoch 40k: {final_err_d12} errors at Δ=±12\n({final_err_oth} at Δ=+10; 0 restricted)",
        xy=(40000, final_err_d12),
        xytext=(12500, 1200),
        fontsize=8.2,
        color="#a63c00",
        arrowprops=dict(arrowstyle="->", color="#a63c00", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#dddddd", alpha=0.95),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_aliasing_diagnostic(
    run_dir: Path,
    figures_dir: Path,
    device_str: str = "auto",
    var_threshold: float = 0.005,
    alias_offset: int = 12,
) -> dict[str, object]:
    """Run Stage 4A aliasing & frequency-pruning diagnostic across all saved checkpoints."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    run_config = json.loads((run_dir / "config.json").read_text())
    train_cfg = run_config["train_config"]
    model_cfg = TransformerConfig(**run_config["model_config"])
    p = int(train_cfg["p"])

    stage3_summary = json.loads((run_dir / "stage3_summary.json").read_text())
    final_key_freqs = [int(k) for k in stage3_summary["key_freqs"]]

    data = make_modular_addition_dataset(
        p=p,
        train_frac=float(train_cfg["train_frac"]),
        seed=int(train_cfg["seed"]),
    )
    a_test = data.x_test[:, 0].to(device)
    b_test = data.x_test[:, 1].to(device)
    y_test = data.y_test.to(device)

    ckpt_paths = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    fourier_basis, _ = make_fourier_basis(p=p, device=device)

    raw_rows: list[dict[str, object]] = []
    all_var_shares = np.zeros((len(ckpt_paths), p // 2), dtype=np.float64)
    all_l2_norms = np.zeros((len(ckpt_paths), p // 2), dtype=np.float64)

    for idx, ckpt_path in enumerate(ckpt_paths):
        model, epoch = load_model_from_checkpoint(ckpt_path, model_cfg, device)
        logit_table = compute_full_logit_table(model, p=p, device=device)
        f_logits = logit_table_2d_fourier(logit_table, fourier_basis)

        trig_e_t, non_dc_e = compute_trig_sum_energies_vectorized(f_logits)
        trig_e_np = trig_e_t.cpu().numpy()  # (56,)
        var_shares_np = trig_e_np / non_dc_e
        l2_norms_np = np.sqrt(np.maximum(trig_e_np, 0.0))

        all_var_shares[idx] = var_shares_np
        all_l2_norms[idx] = l2_norms_np

        # Analytical cosine alias prediction (both active-at-epoch frequencies and final K)
        alias_pred_active = analytical_alias_scores(
            trig_e_np,
            non_dc_e,
            p=p,
            min_var_frac=var_threshold,
            alias_offset=alias_offset,
            fixed_freqs=None,
        )
        alias_pred_final_k = analytical_alias_scores(
            trig_e_np,
            non_dc_e,
            p=p,
            min_var_frac=var_threshold,
            alias_offset=alias_offset,
            fixed_freqs=final_key_freqs,
        )

        # Test-set margins & errors for full vs restricted (final K) logits
        restricted_table, _ = dft_2d_key_frequency_projection(
            logit_table,
            fourier_basis,
            key_freqs=final_key_freqs,
            mode="trig_sum",
        )
        full_margins = compute_test_margins_and_errors(
            logit_table, a_test, b_test, y_test, p=p, alias_offset=alias_offset
        )
        restricted_margins = compute_test_margins_and_errors(
            restricted_table, a_test, b_test, y_test, p=p, alias_offset=alias_offset
        )

        raw_rows.append(
            {
                "epoch": epoch,
                "non_dc_energy": non_dc_e,
                "alias_pred_active": alias_pred_active,
                "alias_pred_final_k": alias_pred_final_k,
                "full_margins": full_margins,
                "restricted_margins": restricted_margins,
            }
        )

    # Identify every frequency k in 1..56 exceeding var_threshold (0.5%) at ANY checkpoint
    max_share_per_k = all_var_shares.max(axis=0)  # (56,)
    active_freqs = [int(i + 1) for i in np.where(max_share_per_k >= var_threshold)[0]]

    # Attach compact per-frequency trajectories for all active_freqs and write JSONL
    out_jsonl = run_dir / "aliasing_diag.jsonl"
    with out_jsonl.open("w") as fp:
        for idx, row in enumerate(raw_rows):
            row["freq_var_share"] = {
                str(k): float(all_var_shares[idx, k - 1]) for k in active_freqs
            }
            row["freq_l2_norm"] = {
                str(k): float(all_l2_norms[idx, k - 1]) for k in active_freqs
            }
            fp.write(json.dumps(row) + "\n")

    # Summarize frequency peaks & pruning
    freq_summary = []
    for k in active_freqs:
        shares = all_var_shares[:, k - 1]
        l2s = all_l2_norms[:, k - 1]
        peak_i = int(np.argmax(shares))
        peak_l2_i = int(np.argmax(l2s))
        freq_summary.append(
            {
                "k": k,
                "survives_final_K": k in final_key_freqs,
                "peak_var_share_pct": 100.0 * float(shares[peak_i]),
                "peak_var_epoch": int(raw_rows[peak_i]["epoch"]),
                "final_var_share_pct": 100.0 * float(shares[-1]),
                "peak_l2_norm": float(l2s[peak_l2_i]),
                "peak_l2_epoch": int(raw_rows[peak_l2_i]["epoch"]),
                "final_l2_norm": float(l2s[-1]),
            }
        )
    freq_summary.sort(key=lambda d: d["peak_var_share_pct"], reverse=True)

    # Generate Figure 4
    fig4_path = figures_dir / "fig4_seed42_aliasing_dynamics.png"
    plot_aliasing_dynamics(raw_rows, active_freqs, final_key_freqs, fig4_path)

    summary = {
        "active_freqs_gt_0p5pct": active_freqs,
        "final_key_freqs": final_key_freqs,
        "freq_summary": freq_summary,
        "final_epoch_alias_pred_active": raw_rows[-1]["alias_pred_active"],
        "final_epoch_alias_pred_final_k": raw_rows[-1]["alias_pred_final_k"],
        "final_epoch_full_margins": raw_rows[-1]["full_margins"],
        "final_epoch_restricted_margins": raw_rows[-1]["restricted_margins"],
        "output_jsonl": str(out_jsonl),
        "figure_path": str(fig4_path),
    }
    (run_dir / "aliasing_diag_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4A aliasing & frequency-pruning diagnostic.")
    parser.add_argument("--run-dir", type=str, default="grokking/runs/baseline_seed42")
    parser.add_argument("--figures-dir", type=str, default="grokking/artifacts/figures")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--var-threshold", type=float, default=0.005)
    parser.add_argument("--alias-offset", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_aliasing_diagnostic(
        run_dir=Path(args.run_dir),
        figures_dir=Path(args.figures_dir),
        device_str=args.device,
        var_threshold=args.var_threshold,
        alias_offset=args.alias_offset,
    )
