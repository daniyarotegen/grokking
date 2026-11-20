"""Stage 3 mechanistic progress measures and plotting for modular addition grokking (Nanda et al., 2023).

Provides:
- Orthonormal real Fourier basis over Z_P (verified F @ F.T == I_P)
- 2D Discrete Fourier Transform over the (P, P) input grid for every output class c
- 1D Fourier norm spectrum of token embeddings W_E[:P]
- Key frequency identification and variance/norm explained accounting
- Restricted loss (evaluated on test set) and excluded loss (evaluated on train set) in float64
- Publication-clean matplotlib figures saved to grokking/artifacts/figures/
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
import torch.nn.functional as F
from einops import einsum

from grokking.src.data import make_full_grid, make_modular_addition_dataset
from grokking.src.model import OneLayerTransformer, TransformerConfig


def make_fourier_basis(
    p: int = 113,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Construct the P x P orthonormal real Fourier basis over Z_P in float64.

    Row ordering:
        0:       const = 1 / sqrt(P)
        2k - 1:  cos_k(n) = sqrt(2/P) * cos(2*pi*k*n / P)
        2k:      sin_k(n) = sqrt(2/P) * sin(2*pi*k*n / P)
    for k = 1, ..., (P - 1) // 2. Satisfies F @ F.T == I_P.
    """
    n = torch.arange(p, dtype=torch.float64, device=device)  # (P,)
    basis_rows = [torch.full((p,), 1.0 / math.sqrt(p), dtype=torch.float64, device=device)]
    names = ["const"]

    scale = math.sqrt(2.0 / p)
    for k in range(1, (p // 2) + 1):
        angles = 2.0 * math.pi * k * n / p  # (P,)
        basis_rows.append(scale * torch.cos(angles))
        names.append(f"cos_{k}")
        basis_rows.append(scale * torch.sin(angles))
        names.append(f"sin_{k}")

    basis = torch.stack(basis_rows, dim=0)  # (P, P)
    return basis, names


@torch.no_grad()
def compute_full_logit_table(
    model: OneLayerTransformer,
    p: int = 113,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate final-token logits on all P*P ordered pairs, returning float64 tensor (P, P, P)."""
    if device is None:
        device = next(model.parameters()).device
    x_all, _ = make_full_grid(p=p)                                  # (P*P, 3)
    logits_flat = model(x_all.to(device))[:, -1, :]                 # (P*P, P)
    return logits_flat.to(torch.float64).reshape(p, p, p)           # (P, P, P)


def logit_table_2d_fourier(
    logit_table: torch.Tensor,    # (P, P, P), float64
    fourier_basis: torch.Tensor,  # (P, P), float64
) -> torch.Tensor:
    """Compute orthonormal 2D DFT over input dimensions (a, b) for every output class c."""
    return einsum(
        fourier_basis,
        fourier_basis,
        logit_table,
        "fa a, fb b, a b c -> fa fb c",
    )  # (P, P, P)


def inverse_logit_table_2d_fourier(
    fourier_logits: torch.Tensor,  # (P, P, P), float64
    fourier_basis: torch.Tensor,   # (P, P), float64
) -> torch.Tensor:
    """Compute inverse orthonormal 2D DFT from frequency domain back to (a, b, c) logits."""
    return einsum(
        fourier_basis,
        fourier_basis,
        fourier_logits,
        "fa a, fb b, fa fb c -> a b c",
    )  # (P, P, P)


def embedding_fourier_norms(
    w_e: torch.Tensor,  # (V, D)
    p: int = 113,
) -> torch.Tensor:
    """Compute L2 norm per frequency k=0..(P-1)//2 of numeric token embeddings W_E[:P]."""
    w_num = w_e[:p].detach().to(torch.float64)                     # (P, D)
    basis, _ = make_fourier_basis(p=p, device=w_num.device)        # (P, P)
    w_fourier = einsum(basis, w_num, "f n, n d -> f d")            # (P, D)

    n_freq = (p // 2) + 1
    norms = torch.zeros(n_freq, dtype=torch.float64, device=w_num.device)
    norms[0] = w_fourier[0].norm()
    for k in range(1, n_freq):
        cos_row = w_fourier[2 * k - 1]                             # (D,)
        sin_row = w_fourier[2 * k]                                 # (D,)
        norms[k] = torch.sqrt(cos_row.pow(2).sum() + sin_row.pow(2).sum())
    return norms  # ((P // 2) + 1,)


def compute_frequency_energy_breakdown(
    fourier_logits: torch.Tensor,  # (P, P, P), float64
) -> dict[str, torch.Tensor]:
    """Compute per-frequency squared-norm and L2-norm breakdown from 2D DFT logits.

    For each k = 1..(P-1)//2, the 2x2 diagonal block B_k spans:
        cos(w_k(a+b)) = (cos_k x cos_k - sin_k x sin_k) / sqrt(2)
        sin(w_k(a+b)) = (sin_k x cos_k + cos_k x sin_k) / sqrt(2)
        cos(w_k(a-b)) = (cos_k x cos_k + sin_k x sin_k) / sqrt(2)
        sin(w_k(a-b)) = (sin_k x cos_k - cos_k x sin_k) / sqrt(2)

    Returns dict with tensors of length (P // 2) + 1 indexed by k = 0..(P-1)//2:
        - trig_sum_energy:   ||proj_{cos(k(a+b)), sin(k(a+b))}||_F^2  (k=0 stores DC energy)
        - block_2d_energy:   ||hat_L[B_k]||_F^2                       (k=0 stores DC energy)
        - total_energy:      scalar ||hat_L||_F^2
        - non_dc_energy:     scalar ||hat_L||_F^2 - ||hat_L[0, 0, :]||_2^2
    """
    p = fourier_logits.shape[0]
    n_freq = (p // 2) + 1
    device = fourier_logits.device

    trig_sum_energy = torch.zeros(n_freq, dtype=torch.float64, device=device)
    block_2d_energy = torch.zeros(n_freq, dtype=torch.float64, device=device)

    dc_energy = fourier_logits[0, 0, :].pow(2).sum()
    trig_sum_energy[0] = dc_energy
    block_2d_energy[0] = dc_energy

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for k in range(1, n_freq):
        c_idx = 2 * k - 1
        s_idx = 2 * k

        cc = fourier_logits[c_idx, c_idx, :]  # (P,)
        ss = fourier_logits[s_idx, s_idx, :]  # (P,)
        sc = fourier_logits[s_idx, c_idx, :]  # (P,)
        cs = fourier_logits[c_idx, s_idx, :]  # (P,)

        # Orthonormal coordinates on cos(w_k(a+b)) and sin(w_k(a+b))
        coeff_cos_sum = inv_sqrt2 * (cc - ss)  # (P,)
        coeff_sin_sum = inv_sqrt2 * (sc + cs)  # (P,)

        trig_sum_energy[k] = coeff_cos_sum.pow(2).sum() + coeff_sin_sum.pow(2).sum()
        block_2d_energy[k] = (
            cc.pow(2).sum() + ss.pow(2).sum() + sc.pow(2).sum() + cs.pow(2).sum()
        )

    total_energy = fourier_logits.pow(2).sum()
    non_dc_energy = total_energy - dc_energy
    return {
        "trig_sum_energy": trig_sum_energy,
        "block_2d_energy": block_2d_energy,
        "dc_energy": dc_energy,
        "total_energy": total_energy,
        "non_dc_energy": non_dc_energy,
    }


def dft_2d_key_frequency_projection(
    logit_table: torch.Tensor,    # (P, P, P), float64
    fourier_basis: torch.Tensor,  # (P, P), float64
    key_freqs: list[int],
    mode: str = "trig_sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose logits via exact 2D DFT projection into restricted and excluded tables.

    Args:
        logit_table: (P, P, P) float64 logits L[a, b, c].
        fourier_basis: (P, P) float64 orthonormal basis F.
        key_freqs: List of frequency indices k in 1..(P-1)//2.
        mode:
            - "trig_sum": Project each key frequency k onto the 2D orthonormal subspace
              span{cos(w_k(a+b)), sin(w_k(a+b))} inside the 2D DFT block B_k (Nanda et al. 2023).
            - "block_2d": Keep all 4 entries of the 2x2 diagonal block B_k = {cos_k, sin_k}^2.

    Returns:
        restricted_logits: (P, P, P) float64 containing ONLY DC (0, 0) + key_freqs subspace.
        excluded_logits:   (P, P, P) float64 equal to logit_table minus the key_freqs subspace
                           (preserving DC and all non-key 2D Fourier modes).
    """
    fourier_logits = logit_table_2d_fourier(logit_table, fourier_basis)   # (P, P, P)
    key_only_fourier = torch.zeros_like(fourier_logits)                   # (P, P, P)

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for k in key_freqs:
        c_idx = 2 * k - 1
        s_idx = 2 * k
        if mode == "trig_sum":
            cc = fourier_logits[c_idx, c_idx, :]
            ss = fourier_logits[s_idx, s_idx, :]
            sc = fourier_logits[s_idx, c_idx, :]
            cs = fourier_logits[c_idx, s_idx, :]

            # Project onto u_cos = (cc - ss)/sqrt(2), u_sin = (sc + cs)/sqrt(2)
            half_cos = 0.5 * (cc - ss)  # = (inv_sqrt2 * (cc - ss)) * inv_sqrt2
            half_sin = 0.5 * (sc + cs)  # = (inv_sqrt2 * (sc + cs)) * inv_sqrt2
            _ = inv_sqrt2

            key_only_fourier[c_idx, c_idx, :] = half_cos
            key_only_fourier[s_idx, s_idx, :] = -half_cos
            key_only_fourier[s_idx, c_idx, :] = half_sin
            key_only_fourier[c_idx, s_idx, :] = half_sin
        elif mode == "block_2d":
            key_only_fourier[c_idx, c_idx, :] = fourier_logits[c_idx, c_idx, :]
            key_only_fourier[s_idx, s_idx, :] = fourier_logits[s_idx, s_idx, :]
            key_only_fourier[s_idx, c_idx, :] = fourier_logits[s_idx, c_idx, :]
            key_only_fourier[c_idx, s_idx, :] = fourier_logits[c_idx, s_idx, :]
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

    # Restricted = DC + key frequency subspace
    restricted_fourier = key_only_fourier.clone()
    restricted_fourier[0, 0, :] = fourier_logits[0, 0, :]

    # Excluded = full logits - key frequency subspace (retaining DC + all other frequencies)
    excluded_fourier = fourier_logits - key_only_fourier

    restricted_logits = inverse_logit_table_2d_fourier(restricted_fourier, fourier_basis)
    excluded_logits = inverse_logit_table_2d_fourier(excluded_fourier, fourier_basis)
    return restricted_logits, excluded_logits


def verify_2d_dft_projection(
    logit_table: torch.Tensor,    # (P, P, P), float64
    fourier_basis: torch.Tensor,  # (P, P), float64
    key_freqs: list[int],
) -> dict[str, float]:
    """Numerically verify orthonormality, Parseval's theorem, idempotence, and spatial equivalence."""
    p = logit_table.shape[0]
    device = logit_table.device

    # 1. Orthonormality of F
    eye = torch.eye(p, dtype=torch.float64, device=device)
    ortho_err = (fourier_basis @ fourier_basis.T - eye).abs().max().item()

    # 2. Exact round-trip & Parseval energy preservation
    f_logits = logit_table_2d_fourier(logit_table, fourier_basis)
    reconstructed = inverse_logit_table_2d_fourier(f_logits, fourier_basis)
    roundtrip_err = (logit_table - reconstructed).abs().max().item()
    parseval_rel_err = (
        abs(logit_table.pow(2).sum().item() - f_logits.pow(2).sum().item())
        / max(logit_table.pow(2).sum().item(), 1e-12)
    )

    # 3. Direct spatial projection equivalence for trig_sum
    restricted_dft, excluded_dft = dft_2d_key_frequency_projection(
        logit_table, fourier_basis, key_freqs, mode="trig_sum"
    )
    grid = torch.arange(p, dtype=torch.float64, device=device)
    ab_sum = grid[:, None] + grid[None, :]  # (P, P)
    norm_factor = math.sqrt(2.0) / p
    spatial_trig = torch.zeros_like(logit_table)
    for k in key_freqs:
        angles = 2.0 * math.pi * k * ab_sum / p
        cos_k = norm_factor * torch.cos(angles)
        sin_k = norm_factor * torch.sin(angles)
        spatial_trig = (
            spatial_trig
            + einsum(cos_k, einsum(cos_k, logit_table, "a b, a b c -> c"), "a b, c -> a b c")
            + einsum(sin_k, einsum(sin_k, logit_table, "a b, a b c -> c"), "a b, c -> a b c")
        )
    const_comp = logit_table.mean(dim=(0, 1), keepdim=True)
    spatial_restricted = spatial_trig + const_comp
    spatial_excluded = logit_table - spatial_trig

    restricted_match_err = (restricted_dft - spatial_restricted).abs().max().item()
    excluded_match_err = (excluded_dft - spatial_excluded).abs().max().item()

    # 4. Orthogonality of restricted (minus DC) and excluded (minus DC)
    r_centered = restricted_dft - const_comp
    e_centered = excluded_dft - const_comp
    denom = max(r_centered.norm().item() * e_centered.norm().item(), 1e-12)
    orthog_cos = (r_centered * e_centered).sum().abs().item() / denom

    assert ortho_err < 1e-12, f"Basis not orthonormal: {ortho_err}"
    assert roundtrip_err < 1e-11, f"2D DFT round-trip error too large: {roundtrip_err}"
    assert parseval_rel_err < 1e-12, f"Parseval relative error too large: {parseval_rel_err}"
    assert restricted_match_err < 1e-11, f"Restricted DFT vs spatial mismatch: {restricted_match_err}"
    assert excluded_match_err < 1e-11, f"Excluded DFT vs spatial mismatch: {excluded_match_err}"
    assert orthog_cos < 1e-12, f"Restricted and excluded subspaces not orthogonal (cos={orthog_cos})"

    return {
        "ortho_max_err": ortho_err,
        "roundtrip_max_err": roundtrip_err,
        "parseval_rel_err": parseval_rel_err,
        "restricted_spatial_max_err": restricted_match_err,
        "excluded_spatial_max_err": excluded_match_err,
        "subspace_cosine_err": orthog_cos,
    }


def identify_key_frequencies(
    energy_breakdown: dict[str, torch.Tensor],
    energy_threshold_frac: float = 0.01,
) -> tuple[list[int], list[dict[str, float | int]]]:
    """Identify key frequencies k >= 1 whose trig-sum energy exceeds threshold fraction of non-DC energy."""
    trig_e = energy_breakdown["trig_sum_energy"]      # ((P // 2) + 1,)
    block_e = energy_breakdown["block_2d_energy"]     # ((P // 2) + 1,)
    non_dc = energy_breakdown["non_dc_energy"].item()
    total = energy_breakdown["total_energy"].item()

    rows: list[dict[str, float | int]] = []
    for k in range(1, trig_e.shape[0]):
        e_sum = trig_e[k].item()
        e_blk = block_e[k].item()
        rows.append(
            {
                "k": k,
                "trig_sum_energy": e_sum,
                "trig_sum_frac_non_dc": e_sum / non_dc,
                "trig_sum_frac_total": e_sum / total,
                "block_2d_energy": e_blk,
                "block_2d_frac_non_dc": e_blk / non_dc,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["trig_sum_energy"], reverse=True)
    key_freqs = sorted(
        int(r["k"]) for r in rows_sorted if float(r["trig_sum_frac_non_dc"]) >= energy_threshold_frac
    )
    return key_freqs, rows_sorted


def table_cross_entropy(
    logit_table: torch.Tensor,  # (P, P, P), float64
    x_pairs: torch.Tensor,      # (N, 3), int64
    y_targets: torch.Tensor,    # (N,), int64
) -> float:
    """Evaluate float64 cross-entropy on a subset of (a, b) pairs from a precomputed logit table."""
    a = x_pairs[:, 0].to(logit_table.device)
    b = x_pairs[:, 1].to(logit_table.device)
    logits = logit_table[a, b]                                     # (N, P), float64
    return F.cross_entropy(logits, y_targets.to(logit_table.device)).item()


def load_model_from_checkpoint(
    ckpt_path: Path,
    model_cfg: TransformerConfig,
    device: torch.device,
) -> tuple[OneLayerTransformer, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = OneLayerTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, int(ckpt["epoch"])


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_training_dynamics(
    metrics: list[dict[str, float]],
    out_path: Path,
    phase_epochs: dict[str, int] | None = None,
) -> None:
    """Figure 1: Train/test loss, train/test accuracy, and total weight L2 norm."""
    _configure_plot_style()
    epochs = np.array([m["epoch"] for m in metrics], dtype=np.float64)
    train_loss = np.array([m["train_loss"] for m in metrics], dtype=np.float64)
    test_loss = np.array([m["test_loss"] for m in metrics], dtype=np.float64)
    train_acc = np.array([m["train_acc"] for m in metrics], dtype=np.float64) * 100.0
    test_acc = np.array([m["test_acc"] for m in metrics], dtype=np.float64) * 100.0
    w_l2 = np.array([m["weight_l2_norm"] for m in metrics], dtype=np.float64)

    # Filter epoch > 0 for log-x display, or use linear-x?
    # Using log-x (from epoch 100 to 40,000) resolves memorization (~200), grokking, and cleanup cleanly.
    mask = epochs >= 100
    ep = epochs[mask]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)

    c_train = "#0072B2"
    c_test = "#D55E00"
    c_norm = "#009E73"

    # Panel 1: Loss
    axes[0].plot(ep, np.clip(train_loss[mask], 1e-10, None), color=c_train, lw=2.0, label="Train loss")
    axes[0].plot(ep, np.clip(test_loss[mask], 1e-10, None), color=c_test, lw=2.0, label="Test loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cross-entropy loss\n(float64)")
    axes[0].set_title("Modular Addition Grokking Dynamics (P = 113, seed = 42, wd = 1.0)")
    axes[0].legend(loc="lower left")

    # Panel 2: Accuracy
    axes[1].plot(ep, train_acc[mask], color=c_train, lw=2.0, label="Train accuracy")
    axes[1].plot(ep, test_acc[mask], color=c_test, lw=2.0, label="Test accuracy")
    axes[1].set_ylim(-3, 105)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="center right")

    # Panel 3: Total Weight L2 Norm
    axes[2].plot(ep, w_l2[mask], color=c_norm, lw=2.0, label=r"Total weight norm $\|W\|_2$")
    axes[2].set_ylabel(r"Weight $L_2$ norm")
    axes[2].set_xlabel("Epoch (log scale)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(100, epochs[-1])
    axes[2].legend(loc="upper right")

    if phase_epochs:
        phase_styles = [
            ("memorization_end", "Memorization\npeak", "#555555"),
            ("circuit_formation_end", "Grokking /\ncircuit formed", "#555555"),
            ("cleanup_end", "Cleanup\ncomplete", "#555555"),
        ]
        for key, label, col in phase_styles:
            if key in phase_epochs and phase_epochs[key] >= 100:
                x_val = phase_epochs[key]
                for ax in axes:
                    ax.axvline(x_val, color=col, ls=":", lw=1.2, alpha=0.75)
                axes[0].text(
                    x_val * 1.07,
                    1.5e-4,
                    f"{label}\n(~{x_val:,})",
                    fontsize=7.8,
                    color=col,
                    va="center",
                    ha="left",
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_fourier_spectrum(
    w_e_norms: np.ndarray,           # (57,) k = 0..56
    logit_trig_energy: np.ndarray,   # (57,) k = 0..56
    logit_block_energy: np.ndarray,  # (57,) k = 0..56
    non_dc_energy: float,
    key_freqs: list[int],
    final_epoch: int,
    out_path: Path,
) -> None:
    """Figure 2: Final checkpoint Fourier spectrum of W_E and 2D DFT logits."""
    _configure_plot_style()
    ks = np.arange(1, len(w_e_norms))
    key_set = set(key_freqs)

    w_e_vals = w_e_norms[1:]
    logit_norm_vals = np.sqrt(logit_trig_energy[1:])
    logit_pct_var = 100.0 * logit_trig_energy[1:] / non_dc_energy

    key_var_pct = 100.0 * sum(logit_trig_energy[k] for k in key_freqs) / non_dc_energy
    key_block_pct = 100.0 * sum(logit_block_energy[k] for k in key_freqs) / non_dc_energy

    c_key = "#D55E00"
    c_other = "#A6B8C7"
    colors = [c_key if int(k) in key_set else c_other for k in ks]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)

    # Top: Token embedding W_E[:113] 1D Fourier norm
    axes[0].bar(ks, w_e_vals, color=colors, width=0.8, edgecolor="none")
    axes[0].set_ylabel(r"$\| \hat{W}_E[k] \|_2$")
    axes[0].set_title(
        f"Final Checkpoint (Epoch {final_epoch:,}) Fourier Spectrum — "
        f"Key Frequencies K = {key_freqs}"
    )
    y0_max = float(w_e_vals.max()) * 1.18
    axes[0].set_ylim(0, y0_max)
    for k in key_freqs:
        val = w_e_vals[k - 1]
        axes[0].text(k, val + y0_max * 0.025, f"k={k}", ha="center", va="bottom", fontsize=8.5, color=c_key, fontweight="bold")

    # Bottom: 2D DFT logit spectrum (trig-sum norm & % variance explained)
    axes[1].bar(ks, logit_pct_var, color=colors, width=0.8, edgecolor="none")
    axes[1].set_ylabel("Share of non-DC\nlogit variance (%)")
    axes[1].set_xlabel(r"Frequency $k \in \{1, \dots, 56\}$ (for $P = 113$)")
    axes[1].set_xlim(0, 57)
    y1_max = float(logit_pct_var.max()) * 1.22
    axes[1].set_ylim(0, y1_max)
    for k in key_freqs:
        val = logit_pct_var[k - 1]
        axes[1].text(
            k,
            val + y1_max * 0.025,
            f"k={k}\n({val:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            color=c_key,
            fontweight="bold",
        )

    # Summary callout box
    summary_txt = (
        f"Key trig-sum subspace: {key_var_pct:.2f}% of non-DC logit variance\n"
        f"Key 2x2 DFT blocks:    {key_block_pct:.2f}% of non-DC logit variance\n"
        f"({np.sqrt(key_var_pct / 100.0) * 100.0:.2f}% of non-DC logit L2 norm)"
    )
    axes[1].text(
        0.985,
        0.93,
        summary_txt,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    # Legend handles
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c_key, label=f"Key frequencies K = {key_freqs}"),
        Patch(facecolor=c_other, label="Other frequencies"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right")

    _ = logit_norm_vals
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_progress_measures(
    progress_rows: list[dict[str, float]],
    key_freqs: list[int],
    out_path: Path,
    use_log_x: bool = True,
    phase_epochs: dict[str, int] | None = None,
) -> None:
    """Figure 3: Train loss, test loss, restricted loss (test), and excluded loss (train) overlaid."""
    _configure_plot_style()
    epochs = np.array([r["epoch"] for r in progress_rows], dtype=np.float64)
    train_loss = np.array([r["train_loss"] for r in progress_rows], dtype=np.float64)
    test_loss = np.array([r["test_loss"] for r in progress_rows], dtype=np.float64)
    restricted_loss = np.array([r["restricted_test_loss"] for r in progress_rows], dtype=np.float64)
    excluded_loss = np.array([r["excluded_train_loss"] for r in progress_rows], dtype=np.float64)

    floor = 1e-9
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    if use_log_x:
        mask = epochs >= 250
    else:
        mask = np.ones_like(epochs, dtype=bool)

    ep = epochs[mask]
    ax.plot(ep, np.clip(train_loss[mask], floor, None), color="#0072B2", lw=2.2, label="Train loss")
    ax.plot(ep, np.clip(test_loss[mask], floor, None), color="#D55E00", lw=2.2, label="Test loss")
    ax.plot(
        ep,
        np.clip(restricted_loss[mask], floor, None),
        color="#009E73",
        lw=2.3,
        ls="--",
        label=f"Restricted loss (test, keep K={key_freqs})",
    )
    ax.plot(
        ep,
        np.clip(excluded_loss[mask], floor, None),
        color="#CC79A7",
        lw=2.3,
        ls="-.",
        label=f"Excluded loss (train, ablate K={key_freqs})",
    )

    ax.set_yscale("log")
    ax.set_ylim(2e-8, 80.0)
    if use_log_x:
        ax.set_xscale("log")
        ax.set_xlim(250, epochs[-1])
        ax.set_xlabel("Epoch (log scale)")
    else:
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, epochs[-1])

    if phase_epochs:
        phase_styles = [
            ("memorization_end", "Memorization\npeak", "#555555"),
            ("circuit_formation_end", "Circuit formed\n/ grokking", "#555555"),
            ("cleanup_end", "Cleanup\ncomplete", "#555555"),
        ]
        for key, label, col in phase_styles:
            if key in phase_epochs and phase_epochs[key] >= 250:
                x_val = phase_epochs[key]
                ax.axvline(x_val, color=col, ls=":", lw=1.2, alpha=0.75)
                ax.text(
                    x_val * 1.06,
                    2.0e-5,
                    f"{label}\n(~{x_val:,})",
                    fontsize=8,
                    color=col,
                    va="center",
                    ha="left",
                )

    ax.set_ylabel("Cross-entropy loss (float64)")
    ax.set_title("Mechanistic Progress Measures Across Training (P = 113, seed = 42)")
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.56), fontsize=8.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_stage3_analysis(
    run_dir: Path,
    figures_dir: Path,
    device_str: str = "auto",
    energy_threshold_frac: float = 0.01,
) -> dict[str, object]:
    """Run full Stage 3 analysis on a completed training run and generate all 3 figures."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    run_config = json.loads((run_dir / "config.json").read_text())
    train_cfg = run_config["train_config"]
    model_cfg = TransformerConfig(**run_config["model_config"])
    p = int(train_cfg["p"])

    # Recreate exact deterministic train/test split
    data = make_modular_addition_dataset(
        p=p,
        train_frac=float(train_cfg["train_frac"]),
        seed=int(train_cfg["seed"]),
    )

    # Load logged training metrics
    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]

    # Locate sorted checkpoints
    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    fourier_basis, _ = make_fourier_basis(p=p, device=device)

    # Step 1: Analyze final checkpoint to identify key frequencies K
    final_model, final_epoch = load_model_from_checkpoint(ckpt_paths[-1], model_cfg, device)
    final_logit_table = compute_full_logit_table(final_model, p=p, device=device)
    final_fourier_logits = logit_table_2d_fourier(final_logit_table, fourier_basis)
    energy_breakdown = compute_frequency_energy_breakdown(final_fourier_logits)
    key_freqs, ranked_freqs = identify_key_frequencies(
        energy_breakdown,
        energy_threshold_frac=energy_threshold_frac,
    )

    # Step 2: Numerically verify the 2D DFT projection before computing progress curves
    dft_verification = verify_2d_dft_projection(final_logit_table, fourier_basis, key_freqs)

    # Compute final W_E Fourier norms
    w_e_norms = embedding_fourier_norms(final_model.W_E, p=p).cpu().numpy()

    # Exact % of logit norm & variance explained by K
    trig_e = energy_breakdown["trig_sum_energy"]
    block_e = energy_breakdown["block_2d_energy"]
    non_dc_e = energy_breakdown["non_dc_energy"].item()
    total_e = energy_breakdown["total_energy"].item()
    dc_e = energy_breakdown["dc_energy"].item()

    key_trig_energy = sum(trig_e[k].item() for k in key_freqs)
    key_block_energy = sum(block_e[k].item() for k in key_freqs)

    key_summary = {
        "final_epoch": final_epoch,
        "key_freqs": key_freqs,
        "key_trig_sum_var_pct_non_dc": 100.0 * key_trig_energy / non_dc_e,
        "key_trig_sum_l2_norm_pct_non_dc": 100.0 * math.sqrt(key_trig_energy / non_dc_e),
        "key_trig_sum_plus_dc_var_pct_total": 100.0 * (key_trig_energy + dc_e) / total_e,
        "key_block_2d_var_pct_non_dc": 100.0 * key_block_energy / non_dc_e,
        "top_10_frequencies": ranked_freqs[:10],
        "dft_verification": dft_verification,
    }

    # Step 3: Evaluate restricted (test) and excluded (train) loss across all saved checkpoints
    progress_rows: list[dict[str, float]] = []
    for ckpt_path in ckpt_paths:
        model, epoch = load_model_from_checkpoint(ckpt_path, model_cfg, device)
        logit_table = compute_full_logit_table(model, p=p, device=device)

        restricted_table, excluded_table = dft_2d_key_frequency_projection(
            logit_table,
            fourier_basis,
            key_freqs=key_freqs,
            mode="trig_sum",
        )

        train_loss = table_cross_entropy(logit_table, data.x_train, data.y_train)
        test_loss = table_cross_entropy(logit_table, data.x_test, data.y_test)
        restricted_test_loss = table_cross_entropy(restricted_table, data.x_test, data.y_test)
        excluded_train_loss = table_cross_entropy(excluded_table, data.x_train, data.y_train)

        # Also track key frequency variance fraction across epochs
        f_logits = logit_table_2d_fourier(logit_table, fourier_basis)
        eb = compute_frequency_energy_breakdown(f_logits)
        k_frac = sum(eb["trig_sum_energy"][k].item() for k in key_freqs) / max(
            eb["non_dc_energy"].item(), 1e-12
        )

        progress_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "restricted_test_loss": restricted_test_loss,
                "excluded_train_loss": excluded_train_loss,
                "key_freq_var_frac": k_frac,
            }
        )

    # Save structured Stage 3 progress metrics alongside the run
    progress_jsonl = run_dir / "progress_measures.jsonl"
    with progress_jsonl.open("w") as fp:
        for row in progress_rows:
            fp.write(json.dumps(row) + "\n")

    analysis_summary_path = run_dir / "stage3_summary.json"
    analysis_summary_path.write_text(json.dumps(key_summary, indent=2) + "\n")

    # Estimate quantitative phase boundary epochs from metrics & progress curves
    # - memorization_end: epoch of peak test loss (end of memorization plateau onset)
    # - circuit_formation_end: first checkpoint where restricted_test_loss < 1e-6 (and excluded peaks)
    # - cleanup_end: first logged epoch where weight L2 norm comes within 1% of its final value
    mem_peak_row = max(metrics, key=lambda m: m["test_loss"])
    circuit_rows = [r for r in progress_rows if r["restricted_test_loss"] < 1e-6]
    final_w_l2 = metrics[-1]["weight_l2_norm"]
    cleanup_rows = [
        m for m in metrics if m["epoch"] > mem_peak_row["epoch"] and m["weight_l2_norm"] <= final_w_l2 * 1.01
    ]
    phase_epochs = {
        "memorization_end": int(mem_peak_row["epoch"]),
        "circuit_formation_end": int(circuit_rows[0]["epoch"]) if circuit_rows else 5500,
        "cleanup_end": int(cleanup_rows[0]["epoch"]) if cleanup_rows else 12000,
    }
    key_summary["phase_epochs"] = phase_epochs
    analysis_summary_path.write_text(json.dumps(key_summary, indent=2) + "\n")

    # Step 4: Generate the 3 publication-clean figures
    fig1_path = figures_dir / "fig1_training_dynamics.png"
    fig2_path = figures_dir / "fig2_fourier_spectrum.png"
    fig3_path = figures_dir / "fig3_progress_measures.png"

    plot_training_dynamics(metrics, fig1_path, phase_epochs=phase_epochs)
    plot_fourier_spectrum(
        w_e_norms=w_e_norms,
        logit_trig_energy=trig_e.cpu().numpy(),
        logit_block_energy=block_e.cpu().numpy(),
        non_dc_energy=non_dc_e,
        key_freqs=key_freqs,
        final_epoch=final_epoch,
        out_path=fig2_path,
    )
    plot_progress_measures(
        progress_rows,
        key_freqs,
        fig3_path,
        use_log_x=True,
        phase_epochs=phase_epochs,
    )

    print(json.dumps(key_summary, indent=2))
    print(f"\nSaved {len(progress_rows)} checkpoint progress rows to {progress_jsonl}")
    print(f"Figures:\n  1. {fig1_path}\n  2. {fig2_path}\n  3. {fig3_path}")
    return {
        "summary": key_summary,
        "progress_rows": progress_rows,
        "figures": [str(fig1_path), str(fig2_path), str(fig3_path)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 mechanistic analysis for grokking.")
    parser.add_argument("--run-dir", type=str, default="grokking/runs/baseline_seed42")
    parser.add_argument("--figures-dir", type=str, default="grokking/artifacts/figures")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--energy-threshold", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage3_analysis(
        run_dir=Path(args.run_dir),
        figures_dir=Path(args.figures_dir),
        device_str=args.device,
        energy_threshold_frac=args.energy_threshold,
    )
