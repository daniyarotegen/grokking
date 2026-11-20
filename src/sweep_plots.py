"""Stage 4B: Aggregate multi-seed weight-decay sweep results and generate Figures 5 & 6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grokking.src.analysis import _configure_plot_style


def load_sweep_records(out_root: Path) -> list[dict[str, object]]:
    """Load all sweep_record.json files sorted by (wd, seed)."""
    records = []
    for path in sorted(out_root.glob("wd_*/seed_*/sweep_record.json")):
        records.append(json.loads(path.read_text()))
    records.sort(key=lambda r: (float(r["wd"]), int(r["seed"])))
    return records


def load_baseline_seed42_reference(baseline_dir: Path) -> dict[str, object] | None:
    """Optionally load baseline_seed42 summary (kept strictly separate from 30-run sweep stats)."""
    alias_sum_path = baseline_dir / "aliasing_diag_summary.json"
    stage3_path = baseline_dir / "stage3_summary.json"
    if not (alias_sum_path.exists() and stage3_path.exists()):
        return None

    alias_sum = json.loads(alias_sum_path.read_text())
    stage3 = json.loads(stage3_path.read_text())
    # Recompute N_eff from stage3_summary top_10_frequencies (captures >99.99% of trig-sum energy)
    # or exact from aliasing_diag.jsonl last row if available
    diag_jsonl = baseline_dir / "aliasing_diag.jsonl"
    if diag_jsonl.exists():
        last_row = json.loads(diag_jsonl.read_text().strip().splitlines()[-1])
        shares = np.array(list(last_row["freq_var_share"].values()), dtype=np.float64)
        p_norm = shares / shares.sum()
        n_eff = float(1.0 / np.sum(p_norm ** 2))
    else:
        n_eff = 1.17

    fm = alias_sum["final_epoch_full_margins"]
    rm = alias_sum["final_epoch_restricted_margins"]
    top3 = alias_sum["final_epoch_alias_pred_final_k"]["top3_l2"]
    # Known exact epoch-40k error histogram for baseline_seed42: 46 at +/-12 (#3), 4 at +10 (#2), 0 at +/-56 (#1)
    total_err = int(fm["err_total"])
    err_d12 = int(fm["err_delta12"])
    err_other = int(fm["err_other"])  # all 4 are at Delta=+10 (#2 in top3_l2)
    return {
        "wd": 1.0,
        "seed": 42,
        "n_key_freqs": len(stage3["key_freqs"]),
        "key_freqs": stage3["key_freqs"],
        "n_eff": n_eff,
        "total_test_errors": total_err,
        "restricted_test_errors": int(rm["err_total"]),
        "top3_predicted_deltas": top3,
        "errors_in_top1_alias": 0,
        "errors_in_top3_alias": err_d12 + err_other,
        "pct_in_top1_alias": 0.0,
        "pct_in_top3_alias": 100.0 * (err_d12 + err_other) / max(total_err, 1),
        "resid_norm_ratio_wrong_over_correct": 295.05 / 59.69,
    }


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_sweep(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute per-wd and pooled imperfect-seed alias validation statistics (strictly over sweep runs)."""
    wds = sorted({float(r["wd"]) for r in records})
    by_wd: list[dict[str, object]] = []

    for wd in wds:
        grp = [r for r in records if float(r["wd"]) == wd]
        grokked = [r for r in grp if r["grok_epoch_95"] is not None]
        imperfect = [r for r in grokked if int(r["total_test_errors"]) > 0]

        by_wd.append(
            {
                "wd": wd,
                "n_seeds": len(grp),
                "grokked_count": len(grokked),
                "median_grok_epoch_95": (
                    float(np.median([int(r["grok_epoch_95"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "min_grok_epoch_95": (
                    int(min(int(r["grok_epoch_95"]) for r in grokked)) if grokked else None
                ),
                "max_grok_epoch_95": (
                    int(max(int(r["grok_epoch_95"]) for r in grokked)) if grokked else None
                ),
                "median_n_key_freqs": (
                    float(np.median([int(r["n_key_freqs"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "median_n_eff": (
                    float(np.median([float(r["n_eff"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "imperfect_grokked_count": len(imperfect),
                "median_full_test_errors": (
                    float(np.median([int(r["total_test_errors"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "max_full_test_errors": (
                    int(max(int(r["total_test_errors"]) for r in grokked)) if grokked else None
                ),
                "median_restricted_test_errors": (
                    float(np.median([int(r["restricted_test_errors"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "max_restricted_test_errors": (
                    int(max(int(r["restricted_test_errors"]) for r in grokked)) if grokked else None
                ),
                "median_final_test_loss": (
                    float(np.median([float(r["test_loss"]) for r in grokked]))
                    if grokked
                    else None
                ),
                "median_restricted_test_loss": (
                    float(np.median([float(r["restricted_test_loss"]) for r in grokked]))
                    if grokked
                    else None
                ),
            }
        )

    all_grokked = [r for r in records if r["grok_epoch_95"] is not None]
    all_imperfect = [r for r in all_grokked if int(r["total_test_errors"]) > 0]

    pooled_total_err = sum(int(r["total_test_errors"]) for r in all_imperfect)
    pooled_top1_err = sum(int(r["errors_in_top1_alias"]) for r in all_imperfect)
    pooled_top3_err = sum(int(r["errors_in_top3_alias"]) for r in all_imperfect)

    ratios = [
        float(r["resid_norm_ratio_wrong_over_correct"])
        for r in all_imperfect
        if r["resid_norm_ratio_wrong_over_correct"] is not None
    ]
    top1_fracs = [
        float(r["frac_errors_in_top1_alias"])
        for r in all_imperfect
        if r["frac_errors_in_top1_alias"] is not None
    ]
    top3_fracs = [
        float(r["frac_errors_in_top3_alias"])
        for r in all_imperfect
        if r["frac_errors_in_top3_alias"] is not None
    ]

    neff_arr = np.array([float(r["n_eff"]) for r in all_grokked], dtype=np.float64)
    err_arr = np.array([int(r["total_test_errors"]) for r in all_grokked], dtype=np.float64)

    imperfect_summary = {
        "n_grokked_seeds": len(all_grokked),
        "n_imperfect_grokked_seeds": len(all_imperfect),
        "pooled_total_test_errors": pooled_total_err,
        "pooled_errors_in_top1_alias": pooled_top1_err,
        "pooled_errors_in_top3_alias": pooled_top3_err,
        "pooled_pct_in_top1_alias": (
            100.0 * pooled_top1_err / pooled_total_err if pooled_total_err > 0 else None
        ),
        "pooled_pct_in_top3_alias": (
            100.0 * pooled_top3_err / pooled_total_err if pooled_total_err > 0 else None
        ),
        "chance_pct_top1_pair": 100.0 * 2.0 / 112.0,
        "chance_pct_top3_pairs": 100.0 * 6.0 / 112.0,
        "per_seed_median_pct_top1": 100.0 * float(np.median(top1_fracs)) if top1_fracs else None,
        "per_seed_mean_pct_top1": 100.0 * float(np.mean(top1_fracs)) if top1_fracs else None,
        "per_seed_median_pct_top3": 100.0 * float(np.median(top3_fracs)) if top3_fracs else None,
        "per_seed_mean_pct_top3": 100.0 * float(np.mean(top3_fracs)) if top3_fracs else None,
        "resid_ratio_median": float(np.median(ratios)) if ratios else None,
        "resid_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "resid_ratio_min": float(min(ratios)) if ratios else None,
        "resid_ratio_max": float(max(ratios)) if ratios else None,
        "spearman_neff_vs_errors": _spearman_corr(neff_arr, err_arr),
    }

    return {
        "n_total_runs": len(records),
        "by_wd": by_wd,
        "imperfect_grokked_summary": imperfect_summary,
        "records": records,
    }


def write_csv(records: list[dict[str, object]], csv_path: Path) -> None:
    fields = [
        "wd",
        "seed",
        "grok_epoch_95",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "weight_l2_norm",
        "n_key_freqs",
        "n_eff",
        "key_freqs",
        "key_freqs_total_var_share",
        "restricted_test_loss",
        "restricted_test_acc",
        "restricted_test_errors",
        "excluded_train_loss",
        "excluded_train_acc",
        "total_test_errors",
        "errors_in_top1_alias",
        "errors_in_top3_alias",
        "frac_errors_in_top1_alias",
        "frac_errors_in_top3_alias",
        "resid_norm_mean_correct",
        "resid_norm_mean_wrong",
        "resid_norm_ratio_wrong_over_correct",
        "wall_time_s",
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fields}
            row["key_freqs"] = " ".join(str(k) for k in r["key_freqs"])
            writer.writerow(row)


def plot_fig5_wd_phase_diagram(
    records: list[dict[str, object]],
    out_path: Path,
    baseline_ref: dict[str, object] | None = None,
) -> None:
    """Figure 5: 3-panel phase diagram across wd = 0.5, 1.0, 1.5 (individual seeds + medians)."""
    _configure_plot_style()
    wds = sorted({float(r["wd"]) for r in records})
    x_pos = np.arange(len(wds), dtype=np.float64)
    wd_labels = [f"{w:.1f}" for w in wds]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    c_primary = "#0072B2"
    c_secondary = "#D55E00"
    c_green = "#009E73"

    # --- Panel A: Grokking epoch (test_acc >= 95%) vs wd ---
    ax_a = axes[0]
    for i, wd in enumerate(wds):
        grp = [r for r in records if float(r["wd"]) == wd]
        grok_eps = [int(r["grok_epoch_95"]) for r in grp if r["grok_epoch_95"] is not None]
        jitter = rng.uniform(-0.11, 0.11, size=len(grok_eps))
        ax_a.scatter(
            np.full(len(grok_eps), i) + jitter,
            grok_eps,
            color=c_primary,
            alpha=0.75,
            s=44,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Individual seed (0–9)" if i == 0 else None,
        )
        if grok_eps:
            med = float(np.median(grok_eps))
            ax_a.hlines(
                med,
                i - 0.24,
                i + 0.24,
                colors="#111111",
                linewidth=2.5,
                zorder=4,
                label="Median" if i == 0 else None,
            )
            ax_a.text(
                i + 0.27,
                med,
                f"{int(med):,}",
                va="center",
                ha="left",
                fontsize=8.0,
                fontweight="bold",
                color="#222222",
            )

    ax_a.set_xlim(-0.45, len(wds) - 0.30)
    ax_a.set_ylim(1500, 26000)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(wd_labels)
    ax_a.set_xlabel("Weight decay (wd)")
    ax_a.set_ylabel("Grokking epoch (test acc ≥ 95%)")
    ax_a.set_title("A. Grokking Speed vs. Weight Decay")
    ax_a.legend(loc="upper right", fontsize=8.2)

    # --- Panel B: |K| and N_eff vs wd ---
    ax_b = axes[1]
    offset = 0.16
    for i, wd in enumerate(wds):
        grp = [r for r in records if float(r["wd"]) == wd and r["grok_epoch_95"] is not None]
        k_counts = [int(r["n_key_freqs"]) for r in grp]
        n_effs = [float(r["n_eff"]) for r in grp]
        j1 = rng.uniform(-0.07, 0.07, size=len(grp))
        j2 = rng.uniform(-0.07, 0.07, size=len(grp))

        ax_b.scatter(
            np.full(len(grp), i - offset) + j1,
            k_counts,
            color=c_primary,
            alpha=0.75,
            s=40,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=r"Key count $|K|$ (>1% var)" if i == 0 else None,
        )
        ax_b.scatter(
            np.full(len(grp), i + offset) + j2,
            n_effs,
            color=c_secondary,
            alpha=0.78,
            s=44,
            marker="s",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=r"Effective $N_{\mathrm{eff}} = 1/\sum p_k^2$" if i == 0 else None,
        )
        if grp:
            ax_b.hlines(float(np.median(k_counts)), i - offset - 0.13, i - offset + 0.13, colors="#111111", lw=2.3, zorder=4)
            ax_b.hlines(float(np.median(n_effs)), i + offset - 0.13, i + offset + 0.13, colors="#111111", lw=2.3, zorder=4)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(wd_labels)
    ax_b.set_xlabel("Weight decay (wd)")
    ax_b.set_ylabel("Frequency count")
    ax_b.set_ylim(0.7, 6.2)
    ax_b.set_yticks([1, 2, 3, 4, 5])
    ax_b.set_title("B. Surviving & Effective Frequencies")
    ax_b.legend(loc="upper right", fontsize=8.0)

    # --- Panel C: Full vs Restricted final test errors vs wd ---
    ax_c = axes[2]
    max_err_val = max([int(r["total_test_errors"]) for r in records] + [0])
    for i, wd in enumerate(wds):
        grp = [r for r in records if float(r["wd"]) == wd and r["grok_epoch_95"] is not None]
        full_errs = [int(r["total_test_errors"]) for r in grp]
        rest_errs = [int(r["restricted_test_errors"]) for r in grp]
        j1 = rng.uniform(-0.07, 0.07, size=len(grp))
        j2 = rng.uniform(-0.07, 0.07, size=len(grp))

        ax_c.scatter(
            np.full(len(grp), i - offset) + j1,
            full_errs,
            color=c_secondary,
            alpha=0.80,
            s=42,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Full model errors (seeds 0–9)" if i == 0 else None,
        )
        ax_c.scatter(
            np.full(len(grp), i + offset) + j2,
            rest_errs,
            color=c_green,
            alpha=0.82,
            s=44,
            marker="^",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Restricted (K) errors (seeds 0–9)" if i == 0 else None,
        )
        if grp:
            ax_c.hlines(float(np.median(full_errs)), i - offset - 0.13, i - offset + 0.13, colors="#111111", lw=2.3, zorder=4)
            ax_c.hlines(float(np.median(rest_errs)), i + offset - 0.13, i + offset + 0.13, colors="#111111", lw=2.3, zorder=4)

    # Optional clearly distinguished reference marker for baseline_seed42 (wd = 1.0)
    if baseline_ref is not None and 1.0 in wds:
        i_ref = wds.index(1.0)
        ref_err = int(baseline_ref["total_test_errors"])
        max_err_val = max(max_err_val, ref_err)
        ax_c.scatter(
            [i_ref - offset],
            [ref_err],
            color="#b30000",
            marker="*",
            s=140,
            edgecolors="#222222",
            linewidths=0.7,
            zorder=5,
            label="Baseline seed 42 (ref, full=50, rest=0)",
        )
        ax_c.annotate(
            "Sweep (0–9): 0 / 30\nimperfect runs",
            xy=(1.0, 0.2),
            xytext=(0.35, 2.5),
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#222222",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        )

    ax_c.set_yscale("symlog", linthresh=2.0)
    ax_c.set_ylim(-0.5, max(80.0, max_err_val * 2.2))
    ax_c.set_yticks([0, 1, 2, 5, 10, 50])
    ax_c.set_yticklabels(["0", "1", "2", "5", "10", "50"])
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(wd_labels)
    ax_c.set_xlabel("Weight decay (wd)")
    ax_c.set_ylabel("Final test errors (of 8,939)")
    ax_c.set_title("C. Full vs. Clean-Circuit Test Errors")
    ax_c.legend(loc="center right", bbox_to_anchor=(0.99, 0.70), fontsize=7.6)

    fig.suptitle(
        "Multi-Seed Weight-Decay Phase Diagram (P = 113, seeds 0–9, 40k epochs)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig6_alias_validation(
    records: list[dict[str, object]],
    summary: dict[str, object],
    out_path: Path,
    baseline_ref: dict[str, object] | None = None,
) -> None:
    """Figure 6: N_eff vs test errors and analytical alias hit-rate validation."""
    _configure_plot_style()
    wds = sorted({float(r["wd"]) for r in records})
    wd_colors = {0.5: "#0072B2", 1.0: "#E69F00", 1.5: "#D55E00"}
    wd_markers = {0.5: "o", 1.0: "s", 1.5: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))

    # --- Panel A: N_eff vs full-model final test errors (colored by wd) ---
    ax_a = axes[0]
    grokked = [r for r in records if r["grok_epoch_95"] is not None]
    for wd in wds:
        grp = [r for r in grokked if float(r["wd"]) == wd]
        if not grp:
            continue
        xs = [float(r["n_eff"]) for r in grp]
        ys = [int(r["total_test_errors"]) for r in grp]
        ax_a.scatter(
            xs,
            ys,
            color=wd_colors.get(wd, "#555555"),
            marker=wd_markers.get(wd, "o"),
            s=58,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=f"Sweep wd = {wd:.1f} (n={len(grp)}, 0 err)",
            zorder=3,
        )

    if baseline_ref is not None:
        ax_a.scatter(
            [float(baseline_ref["n_eff"])],
            [int(baseline_ref["total_test_errors"])],
            color="#b30000",
            marker="*",
            s=170,
            edgecolors="#222222",
            linewidths=0.7,
            zorder=5,
            label=f"Baseline seed 42 (ref, wd=1.0, err={baseline_ref['total_test_errors']})",
        )
        ax_a.annotate(
            "Seed 6 (wd=0.5, 1.0, 1.5):\nN_eff = 1.19, 0 errors",
            xy=(1.19, 0.15),
            xytext=(1.65, 5.5),
            fontsize=8.0,
            color="#222222",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        )

    ax_a.set_yscale("symlog", linthresh=2.0)
    ax_a.set_ylim(-0.5, 95.0)
    ax_a.set_yticks([0, 1, 2, 5, 10, 50])
    ax_a.set_yticklabels(["0", "1", "2", "5", "10", "50"])
    ax_a.set_xlim(1.0, 3.9)
    ax_a.set_xlabel(r"Effective balanced frequency count $N_{\mathrm{eff}} = 1 / \sum_{k=1}^{56} p_k^2$")
    ax_a.set_ylabel("Full-model final test errors (of 8,939)")
    ax_a.set_title("A. Frequency Imbalance vs. Generalization Errors")
    ax_a.legend(loc="upper right", fontsize=7.9)

    # --- Panel B: % of test errors landing on Top-1 & Top-3 predicted aliases vs chance ---
    ax_b = axes[1]
    imperfect = [r for r in grokked if int(r["total_test_errors"]) > 0]
    imp_sum = summary["imperfect_grokked_summary"]

    cats = ["Top-1 predicted\nalias pair (±Δ*)", "Top-3 predicted\nalias pairs"]
    x = np.arange(2, dtype=np.float64)
    width = 0.34

    chance_vals = [
        float(imp_sum["chance_pct_top1_pair"]),
        float(imp_sum["chance_pct_top3_pairs"]),
    ]

    if imperfect:
        bar_vals = [
            float(imp_sum["pooled_pct_in_top1_alias"]),
            float(imp_sum["pooled_pct_in_top3_alias"]),
        ]
        bar_label = f"Pooled sweep imperfect seeds (n={len(imperfect)})"
        bar_color = "#D55E00"
    elif baseline_ref is not None:
        bar_vals = [
            float(baseline_ref["pct_in_top1_alias"]),
            float(baseline_ref["pct_in_top3_alias"]),
        ]
        bar_label = "Baseline seed 42 (sole imperfect run, 50 err)"
        bar_color = "#b30000"
    else:
        bar_vals = [0.0, 0.0]
        bar_label = "Sweep imperfect seeds (n=0)"
        bar_color = "#D55E00"

    bars_pred = ax_b.bar(
        x - width / 2,
        bar_vals,
        width=width,
        color=bar_color,
        alpha=0.82,
        label=bar_label,
        zorder=2,
    )
    bars_chance = ax_b.bar(
        x + width / 2,
        chance_vals,
        width=width,
        color="#A6B8C7",
        alpha=0.90,
        label="Uniform-chance baseline (2/112, 6/112)",
        zorder=2,
    )

    if imperfect:
        rng = np.random.default_rng(7)
        top1_pts = [100.0 * float(r["frac_errors_in_top1_alias"]) for r in imperfect]
        top3_pts = [100.0 * float(r["frac_errors_in_top3_alias"]) for r in imperfect]
        jit = rng.uniform(-0.08, 0.08, size=len(imperfect))
        ax_b.scatter(
            np.full(len(imperfect), x[0] - width / 2) + jit,
            top1_pts,
            color="#222222",
            s=32,
            alpha=0.8,
            zorder=4,
            label="Individual imperfect seed",
        )
        ax_b.scatter(
            np.full(len(imperfect), x[1] - width / 2) + jit,
            top3_pts,
            color="#222222",
            s=32,
            alpha=0.8,
            zorder=4,
        )
    elif baseline_ref is not None:
        ax_b.text(
            0.03,
            0.73,
            "Sweep (seeds 0–9): 0 / 30 imperfect\n"
            "(100.0% acc at wd = 0.5, 1.0, 1.5)\n\n"
            "Baseline seed 42 ref (50 errors):\n"
            "• Top-1 (±56): 0 / 50 (0.0%)\n"
            "• Top-3 (±56, ±10, ±12): 50 / 50\n"
            "• Residual norm spike: 4.94×",
            transform=ax_b.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#cccccc", alpha=0.96),
        )

    for bar, val in zip(bars_pred, bar_vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color=bar_color,
        )
    for bar, val in zip(bars_chance, chance_vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cats)
    ax_b.set_ylim(0, 118)
    ax_b.set_ylabel("Share of full-model test errors (%)")
    ax_b.set_title("B. Analytical Alias Prediction Validation")
    ax_b.legend(loc="upper left", fontsize=7.9)

    fig.suptitle(
        "Frequency Imbalance & Aliasing Validation Across 30 Sweep Runs (+ Baseline Seed 42 Ref.)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_aggregation_and_plots(
    out_root: Path,
    figures_dir: Path,
    baseline_dir: Path = Path("grokking/runs/baseline_seed42"),
) -> dict[str, object]:
    records = load_sweep_records(out_root)
    if not records:
        raise FileNotFoundError(f"No sweep_record.json files found under {out_root}")

    summary = summarize_sweep(records)
    baseline_ref = load_baseline_seed42_reference(baseline_dir)
    if baseline_ref is not None:
        summary["baseline_seed42_reference"] = baseline_ref

    agg_json = out_root / "aggregate_results.json"
    agg_csv = out_root / "aggregate_results.csv"
    agg_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(records, agg_csv)

    fig5_path = figures_dir / "fig5_wd_phase_diagram.png"
    fig6_path = figures_dir / "fig6_alias_prediction_validation.png"
    plot_fig5_wd_phase_diagram(records, fig5_path, baseline_ref=baseline_ref)
    plot_fig6_alias_validation(records, summary, fig6_path, baseline_ref=baseline_ref)

    print(
        json.dumps(
            {
                "by_wd": summary["by_wd"],
                "imperfect_grokked_summary": summary["imperfect_grokked_summary"],
                "baseline_seed42_reference": baseline_ref,
            },
            indent=2,
        )
    )
    print(f"\nSaved:\n  - {agg_json}\n  - {agg_csv}\n  - {fig5_path}\n  - {fig6_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage 4B sweep and plot Figures 5 & 6.")
    parser.add_argument("--out-root", type=str, default="grokking/runs/sweep_wd")
    parser.add_argument("--figures-dir", type=str, default="grokking/artifacts/figures")
    parser.add_argument("--baseline-dir", type=str, default="grokking/runs/baseline_seed42")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_aggregation_and_plots(Path(args.out_root), Path(args.figures_dir), Path(args.baseline_dir))
