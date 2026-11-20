"""Robustness checks for the Stage 4B sweep claims.

Reads the committed sweep outputs and recomputes three claims with seed treated
as the unit of replication (the 30 runs are 10 seeds x 3 weight decays, not 30
independent samples). Needs only numpy and scipy, not torch.

    python -m grokking.src.robustness
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_sweep(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        "wd": np.array([float(r["wd"]) for r in rows]),
        "seed": np.array([int(r["seed"]) for r in rows]),
        "grok": np.array([float(r["grok_epoch_95"]) for r in rows]),
        "n_key": np.array([float(r["n_key_freqs"]) for r in rows]),
        "n_eff": np.array([float(r["n_eff"]) for r in rows]),
        "resid": np.array([float(r["resid_norm_mean_correct"]) for r in rows]),
    }


def wd_exponent(d: dict[str, np.ndarray]) -> None:
    """Fit log(grok_epoch) = a_seed + beta * log(wd) with seed fixed effects."""
    y, x, seed = np.log(d["grok"]), np.log(d["wd"]), d["seed"]
    seeds = np.unique(seed)
    yd, xd = y.copy(), x.copy()
    for s in seeds:
        m = seed == s
        yd[m] -= y[m].mean()
        xd[m] -= x[m].mean()
    beta = (xd * yd).sum() / (xd * xd).sum()
    resid = yd - beta * xd
    dof = len(y) - len(seeds) - 1
    se = np.sqrt((resid ** 2).sum() / dof / (xd * xd).sum())
    half = stats.t.ppf(0.975, dof) * se
    per_seed = [np.polyfit(x[seed == s], y[seed == s], 1)[0] for s in seeds]
    print("1. grok_epoch_95 ~ wd^beta, seed fixed effects")
    print(f"   beta = {beta:.3f}, 95% CI [{beta - half:.3f}, {beta + half:.3f}] (t, dof={dof})")
    print(f"   per-seed slopes: {np.round(per_seed, 2).tolist()}")


def clustered_spearman(d: dict[str, np.ndarray]) -> None:
    """Compare pooled Spearman (n=30) with one point per seed (n=10)."""
    seeds = np.unique(d["seed"])
    resid_seed = [d["resid"][d["seed"] == s].mean() for s in seeds]
    print("2. Sparsity vs residual: pooled runs vs one point per seed")
    for name in ("n_key", "n_eff"):
        pooled = stats.spearmanr(d[name], d["resid"])
        per_seed = stats.spearmanr([d[name][d["seed"] == s].mean() for s in seeds], resid_seed)
        print(f"   {name:5s} pooled n={len(d[name])}: rho={pooled.statistic:+.3f}, p={pooled.pvalue:.2g}"
              f" | per seed n={len(seeds)}: rho={per_seed.statistic:+.3f}, p={per_seed.pvalue:.2g}")


def alias_breakdown(record_path: Path, summary_path: Path) -> None:
    """Show where seed-42 errors fell against the predicted alias offsets."""
    print("3. Seed 42 alias predictions vs observed error offsets")
    if record_path.exists():
        rec = json.loads(record_path.read_text())
        pred = [p["delta"] for p in rec["top3_predicted_deltas"]]
        print(f"   wd=0.5 predicted top-3 (amplitude weights): +/-{pred}")
        print(f"   wd=0.5 observed offsets (mod 113) and counts: {rec['error_offset_histogram']}")
    if summary_path.exists():
        summ = json.loads(summary_path.read_text())["final_epoch_alias_pred_final_k"]
        l2 = [(p["delta"], round(p["score"], 3)) for p in summ["top3_l2"]]
        var = [(p["delta"], round(p["score"], 3)) for p in summ["top3_var"]]
        print(f"   wd=1.0 top-3, amplitude weights sqrt(E_k): {l2}")
        print(f"   wd=1.0 top-3, variance weights E_k:        {var}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", default="grokking/runs/sweep_wd/aggregate_results.csv")
    parser.add_argument("--seed42-wd05", default="grokking/runs/seed42_wd_check/wd_0.5/seed_42/sweep_record.json")
    parser.add_argument("--baseline-summary", default="grokking/runs/baseline_seed42/aliasing_diag_summary.json")
    args = parser.parse_args()

    d = load_sweep(Path(args.sweep_csv))
    wd_exponent(d)
    clustered_spearman(d)
    alias_breakdown(Path(args.seed42_wd05), Path(args.baseline_summary))


if __name__ == "__main__":
    main()
