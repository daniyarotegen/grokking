"""Stage 5 analysis: score the weight-decay experiments against frozen predictions.

    python -m grokking.src.wd_analysis --freeze-predictions   # once, before any run exists
    python -m grokking.src.wd_analysis                        # after (or during) the runs

Freezing writes prereg/wd_predictions.json from the Stage 4B sweep (seeds 0-4) and refuses
to overwrite an existing file. The analysis reads that file, so predictions cannot drift
after the data exist. It works on partial results: missing runs are reported, not imputed.
Needs only numpy and scipy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

BETA_WD = -0.964          # seed fixed-effects estimate from Stage 4B (src/robustness.py)
BASE_LR = 1e-3
ALARM_DROP = 0.10         # weight-norm alarm: norm falls 10% below its running maximum


# ---------------------------------------------------------------- frozen predictions

def freeze_predictions(sweep_csv: Path, out_path: Path) -> None:
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; predictions are frozen, not rewritten.")
    with sweep_csv.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["seed"]) < 5]
    a_seed = {}
    for s in range(5):
        r = [x for x in rows if int(x["seed"]) == s]
        a_seed[s] = float(np.mean([np.log(float(x["grok_epoch_95"])) - BETA_WD * np.log(float(x["wd"]))
                                   for x in r]))

    def t_pred(s: int, wd: float, lr: float) -> float:
        # 1/wd law from Stage 4B, plus the untested assumption that the clock is lr * wd.
        return float(np.exp(a_seed[s] + BETA_WD * np.log(wd)) * (BASE_LR / lr))

    grid = {}
    for lr, wd in itertools.product((5e-4, 1e-3, 2e-3), (0.5, 1.0, 2.0)):
        grid[f"lr{lr:g}_wd{wd:g}"] = {str(s): t_pred(s, wd, lr) for s in range(5)}
    for wd in (0.05, 0.1, 0.2, 3.0, 5.0):
        grid[f"lr{BASE_LR:g}_wd{wd:g}"] = {str(s): t_pred(s, wd, BASE_LR) for s in range(5)}

    out = {
        "source": str(sweep_csv),
        "beta_wd": BETA_WD,
        "a_seed": {str(k): v for k, v in a_seed.items()},
        "predicted_grok_epoch_95": grid,
        "E1_predicted_slope_epochs_per_ln_alpha": 1.0 / (BASE_LR * 1.0),
        "alarm_rule": f"first logged epoch where weight_l2_norm <= {1 - ALARM_DROP:.2f} * running max",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"froze predictions -> {out_path}")


# ---------------------------------------------------------------- loading

def load_runs(root: Path) -> list[dict]:
    runs = []
    for rec_path in sorted(root.glob("*/sweep_record.json")):
        rec = json.loads(rec_path.read_text())
        metrics = [json.loads(line) for line in (rec_path.parent / "metrics.jsonl").open()]
        ep = np.array([m["epoch"] for m in metrics])
        wn = np.array([m["weight_l2_norm"] for m in metrics])
        hit = np.where(wn <= (1 - ALARM_DROP) * np.maximum.accumulate(wn))[0]
        rec["alarm_epoch"] = int(ep[hit[0]]) if len(hit) else None
        rec["grokked"] = rec["grok_epoch_95"] is not None
        runs.append(rec)
    return runs


def select(runs, **kw):
    return [r for r in runs if all(np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
                                   for k, v in kw.items())]


def fe_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with one intercept per group (demeaning). Returns coefficients, 95% half-widths, dof."""
    yd, Xd = y.astype(float).copy(), X.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        yd[m] -= yd[m].mean()
        Xd[m] -= Xd[m].mean(axis=0)
    coef, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    dof = len(y) - len(np.unique(groups)) - X.shape[1]
    sigma2 = ((yd - Xd @ coef) ** 2).sum() / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xd.T @ Xd)))
    return coef, stats.t.ppf(0.975, dof) * se, dof


# ---------------------------------------------------------------- tests

def h0_sanity(runs, sweep_root: Path) -> None:
    print("\nH0  Reproduces Stage 4B (lr 1e-3, wd 1.0, alpha 1, init = split = seed)")
    for s in range(5):
        new = select(runs, lr=1e-3, wd=1.0, init_scale=1.0, init_seed=s, split_seed=s, epochs=60_000)
        old_path = sweep_root / "wd_1.0" / f"seed_{s}" / "sweep_record.json"
        if not new or not old_path.exists():
            print(f"   seed {s}: missing")
            continue
        old, t = json.loads(old_path.read_text())["grok_epoch_95"], new[0]["grok_epoch_95"]
        ok = t is not None and old - 500 < t <= old
        print(f"   seed {s}: old {old} (500-ep logging), new {t} (100-ep) -> {'consistent' if ok else 'MISMATCH'}")


def h1_lr_wd(runs, pred) -> None:
    print("\nH1  E2: is the clock lr x wd?  log t = a_seed + b_lr log lr + b_wd log wd")
    e2 = [r for r in runs if r["init_scale"] == 1.0 and r["init_seed"] == r["split_seed"]
          and r["lr"] in (5e-4, 1e-3, 2e-3) and r["wd"] in (0.5, 1.0, 2.0) and r["epochs"] == 60_000]
    g = [r for r in e2 if r["grokked"]]
    print(f"   {len(e2)}/45 runs present, {len(g)} grokked")
    if len(g) < 10:
        return
    y = np.log([r["grok_epoch_95"] for r in g])
    X = np.column_stack([np.log([r["lr"] for r in g]), np.log([r["wd"] for r in g])])
    coef, half, dof = fe_ols(y, X, np.array([r["init_seed"] for r in g]))
    for name, c, h in zip(("b_lr", "b_wd"), coef, half):
        print(f"   {name} = {c:+.3f}, 95% CI [{c - h:+.3f}, {c + h:+.3f}]  (prediction -1; dof {dof})")
    print("   Equal lr*wd = 1e-3 cells, grok epoch by seed:")
    for lr, wd in ((5e-4, 2.0), (1e-3, 1.0), (2e-3, 0.5)):
        cell = {r["init_seed"]: r["grok_epoch_95"] for r in e2 if np.isclose(r["lr"], lr) and np.isclose(r["wd"], wd)}
        print(f"     lr {lr:g}, wd {wd:g}: " + " ".join(f"{cell.get(s, '-'):>6}" for s in range(5)))
    errs = [np.log(r["grok_epoch_95"] / pred[f"lr{r['lr']:g}_wd{r['wd']:g}"][str(r["init_seed"])]) for r in g]
    print(f"   per-seed forecast error |log(observed/predicted)|: median {np.median(np.abs(errs)):.2f} "
          f"(0.10 = 10% off)")


def h2_init_scale(runs, pred) -> None:
    print("\nH2  E1: does grokking time rise with ln(alpha)?  t = a_seed + slope * ln(alpha)")
    e1 = [r for r in runs if np.isclose(r["lr"], 1e-3) and r["wd"] == 1.0 and r["init_seed"] == r["split_seed"]
          and r["init_scale"] in (0.5, 1.0, 2.0, 4.0) and r["epochs"] == 60_000]
    g = [r for r in e1 if r["grokked"]]
    print(f"   {len(e1)}/20 runs present, {len(g)} grokked")
    for a in (0.5, 1.0, 2.0, 4.0):
        cell = {r["init_seed"]: r["grok_epoch_95"] for r in e1 if r["init_scale"] == a}
        print(f"     alpha {a:g}: " + " ".join(f"{cell.get(s, '-'):>6}" for s in range(5)))
    if len(g) < 8:
        return
    coef, half, dof = fe_ols(np.array([r["grok_epoch_95"] for r in g], float),
                             np.log([[r["init_scale"]] for r in g]),
                             np.array([r["init_seed"] for r in g]))
    print(f"   slope = {coef[0]:.0f} epochs per unit ln(alpha), 95% CI [{coef[0] - half[0]:.0f}, "
          f"{coef[0] + half[0]:.0f}]  (direction > 0; Lyu leading order {pred:.0f})")


def h3_range(runs, pred) -> None:
    print("\nH3  E3: does 1/wd extrapolate?  (budget 120k epochs; '-' = did not grok)")
    scored: list[bool] = []
    for wd in (0.0, 0.05, 0.1, 0.2, 3.0, 5.0):
        cell = {r["init_seed"]: r for r in runs if np.isclose(r["lr"], 1e-3) and np.isclose(r["wd"], wd)
                and r["epochs"] == 120_000 and r["init_scale"] == 1.0}
        key = f"lr0.001_wd{wd:g}"
        line = []
        for s in range(5):
            r = cell.get(s)
            obs = "  n/a " if r is None else (f"{r['grok_epoch_95']:>6}" if r["grokked"] else "     -")
            p = pred.get(key, {}).get(str(s))
            line.append((f"{obs}/{p / 1000:.0f}k" if p >= 10_000 else f"{obs}/{p / 1000:.1f}k") if p else f"{obs}/never")
        print(f"   wd {wd:<5g} observed/predicted: " + "  ".join(line))
        for s in range(5):
            r, p = cell.get(s), pred.get(key, {}).get(str(s))
            if r is None:
                continue
            if p is None or p > r["epochs"]:          # predicted not to grok within budget
                scored.append(not r["grokked"])
            else:                                      # predicted to grok: within a factor of 1.5?
                scored.append(r["grokked"] and abs(np.log(r["grok_epoch_95"] / p)) <= np.log(1.5))
    if scored:
        print(f"   pre-registered score: {sum(scored)}/{len(scored)} runs correct "
              f"(grok within x1.5 of prediction, or no grok where none predicted); "
              f"criterion >= 80%: {'met' if sum(scored) >= 0.8 * len(scored) else 'NOT met'}")


def h4_init_split(runs) -> None:
    print("\nH4  E4: initialisation vs split (wd 1.0, 5 x 5)")
    e4 = {(r["init_seed"], r["split_seed"]): r for r in runs if np.isclose(r["lr"], 1e-3)
          and r["wd"] == 1.0 and r["init_scale"] == 1.0 and r["epochs"] == 60_000}
    if len(e4) < 25 or not all(r["grokked"] for r in e4.values()):
        print(f"   {len(e4)}/25 runs present; needs all 25 grokked")
        return
    T = np.array([[np.log(e4[(i, s)]["grok_epoch_95"]) for s in range(5)] for i in range(5)])
    grand = T.mean()
    ss_init = 5 * ((T.mean(axis=1) - grand) ** 2).sum()
    ss_split = 5 * ((T.mean(axis=0) - grand) ** 2).sum()
    ss_res = ((T - T.mean(axis=1, keepdims=True) - T.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    total = ss_init + ss_split + ss_res
    for name, ss in (("init", ss_init), ("split", ss_split)):
        f = (ss / 4) / (ss_res / 16)
        print(f"   {name:5s}: {ss / total:5.1%} of variance in log grok time, F(4,16) = {f:.2f}, "
              f"p = {stats.f.sf(f, 4, 16):.2g}")
    print(f"   residual (init x split): {ss_res / total:5.1%}")

    def jacc(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else np.nan
    pairs = {"same init, different split": [], "same split, different init": [], "neither shared": []}
    for (k1, r1), (k2, r2) in itertools.combinations(e4.items(), 2):
        j = jacc(r1["key_freqs"], r2["key_freqs"])
        if k1[0] == k2[0]:
            pairs["same init, different split"].append(j)
        elif k1[1] == k2[1]:
            pairs["same split, different init"].append(j)
        else:
            pairs["neither shared"].append(j)
    for name, js in pairs.items():
        print(f"   key-frequency Jaccard, {name:27s}: median {np.nanmedian(js):.2f} (n = {len(js)})")


def h5_alarm(runs) -> None:
    print(f"\nH5  Weight-norm alarm (norm {ALARM_DROP:.0%} below running max), all experiments")
    fired = [r for r in runs if r["alarm_epoch"] is not None]
    g = [r for r in fired if r["grokked"]]
    false = [r for r in fired if not r["grokked"]]
    silent = [r for r in runs if r["alarm_epoch"] is None]
    print(f"   {len(runs)} runs: alarm fired in {len(fired)} "
          f"({len(g)} then grokked, {len(false)} did not = false alarms within budget); "
          f"silent in {len(silent)} ({sum(r['grokked'] for r in silent)} of those grokked = misses)")
    lead = [r for r in g if r["alarm_epoch"] < r["grok_epoch_95"]]
    if g:
        frac = np.median([(r["grok_epoch_95"] - r["alarm_epoch"]) / r["grok_epoch_95"] for r in g])
        print(f"   alarm before grokking in {len(lead)}/{len(g)}; median lead {frac:.0%} of time-to-grok")
    # Does the alarm say anything about the seed once the configuration is known?
    cfg = np.array([f"{r['lr']}_{r['wd']}_{r['init_scale']}_{r['epochs']}" for r in g])
    if len(g) >= 20:
        coef, half, dof = fe_ols(np.log([r["grok_epoch_95"] for r in g]),
                                 np.log([[r["alarm_epoch"]] for r in g]), cfg)
        print(f"   within-configuration slope of log grok on log alarm: {coef[0]:+.2f}, "
              f"95% CI [{coef[0] - half[0]:+.2f}, {coef[0] + half[0]:+.2f}]  "
              f"(0 = alarm only knows the configuration; 1 = it tracks the seed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="grokking/runs/wd_push")
    parser.add_argument("--sweep-root", default="grokking/runs/sweep_wd")
    parser.add_argument("--predictions", default="grokking/prereg/wd_predictions.json")
    parser.add_argument("--freeze-predictions", action="store_true")
    args = parser.parse_args()

    if args.freeze_predictions:
        freeze_predictions(Path(args.sweep_root) / "aggregate_results.csv", Path(args.predictions))
        return

    pred = json.loads(Path(args.predictions).read_text())
    runs = load_runs(Path(args.root))
    print(f"{len(runs)} completed runs under {args.root}")
    h0_sanity(runs, Path(args.sweep_root))
    h1_lr_wd(runs, pred["predicted_grok_epoch_95"])
    h2_init_scale(runs, pred["E1_predicted_slope_epochs_per_ln_alpha"])
    h3_range(runs, pred["predicted_grok_epoch_95"])
    h4_init_split(runs)
    h5_alarm(runs)


if __name__ == "__main__":
    main()
