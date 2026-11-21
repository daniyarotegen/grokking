"""Stage 4B: Resumable multi-seed & weight-decay aliasing sweep runner.

Trains 1-layer causal Transformers on (a + b) mod 113 across specified seeds for a given
weight decay, saving only the final checkpoint (~890 KB) and lightweight metrics every 500
epochs, then immediately computes float64 mechanistic & aliasing diagnostics into
sweep_record.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from grokking.src.aliasing_diag import (
    analytical_alias_scores,
    compute_trig_sum_energies_vectorized,
)
from grokking.src.analysis import (
    compute_full_logit_table,
    dft_2d_key_frequency_projection,
    logit_table_2d_fourier,
    make_fourier_basis,
    table_cross_entropy,
)
from grokking.src.data import (
    get_git_commit_hash,
    make_modular_addition_dataset,
    set_seed,
)
from grokking.src.model import OneLayerTransformer, TransformerConfig
from grokking.src.train import (
    compute_weight_l2_norm,
    cross_entropy_high_precision,
    evaluate_split,
    resolve_device,
    save_checkpoint,
)


def run_single_sweep_member(
    wd: float,
    seed: int,
    epochs: int,
    out_root: Path,
    device: torch.device,
    log_every: int = 500,
    p: int = 113,
    train_frac: float = 0.3,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.98,
    key_var_threshold: float = 0.01,
    init_seed: int | None = None,
    split_seed: int | None = None,
    init_scale: float = 1.0,
    run_dir: Path | None = None,
) -> dict[str, object]:
    """Train one (wd, seed) run (or skip if already completed) and save sweep_record.json.

    By default `seed` sets both the weight initialisation and the train/test split.
    `init_seed` and `split_seed` override them separately; `init_scale` multiplies every
    parameter after initialisation (the alpha of Lyu et al., 2024). With the defaults the
    run is identical to the original Stage 4B sweep.
    """
    init_seed = seed if init_seed is None else init_seed
    split_seed = seed if split_seed is None else split_seed
    if run_dir is None:
        run_dir = out_root / f"wd_{wd:.1f}" / f"seed_{seed}"
    record_path = run_dir / "sweep_record.json"
    if record_path.exists():
        print(f"[SKIP] {run_dir} already done", flush=True)
        return json.loads(record_path.read_text())

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit_hash()
    set_seed(init_seed)

    # The split uses its own torch.Generator, so it does not consume the global RNG
    # that initialises the model below.
    data = make_modular_addition_dataset(p=p, train_frac=train_frac, seed=split_seed).to(device)
    model_cfg = TransformerConfig(
        vocab_size=p + 1,
        n_classes=p,
        seq_len=3,
        use_mlp_bias=True,
    )
    model = OneLayerTransformer(model_cfg).to(device)
    if init_scale != 1.0:
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(init_scale)
    initial_weight_l2 = compute_weight_l2_norm(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(beta1, beta2),
    )

    run_meta = {
        "wd": wd,
        "seed": seed,
        "init_seed": init_seed,
        "split_seed": split_seed,
        "init_scale": init_scale,
        "initial_weight_l2_norm": initial_weight_l2,
        "epochs": epochs,
        "lr": lr,
        "betas": [beta1, beta2],
        "p": p,
        "train_frac": train_frac,
        "log_every": log_every,
        "model_config": asdict(model_cfg),
        "num_parameters": model.count_parameters(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "git_commit": git_commit,
    }
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    grok_epoch_95: int | None = None
    metrics_path = run_dir / "metrics.jsonl"
    last_metrics: dict[str, float | int] = {}

    def record_metrics(epoch: int, fp) -> dict[str, float | int]:
        nonlocal grok_epoch_95
        train_loss, train_acc = evaluate_split(model, data.x_train, data.y_train)
        test_loss, test_acc = evaluate_split(model, data.x_test, data.y_test)
        weight_l2 = compute_weight_l2_norm(model)
        elapsed_s = time.perf_counter() - t_start
        if grok_epoch_95 is None and test_acc >= 0.95:
            grok_epoch_95 = epoch
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "weight_l2_norm": weight_l2,
            "elapsed_s": elapsed_s,
        }
        fp.write(json.dumps(row) + "\n")
        return row

    with metrics_path.open("w") as fp:
        last_metrics = record_metrics(0, fp)
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            logits_final = model(data.x_train)[:, -1, :]  # (N_train, 113)
            loss = cross_entropy_high_precision(logits_final, data.y_train)
            loss.backward()
            optimizer.step()

            if (epoch % log_every == 0) or (epoch == epochs):
                last_metrics = record_metrics(epoch, fp)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time_s = time.perf_counter() - t_start

    # Save ONLY final checkpoint (~890 KB)
    final_ckpt_path = ckpt_dir / f"ckpt_{epochs:05d}.pt"
    save_checkpoint(final_ckpt_path, epochs, model, last_metrics)

    # --- Post-analysis in float64 ---
    model.eval()
    fourier_basis, _ = make_fourier_basis(p=p, device=device)
    logit_table = compute_full_logit_table(model, p=p, device=device)       # (P, P, P), float64
    f_logits = logit_table_2d_fourier(logit_table, fourier_basis)           # (P, P, P), float64
    trig_e_t, non_dc_e = compute_trig_sum_energies_vectorized(f_logits)     # (56,), float
    trig_e_np = trig_e_t.cpu().numpy()

    var_shares_non_dc = trig_e_np / non_dc_e                                # (56,)
    trig_sum_total = float(trig_e_np.sum())
    if trig_sum_total > 0.0:
        p_norm = trig_e_np / trig_sum_total                                 # sums to 1 over k=1..56
        n_eff = float(1.0 / np.sum(p_norm ** 2))
    else:
        n_eff = 0.0

    # Key frequencies K (> 1.0% non-DC trig-sum variance share)
    key_freqs = [int(i + 1) for i in np.where(var_shares_non_dc > key_var_threshold)[0]]
    key_freq_var_shares = {str(k): float(var_shares_non_dc[k - 1]) for k in key_freqs}
    if key_freqs:
        key_l2 = np.sqrt(np.maximum(trig_e_np[np.array(key_freqs) - 1], 0.0))
        key_l2_sum = float(key_l2.sum())
        key_freq_amp_weights = {
            str(k): float(val / max(key_l2_sum, 1e-12)) for k, val in zip(key_freqs, key_l2)
        }
    else:
        key_freq_amp_weights = {}

    # Restricted (test) and Excluded (train) losses & accuracies in float64
    restricted_table, excluded_table = dft_2d_key_frequency_projection(
        logit_table,
        fourier_basis,
        key_freqs=key_freqs,
        mode="trig_sum",
    )

    a_te, b_te, y_te = data.x_test[:, 0], data.x_test[:, 1], data.y_test
    a_tr, b_tr, y_tr = data.x_train[:, 0], data.x_train[:, 1], data.y_train

    restricted_test_loss = table_cross_entropy(restricted_table, data.x_test, data.y_test)
    rest_preds_te = restricted_table[a_te, b_te].argmax(dim=-1)
    restricted_test_errors = int((rest_preds_te != y_te).sum().item())
    restricted_test_acc = float(1.0 - restricted_test_errors / y_te.shape[0])

    excluded_train_loss = table_cross_entropy(excluded_table, data.x_train, data.y_train)
    excl_preds_tr = excluded_table[a_tr, b_tr].argmax(dim=-1)
    excluded_train_acc = float((excl_preds_tr == y_tr).to(torch.float64).mean().item())

    # Full-model test error histogram & residual L2 norm amplification
    full_logits_te = logit_table[a_te, b_te]                                # (N_test, P), float64
    rest_logits_te = restricted_table[a_te, b_te]                           # (N_test, P), float64
    full_preds_te = full_logits_te.argmax(dim=-1)
    offsets_te = (full_preds_te - y_te) % p
    wrong_mask = offsets_te != 0
    right_mask = ~wrong_mask
    total_test_errors = int(wrong_mask.sum().item())

    wrong_offsets = offsets_te[wrong_mask].tolist()
    offset_hist_counter = Counter(wrong_offsets)
    error_offset_histogram = {str(k): int(v) for k, v in sorted(offset_hist_counter.items())}

    resid_norms = (full_logits_te - rest_logits_te).norm(dim=-1)            # (N_test,)
    resid_mean_correct = (
        float(resid_norms[right_mask].mean().item()) if bool(right_mask.any()) else None
    )
    resid_mean_wrong = (
        float(resid_norms[wrong_mask].mean().item()) if total_test_errors > 0 else None
    )
    resid_ratio = (
        float(resid_mean_wrong / resid_mean_correct)
        if (resid_mean_wrong is not None and resid_mean_correct is not None and resid_mean_correct > 0)
        else None
    )

    # Analytical alias validation using amplitude weights w_k proportional to sqrt(E_k) over K
    if key_freqs:
        alias_info = analytical_alias_scores(
            trig_e_np,
            non_dc_e,
            p=p,
            min_var_frac=key_var_threshold,
            fixed_freqs=key_freqs,
        )
        top3_predicted_deltas = alias_info["top3_l2"]
        top1_d = int(top3_predicted_deltas[0]["delta"])
        top3_set = {int(item["delta"]) for item in top3_predicted_deltas}

        canonical_wrong = [min(d, p - d) for d in wrong_offsets]
        errors_in_top1 = sum(1 for d in canonical_wrong if d == top1_d)
        errors_in_top3 = sum(1 for d in canonical_wrong if d in top3_set)
        frac_top1 = float(errors_in_top1 / total_test_errors) if total_test_errors > 0 else None
        frac_top3 = float(errors_in_top3 / total_test_errors) if total_test_errors > 0 else None
    else:
        top3_predicted_deltas = []
        errors_in_top1 = 0
        errors_in_top3 = 0
        frac_top1 = None
        frac_top3 = None

    sweep_record: dict[str, object] = {
        "wd": wd,
        "seed": seed,
        "init_seed": init_seed,
        "split_seed": split_seed,
        "init_scale": init_scale,
        "lr": lr,
        "log_every": log_every,
        "epochs": epochs,
        "git_commit": git_commit,
        "wall_time_s": wall_time_s,
        "grok_epoch_95": grok_epoch_95,
        "train_loss": float(last_metrics["train_loss"]),
        "test_loss": float(last_metrics["test_loss"]),
        "train_acc": float(last_metrics["train_acc"]),
        "test_acc": float(last_metrics["test_acc"]),
        "weight_l2_norm": float(last_metrics["weight_l2_norm"]),
        "key_freqs": key_freqs,
        "n_key_freqs": len(key_freqs),
        "key_freq_var_shares": key_freq_var_shares,
        "key_freq_amp_weights": key_freq_amp_weights,
        "key_freqs_total_var_share": float(sum(key_freq_var_shares.values())),
        "trig_sum_total_var_share": float(trig_sum_total / non_dc_e),
        "n_eff": n_eff,
        "restricted_test_loss": restricted_test_loss,
        "restricted_test_acc": restricted_test_acc,
        "restricted_test_errors": restricted_test_errors,
        "excluded_train_loss": excluded_train_loss,
        "excluded_train_acc": excluded_train_acc,
        "total_test_errors": total_test_errors,
        "error_offset_histogram": error_offset_histogram,
        "top3_predicted_deltas": top3_predicted_deltas,
        "errors_in_top1_alias": errors_in_top1,
        "errors_in_top3_alias": errors_in_top3,
        "frac_errors_in_top1_alias": frac_top1,
        "frac_errors_in_top3_alias": frac_top3,
        "resid_norm_mean_correct": resid_mean_correct,
        "resid_norm_mean_wrong": resid_mean_wrong,
        "resid_norm_ratio_wrong_over_correct": resid_ratio,
    }

    # Atomic write so resumability is crash-safe
    tmp_path = run_dir / "sweep_record.json.tmp"
    tmp_path.write_text(json.dumps(sweep_record, indent=2) + "\n")
    tmp_path.replace(record_path)

    print(
        f"[DONE] {run_dir.parent.name}/{run_dir.name} | t={wall_time_s:5.1f}s | "
        f"grok95={str(grok_epoch_95):>6s} | test_acc={sweep_record['test_acc']:.4f} "
        f"(err={total_test_errors:4d}, rest_err={restricted_test_errors:4d}) | "
        f"|K|={len(key_freqs)} N_eff={n_eff:.2f} K={key_freqs} | "
        f"top3_hit={f'{frac_top3*100:.1f}%' if frac_top3 is not None else 'n/a'}",
        flush=True,
    )
    return sweep_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable multi-seed weight-decay sweep.")
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=40_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--out-root", type=str, default="grokking/runs/sweep_wd")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    t0 = time.perf_counter()
    for seed in args.seeds:
        run_single_sweep_member(
            wd=args.wd,
            seed=seed,
            epochs=args.epochs,
            out_root=out_root,
            device=device,
            log_every=args.log_every,
        )
    print(
        f"\nBatch wd={args.wd:.1f} ({len(args.seeds)} seeds) finished in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
