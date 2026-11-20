"""Full-batch training loop for modular addition grokking (Nanda et al., 2023)."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.src.data import (
    ModularAdditionData,
    get_git_commit_hash,
    make_modular_addition_dataset,
    set_seed,
)
from grokking.src.model import OneLayerTransformer, TransformerConfig


@dataclass(frozen=True)
class TrainConfig:
    p: int = 113
    train_frac: float = 0.3
    seed: int = 42
    epochs: int = 40_000
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    log_every: int = 100
    save_every: int = 1_000
    out_dir: str = "grokking/runs/baseline_seed42"
    device: str = "auto"
    use_mlp_bias: bool = True


def cross_entropy_high_precision(
    logits: torch.Tensor,  # (B, C)
    targets: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """Compute cross-entropy in float64 to prevent float32 softmax underflow / slingshots."""
    return F.cross_entropy(logits.to(torch.float64), targets)


@torch.no_grad()
def compute_weight_l2_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm across all trainable parameters."""
    sq_sum = sum(
        p.detach().to(torch.float64).pow(2).sum().item()
        for p in model.parameters()
    )
    return math.sqrt(sq_sum)


@torch.no_grad()
def evaluate_split(
    model: OneLayerTransformer,
    x: torch.Tensor,  # (N, 3)
    y: torch.Tensor,  # (N,)
) -> tuple[float, float]:
    """Evaluate float64 cross-entropy loss and exact-match accuracy on final token."""
    logits = model(x)[:, -1, :]                          # (N, 113)
    loss = cross_entropy_high_precision(logits, y).item()
    preds = logits.argmax(dim=-1)                        # (N,)
    acc = (preds == y).to(torch.float64).mean().item()
    return loss, acc


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: OneLayerTransformer,
    metrics: dict[str, float | int],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "metrics": metrics,
    }
    torch.save(payload, path)


def train(train_cfg: TrainConfig) -> dict[str, object]:
    """Execute full-batch training, logging lightweight JSONL metrics and compact checkpoints."""
    device = resolve_device(train_cfg.device)
    out_dir = Path(train_cfg.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Fix all seeds before dataset creation and model initialization
    set_seed(train_cfg.seed)

    data: ModularAdditionData = make_modular_addition_dataset(
        p=train_cfg.p,
        train_frac=train_cfg.train_frac,
        seed=train_cfg.seed,
    ).to(device)

    model_cfg = TransformerConfig(
        vocab_size=train_cfg.p + 1,
        n_classes=train_cfg.p,
        seq_len=3,
        use_mlp_bias=train_cfg.use_mlp_bias,
    )
    model = OneLayerTransformer(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    run_meta = {
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "num_parameters": model.count_parameters(),
        "n_train": int(data.x_train.shape[0]),
        "n_test": int(data.x_test.shape[0]),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "git_commit": get_git_commit_hash(),
    }
    (out_dir / "config.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    metrics_path = out_dir / "metrics.jsonl"
    saved_ckpt_paths: list[Path] = []
    start_time = time.perf_counter()

    def log_and_maybe_save(epoch: int, fp) -> dict[str, float | int]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_loss, train_acc = evaluate_split(model, data.x_train, data.y_train)
        test_loss, test_acc = evaluate_split(model, data.x_test, data.y_test)
        weight_l2 = compute_weight_l2_norm(model)
        elapsed_s = time.perf_counter() - start_time

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "weight_l2_norm": weight_l2,
            "elapsed_s": elapsed_s,
        }
        fp.write(json.dumps(record) + "\n")
        fp.flush()

        print(
            f"epoch={epoch:5d} | "
            f"train_loss={train_loss:.6e} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.6e} test_acc={test_acc:.4f} | "
            f"w_l2={weight_l2:.3f} | t={elapsed_s:.2f}s",
            flush=True,
        )

        if epoch == 0 or epoch % train_cfg.save_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = ckpt_dir / f"ckpt_{epoch:05d}.pt"
            save_checkpoint(ckpt_path, epoch, model, record)
            saved_ckpt_paths.append(ckpt_path)

        return record

    with metrics_path.open("w") as fp:
        # Epoch 0 baseline (untrained)
        last_record = log_and_maybe_save(0, fp)

        # Reset timer specifically for pure training-loop throughput measurement
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        loop_start = time.perf_counter()

        for epoch in range(1, train_cfg.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            logits_final = model(data.x_train)[:, -1, :]  # (N_train, 113)
            loss = cross_entropy_high_precision(logits_final, data.y_train)
            loss.backward()
            optimizer.step()

            should_log = (epoch % train_cfg.log_every == 0) or (epoch == train_cfg.epochs)
            should_save = (epoch % train_cfg.save_every == 0) or (epoch == train_cfg.epochs)
            if should_log or should_save:
                last_record = log_and_maybe_save(epoch, fp)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        loop_elapsed_s = time.perf_counter() - loop_start

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    peak_vram_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    sec_per_1k_epochs = (
        (loop_elapsed_s / train_cfg.epochs) * 1000.0 if train_cfg.epochs > 0 else 0.0
    )
    ckpt_size_bytes = saved_ckpt_paths[-1].stat().st_size if saved_ckpt_paths else 0

    summary: dict[str, object] = {
        "epochs": train_cfg.epochs,
        "loop_elapsed_s": loop_elapsed_s,
        "sec_per_1k_epochs": sec_per_1k_epochs,
        "peak_vram_allocated_mb": peak_vram_mb,
        "peak_vram_reserved_mb": peak_vram_reserved_mb,
        "checkpoint_size_bytes": ckpt_size_bytes,
        "num_checkpoints": len(saved_ckpt_paths),
        "final_metrics": last_record,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\nDone: {train_cfg.epochs} epochs in {loop_elapsed_s:.2f}s "
        f"({sec_per_1k_epochs:.2f} s / 1k epochs) | "
        f"peak VRAM alloc={peak_vram_mb:.1f} MB (reserved={peak_vram_reserved_mb:.1f} MB) | "
        f"ckpt size={ckpt_size_bytes / 1024:.1f} KB ({len(saved_ckpt_paths)} files)",
        flush=True,
    )
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train 1-layer Transformer on modular addition.")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--out-dir", type=str, default="grokking/runs/baseline_seed42")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-mlp-bias", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        p=args.p,
        train_frac=args.train_frac,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=args.out_dir,
        device=args.device,
        use_mlp_bias=not args.no_mlp_bias,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
