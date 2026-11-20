"""Dataset generation and deterministic seeding for modular addition grokking."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNG seeds for reproducible initialization, data splits, and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ModularAdditionData:
    """Container for full-batch modular addition train/test splits."""

    p: int
    eq_token: int
    train_frac: float
    seed: int
    x_train: torch.Tensor  # (N_train, 3), int64
    y_train: torch.Tensor  # (N_train,), int64
    x_test: torch.Tensor   # (N_test, 3), int64
    y_test: torch.Tensor   # (N_test,), int64

    def to(self, device: torch.device | str) -> ModularAdditionData:
        return ModularAdditionData(
            p=self.p,
            eq_token=self.eq_token,
            train_frac=self.train_frac,
            seed=self.seed,
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_test=self.x_test.to(device),
            y_test=self.y_test.to(device),
        )


def make_full_grid(p: int = 113) -> tuple[torch.Tensor, torch.Tensor]:
    """Build all P*P ordered pairs [a, b, =] and targets (a + b) mod P.

    Row index is `a * p + b` so outputs reshape directly to `(p, p, ...)`.
    """
    a = torch.arange(p, dtype=torch.long).repeat_interleave(p)  # (P*P,)
    b = torch.arange(p, dtype=torch.long).repeat(p)             # (P*P,)
    eq = torch.full((p * p,), p, dtype=torch.long)              # (P*P,), token ID = P
    x = torch.stack([a, b, eq], dim=-1)                         # (P*P, 3)
    y = (a + b) % p                                             # (P*P,)
    return x, y


def make_modular_addition_dataset(
    p: int = 113,
    train_frac: float = 0.3,
    seed: int = 42,
) -> ModularAdditionData:
    """Create disjoint deterministic train/test splits for (a + b) mod P.

    For P = 113 and train_frac = 0.3:
        total = 12,769 pairs -> 3,830 train / 8,939 test.
    """
    x_all, y_all = make_full_grid(p=p)  # (P*P, 3), (P*P,)
    n_total = p * p
    n_train = int(n_total * train_frac)

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)  # (P*P,)

    train_idx = perm[:n_train]  # (N_train,)
    test_idx = perm[n_train:]   # (N_test,)

    return ModularAdditionData(
        p=p,
        eq_token=p,
        train_frac=train_frac,
        seed=seed,
        x_train=x_all[train_idx].clone(),  # (N_train, 3)
        y_train=y_all[train_idx].clone(),  # (N_train,)
        x_test=x_all[test_idx].clone(),    # (N_test, 3)
        y_test=y_all[test_idx].clone(),    # (N_test,)
    )


def get_git_commit_hash() -> str:
    """Return current git HEAD hash if inside a git repo, else 'not-a-git-repo'."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.strip()
    except Exception:
        return "not-a-git-repo"
