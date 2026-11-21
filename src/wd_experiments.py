"""Stage 5: pushing on the weight-decay law (pre-registered in prereg/WD_PREREGISTRATION.md).

Four experiments, each a grid of runs that reuses the Stage 4B training and diagnostics
(src/sweep.py) with metrics logged every 100 epochs:

    E1  init scale   alpha in {0.5, 1, 2, 4} at wd = 1.0, lr = 1e-3
    E2  lr x wd      lr in {5e-4, 1e-3, 2e-3} x wd in {0.5, 1.0, 2.0}
    E3  wd range     wd in {0, 0.05, 0.1, 0.2, 3.0, 5.0}, 120k epochs
    E4  init x split init seed 0-4 x split seed 0-4 at wd = 1.0

All use seeds 0-4. Each run is stored once under a folder named by its full configuration,
so runs shared between experiments train only once, and an interrupted batch resumes
where it stopped.

    python -m grokking.src.wd_experiments --experiments E2 E3 E1 E4
    python -m grokking.src.wd_experiments --dry-run          # list runs, train nothing
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

SEEDS = (0, 1, 2, 3, 4)
BASE_LR = 1e-3
BASE_WD = 1.0


@dataclass(frozen=True)
class RunSpec:
    lr: float
    wd: float
    init_scale: float
    init_seed: int
    split_seed: int
    epochs: int

    @property
    def name(self) -> str:
        return (
            f"lr{self.lr:.0e}_wd{self.wd:g}_a{self.init_scale:g}"
            f"_init{self.init_seed}_split{self.split_seed}_ep{self.epochs // 1000}k"
        )


def experiment_grid(name: str) -> list[RunSpec]:
    if name == "E1":
        return [RunSpec(BASE_LR, BASE_WD, a, s, s, 60_000)
                for a in (0.5, 1.0, 2.0, 4.0) for s in SEEDS]
    if name == "E2":
        return [RunSpec(lr, wd, 1.0, s, s, 60_000)
                for lr in (5e-4, 1e-3, 2e-3) for wd in (0.5, 1.0, 2.0) for s in SEEDS]
    if name == "E3":
        return [RunSpec(BASE_LR, wd, 1.0, s, s, 120_000)
                for wd in (0.0, 0.05, 0.1, 0.2, 3.0, 5.0) for s in SEEDS]
    if name == "E4":
        return [RunSpec(BASE_LR, BASE_WD, 1.0, i, sp, 60_000)
                for i in SEEDS for sp in SEEDS]
    raise ValueError(f"unknown experiment {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["E2", "E3", "E1", "E4"])
    parser.add_argument("--out-root", default="grokking/runs/wd_push")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="cap epochs for a quick pilot (stored under a separate name)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs: list[RunSpec] = []
    for exp in args.experiments:
        for spec in experiment_grid(exp):
            if args.max_epochs is not None:
                spec = RunSpec(spec.lr, spec.wd, spec.init_scale, spec.init_seed,
                               spec.split_seed, min(spec.epochs, args.max_epochs))
            if spec not in specs:
                specs.append(spec)

    out_root = Path(args.out_root)
    done = [s for s in specs if (out_root / s.name / "sweep_record.json").exists()]
    total_epochs = sum(s.epochs for s in specs if s not in done)
    print(f"{len(specs)} unique runs, {len(done)} already done, "
          f"{total_epochs / 1e6:.2f}M epochs to train", flush=True)
    if args.dry_run:
        for s in specs:
            print(("done  " if s in done else "todo  ") + s.name)
        return

    # Imported here so --dry-run works without torch installed.
    from grokking.src.sweep import run_single_sweep_member
    from grokking.src.train import resolve_device

    device = resolve_device(args.device)
    t0 = time.perf_counter()
    for i, s in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {s.name}", flush=True)
        run_single_sweep_member(
            wd=s.wd,
            seed=s.init_seed,
            epochs=s.epochs,
            out_root=out_root,
            device=device,
            log_every=args.log_every,
            lr=s.lr,
            init_seed=s.init_seed,
            split_seed=s.split_seed,
            init_scale=s.init_scale,
            run_dir=out_root / s.name,
        )
    print(f"\nAll runs finished in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
