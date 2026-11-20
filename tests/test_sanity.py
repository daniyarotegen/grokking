"""Mandatory pre-training sanity tests for modular addition grokking."""

from __future__ import annotations

import pytest
import torch

from grokking.src.data import make_modular_addition_dataset, set_seed
from grokking.src.model import OneLayerTransformer, TransformerConfig
from grokking.src.train import cross_entropy_high_precision


def test_output_shape_and_param_count() -> None:
    """1. Output shape is (batch, 3, 113) and parameter count matches expected 1-layer config."""
    set_seed(42)
    cfg = TransformerConfig()
    model = OneLayerTransformer(cfg)

    batch_size = 16
    tokens = torch.randint(0, 113, (batch_size, 3), dtype=torch.long)
    tokens[:, 2] = 113
    logits = model(tokens)

    assert logits.shape == (batch_size, 3, 113)
    assert model.count_parameters() == cfg.expected_num_parameters()
    assert model.count_parameters() == 226_688

    # Verify all 9 required weight matrices are directly accessible with exact shapes
    expected_shapes = {
        "W_E": (114, 128),
        "W_pos": (3, 128),
        "W_Q": (4, 128, 32),
        "W_K": (4, 128, 32),
        "W_V": (4, 128, 32),
        "W_O": (4, 32, 128),
        "W_in": (128, 512),
        "W_out": (512, 128),
        "W_U": (128, 113),
    }
    for name, shape in expected_shapes.items():
        param = getattr(model, name)
        assert isinstance(param, torch.nn.Parameter), f"{name} must be an nn.Parameter"
        assert tuple(param.shape) == shape, f"{name} shape {tuple(param.shape)} != {shape}"

    # Verify W_E and W_U are untied (distinct storage)
    assert model.W_E.data_ptr() != model.W_U.data_ptr()

    # Verify no-MLP-bias config also matches exact analytical count
    cfg_no_bias = TransformerConfig(use_mlp_bias=False)
    model_no_bias = OneLayerTransformer(cfg_no_bias)
    assert model_no_bias.count_parameters() == cfg_no_bias.expected_num_parameters() == 226_048


def test_causal_mask_blocks_future_tokens() -> None:
    """2. Causal mask prevents position i from attending to j > i."""
    set_seed(123)
    model = OneLayerTransformer(TransformerConfig())
    model.eval()

    x_base = torch.tensor(
        [
            [5, 17, 113],
            [42, 99, 113],
            [0, 112, 113],
            [73, 11, 113],
        ],
        dtype=torch.long,
    )  # (4, 3)

    with torch.no_grad():
        logits_base = model(x_base)  # (4, 3, 113)

        # Perturb position 2 -> positions 0 and 1 must be completely unchanged
        x_pert_pos2 = x_base.clone()
        x_pert_pos2[:, 2] = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        logits_pert_pos2 = model(x_pert_pos2)

        assert torch.allclose(logits_base[:, 0, :], logits_pert_pos2[:, 0, :], atol=1e-6, rtol=0.0)
        assert torch.allclose(logits_base[:, 1, :], logits_pert_pos2[:, 1, :], atol=1e-6, rtol=0.0)
        assert not torch.allclose(logits_base[:, 2, :], logits_pert_pos2[:, 2, :], atol=1e-4)

        # Perturb position 1 -> position 0 must be unchanged; positions 1 and 2 must change
        x_pert_pos1 = x_base.clone()
        x_pert_pos1[:, 1] = (x_pert_pos1[:, 1] + 1) % 113
        logits_pert_pos1 = model(x_pert_pos1)

        assert torch.allclose(logits_base[:, 0, :], logits_pert_pos1[:, 0, :], atol=1e-6, rtol=0.0)
        assert not torch.allclose(logits_base[:, 1, :], logits_pert_pos1[:, 1, :], atol=1e-4)
        assert not torch.allclose(logits_base[:, 2, :], logits_pert_pos1[:, 2, :], atol=1e-4)


def test_dataset_split_disjoint_and_complete() -> None:
    """3. Train and test pairs have zero overlap and cover all 113 * 113 = 12,769 pairs."""
    data = make_modular_addition_dataset(p=113, train_frac=0.3, seed=42)

    assert data.x_train.shape == (3830, 3)
    assert data.y_train.shape == (3830,)
    assert data.x_test.shape == (8939, 3)
    assert data.y_test.shape == (8939,)

    # Check '=' token and modular addition labels
    assert torch.all(data.x_train[:, 2] == 113)
    assert torch.all(data.x_test[:, 2] == 113)
    assert torch.equal(data.y_train, (data.x_train[:, 0] + data.x_train[:, 1]) % 113)
    assert torch.equal(data.y_test, (data.x_test[:, 0] + data.x_test[:, 1]) % 113)

    train_pairs = {(int(a), int(b)) for a, b in data.x_train[:, :2].tolist()}
    test_pairs = {(int(a), int(b)) for a, b in data.x_test[:, :2].tolist()}

    assert len(train_pairs) == 3830
    assert len(test_pairs) == 8939
    assert len(train_pairs & test_pairs) == 0
    assert len(train_pairs | test_pairs) == 113 * 113 == 12_769


def _run_short_training(seed: int, device: torch.device, steps: int = 10):
    set_seed(seed)
    data = make_modular_addition_dataset(p=113, train_frac=0.3, seed=seed).to(device)
    model = OneLayerTransformer(TransformerConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1.0,
        betas=(0.9, 0.98),
    )

    x_sub = data.x_train[:128]
    y_sub = data.y_train[:128]

    with torch.no_grad():
        step0_loss = cross_entropy_high_precision(model(x_sub)[:, -1, :], y_sub).item()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy_high_precision(model(x_sub)[:, -1, :], y_sub)
        loss.backward()
        optimizer.step()

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return step0_loss, state


@pytest.mark.parametrize(
    "device_str",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_determinism_step0_and_step10(device_str: str) -> None:
    """4. Identical seeds produce identical step-0 loss and step-10 weights."""
    device = torch.device(device_str)
    loss_a, weights_a = _run_short_training(seed=42, device=device, steps=10)
    loss_b, weights_b = _run_short_training(seed=42, device=device, steps=10)

    assert loss_a == loss_b
    for name in weights_a:
        assert torch.equal(weights_a[name], weights_b[name]), f"Mismatch in {name} on {device_str}"


def test_single_batch_overfit() -> None:
    """5. A 64-example slice reaches >99% accuracy within 300 steps with weight_decay=0.0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    data = make_modular_addition_dataset(p=113, train_frac=0.3, seed=42).to(device)
    x_batch = data.x_train[:64]   # (64, 3)
    y_batch = data.y_train[:64]   # (64,)

    model = OneLayerTransformer(TransformerConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.98),
    )

    acc = 0.0
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)[:, -1, :]  # (64, 113)
        loss = cross_entropy_high_precision(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = logits.detach().argmax(dim=-1)
        acc = (preds == y_batch).to(torch.float64).mean().item()
        if acc > 0.99:
            break

    assert acc > 0.99, f"Expected >99% accuracy within 300 steps, got {acc:.4f}"
