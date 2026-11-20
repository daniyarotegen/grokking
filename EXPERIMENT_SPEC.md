# Experiment Spec: Progress Measures for Grokking (Nanda et al., 2023)

## Objective
Reproduce the modular addition grokking phase transition and its mechanistic progress measures (Fourier spectrum, restricted loss, excluded loss) from scratch in PyTorch + `einops`.

## Task & Dataset
- Operation: `(a + b) mod P` with `P = 113`.
- Sequence format: 3 tokens `[a, b, =]` where `=` has token ID `113` (vocab size `114`, output classes `113`).
- Prediction target: final token position (index 2) logits.
- Split: 30% train (3,830 pairs) / 70% test (8,939 pairs), generated deterministically from `seed`.
- Batching: full-batch gradient descent (all train pairs per step).

## Model Architecture (~200k params)
- 1-layer causal Transformer decoder, implemented from scratch with directly accessible weight matrices (`W_E`, `W_pos`, `W_Q`, `W_K`, `W_V`, `W_O`, `W_in`, `W_out`, `W_U`).
- `d_model = 128`, `n_heads = 4` (`d_head = 32`), `d_mlp = 512`, ReLU activation.
- Learned positional embeddings (length 3).
- NO LayerNorm.
- Untied embedding (`W_E`) and unembedding (`W_U`).

## Optimization & Numerics
- Optimizer: AdamW, `lr = 1e-3`, `weight_decay = 1.0`, `betas = (0.9, 0.98)`.
- Epochs: `40_000` full-batch steps (single baseline seed first: `seed = 42`).
- CRITICAL: Cast logits to `torch.float64` before `F.cross_entropy` to prevent `float32` softmax underflow / slingshot artifacts during the memorization plateau.
- Log train loss, test loss, train acc, test acc, and total weight L2 norm every 100 epochs; save checkpoints every 1,000 epochs (plus epoch 0 and final epoch).

## Mandatory Pre-Training Verification (`tests/test_sanity.py`)
1. Output shape is `(batch, 3, 113)` and parameter count matches expected 1-layer config.
2. Causal mask prevents position `i` from attending to `j > i` (verified by perturbing later tokens).
3. Train and test pairs have strictly zero overlap and cover all `113 * 113 = 12,769` pairs.
4. Determinism: identical seeds produce identical step-0 loss and step-10 weights.
5. Single-batch overfit: a 64-example slice reaches >99% accuracy within 300 steps with `weight_decay=0.0`.

## Stage 3 Mechanistic Progress Measures (Post-Training)
- Compute the 2D Discrete Fourier Transform over the `113 x 113` final-token logit table (and embedding matrix `W_E`).
- Identify top key frequencies `K` (typically ~5 frequencies carrying >90% of logit norm after grokking).
- Compute across saved checkpoints:
  - **Restricted loss:** evaluate cross-entropy keeping only key Fourier components `K` (must drop steadily during the flat test-loss memorization plateau).
  - **Excluded loss:** evaluate cross-entropy ablating key frequencies `K` (must rise during the cleanup phase).
- Save publication-clean matplotlib figures under `grokking/artifacts/figures/`.
