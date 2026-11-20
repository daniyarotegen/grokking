"""From-scratch 1-layer causal Transformer decoder for modular addition grokking.

Follows Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability":
- 1 Transformer block: multi-head causal self-attention + ReLU MLP
- NO LayerNorm
- Learned positional embeddings (length 3)
- Untied embedding (W_E) and unembedding (W_U)
- Directly accessible weight matrices: W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 114     # 0..112 plus '=' token at 113
    n_classes: int = 113      # output classes 0..112
    seq_len: int = 3          # [a, b, =]
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    use_mlp_bias: bool = True

    def __post_init__(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * d_head ({self.d_head}) must equal d_model ({self.d_model})"
            )

    def expected_num_parameters(self) -> int:
        n_embed = self.vocab_size * self.d_model + self.seq_len * self.d_model
        n_attn = 4 * self.n_heads * self.d_model * self.d_head
        n_mlp = 2 * self.d_model * self.d_mlp
        if self.use_mlp_bias:
            n_mlp += self.d_mlp + self.d_model
        n_unembed = self.d_model * self.n_classes
        return n_embed + n_attn + n_mlp + n_unembed


class OneLayerTransformer(nn.Module):
    """1-layer causal Transformer without LayerNorm and with explicit weight matrices."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embeddings (untied from W_U)
        self.W_E = nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.d_model)       # (V=114, D=128)
        )
        self.W_pos = nn.Parameter(
            torch.empty(cfg.seq_len, cfg.d_model)          # (S=3, D=128)
        )

        # Multi-head self-attention weights (no biases, per Nanda et al. 2023)
        self.W_Q = nn.Parameter(
            torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head)  # (H=4, D=128, K=32)
        )
        self.W_K = nn.Parameter(
            torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head)  # (H=4, D=128, K=32)
        )
        self.W_V = nn.Parameter(
            torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head)  # (H=4, D=128, K=32)
        )
        self.W_O = nn.Parameter(
            torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model)  # (H=4, K=32, D=128)
        )

        # 2-layer ReLU MLP
        self.W_in = nn.Parameter(
            torch.empty(cfg.d_model, cfg.d_mlp)            # (D=128, M=512)
        )
        self.W_out = nn.Parameter(
            torch.empty(cfg.d_mlp, cfg.d_model)            # (M=512, D=128)
        )
        if cfg.use_mlp_bias:
            self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))     # (M=512,)
            self.b_out = nn.Parameter(torch.zeros(cfg.d_model))  # (D=128,)
        else:
            self.register_parameter("b_in", None)
            self.register_parameter("b_out", None)

        # Untied unembedding (no bias)
        self.W_U = nn.Parameter(
            torch.empty(cfg.d_model, cfg.n_classes)        # (D=128, C=113)
        )

        # Upper-triangular boolean causal mask: True where key_pos > query_pos
        causal_mask = torch.triu(
            torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool),
            diagonal=1,
        )  # (S, S)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with 1 / sqrt(fan_in) normal scaling and zero MLP biases."""
        inv_sqrt_d = 1.0 / math.sqrt(self.cfg.d_model)
        inv_sqrt_m = 1.0 / math.sqrt(self.cfg.d_mlp)

        nn.init.normal_(self.W_E, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_pos, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_Q, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_K, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_V, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_O, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_in, mean=0.0, std=inv_sqrt_d)
        nn.init.normal_(self.W_out, mean=0.0, std=inv_sqrt_m)
        nn.init.normal_(self.W_U, mean=0.0, std=inv_sqrt_d)

        if self.b_in is not None:
            nn.init.zeros_(self.b_in)
        if self.b_out is not None:
            nn.init.zeros_(self.b_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run 1-layer causal Transformer forward pass.

        Args:
            tokens: Integer token IDs of shape (B, S) with S <= cfg.seq_len.

        Returns:
            logits: Float tensor of shape (B, S, C=113).
        """
        _, seq_len = tokens.shape
        if seq_len > self.cfg.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max seq_len {self.cfg.seq_len}")

        # 1. Token + learned positional embeddings
        tok_emb = F.embedding(tokens, self.W_E)                            # (B, S, D)
        pos_emb = self.W_pos[:seq_len]                                     # (S, D)
        x = tok_emb + pos_emb                                              # (B, S, D)

        # 2. Causal multi-head self-attention (no LayerNorm)
        q = einsum(x, self.W_Q, "b s d, h d k -> b h s k")                 # (B, H, S, K)
        k = einsum(x, self.W_K, "b s d, h d k -> b h s k")                 # (B, H, S, K)
        v = einsum(x, self.W_V, "b s d, h d k -> b h s k")                 # (B, H, S, K)

        attn_scores = (
            einsum(q, k, "b h sq k, b h sk k -> b h sq sk")                # (B, H, S, S)
            / math.sqrt(self.cfg.d_head)
        )
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len],
            float("-inf"),
        )                                                                  # (B, H, S, S)
        attn_pattern = F.softmax(attn_scores, dim=-1)                      # (B, H, S, S)

        z = einsum(attn_pattern, v, "b h sq sk, b h sk k -> b h sq k")     # (B, H, S, K)
        attn_out = einsum(z, self.W_O, "b h s k, h k d -> b s d")          # (B, S, D)
        x = x + attn_out                                                   # (B, S, D)

        # 3. ReLU MLP (no LayerNorm)
        mlp_pre = einsum(x, self.W_in, "b s d, d m -> b s m")              # (B, S, M)
        if self.b_in is not None:
            mlp_pre = mlp_pre + self.b_in                                  # (B, S, M)
        mlp_act = F.relu(mlp_pre)                                          # (B, S, M)
        mlp_out = einsum(mlp_act, self.W_out, "b s m, m d -> b s d")       # (B, S, D)
        if self.b_out is not None:
            mlp_out = mlp_out + self.b_out                                 # (B, S, D)
        x = x + mlp_out                                                    # (B, S, D)

        # 4. Untied unembedding
        logits = einsum(x, self.W_U, "b s d, d c -> b s c")                # (B, S, C)
        return logits

    @torch.no_grad()
    def forward_with_cache(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass returning intermediate activations for mechanistic analysis."""
        _, seq_len = tokens.shape
        tok_emb = F.embedding(tokens, self.W_E)                            # (B, S, D)
        pos_emb = self.W_pos[:seq_len]                                     # (S, D)
        resid_pre = tok_emb + pos_emb                                      # (B, S, D)

        q = einsum(resid_pre, self.W_Q, "b s d, h d k -> b h s k")         # (B, H, S, K)
        k = einsum(resid_pre, self.W_K, "b s d, h d k -> b h s k")         # (B, H, S, K)
        v = einsum(resid_pre, self.W_V, "b s d, h d k -> b h s k")         # (B, H, S, K)

        attn_scores = (
            einsum(q, k, "b h sq k, b h sk k -> b h sq sk")
            / math.sqrt(self.cfg.d_head)
        )
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len],
            float("-inf"),
        )
        attn_pattern = F.softmax(attn_scores, dim=-1)                      # (B, H, S, S)
        z = einsum(attn_pattern, v, "b h sq sk, b h sk k -> b h sq k")     # (B, H, S, K)
        attn_out = einsum(z, self.W_O, "b h s k, h k d -> b s d")          # (B, S, D)
        resid_mid = resid_pre + attn_out                                   # (B, S, D)

        mlp_pre = einsum(resid_mid, self.W_in, "b s d, d m -> b s m")      # (B, S, M)
        if self.b_in is not None:
            mlp_pre = mlp_pre + self.b_in
        mlp_act = F.relu(mlp_pre)                                          # (B, S, M)
        mlp_out = einsum(mlp_act, self.W_out, "b s m, m d -> b s d")       # (B, S, D)
        if self.b_out is not None:
            mlp_out = mlp_out + self.b_out
        resid_post = resid_mid + mlp_out                                   # (B, S, D)

        logits = einsum(resid_post, self.W_U, "b s d, d c -> b s c")       # (B, S, C)
        cache = {
            "resid_pre": resid_pre,
            "attn_pattern": attn_pattern,
            "attn_out": attn_out,
            "resid_mid": resid_mid,
            "mlp_pre": mlp_pre,
            "mlp_act": mlp_act,
            "mlp_out": mlp_out,
            "resid_post": resid_post,
        }
        return logits, cache
