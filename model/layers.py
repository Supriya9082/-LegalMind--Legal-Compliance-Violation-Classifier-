"""
Building blocks for LegalMind GPT:
  RMSNorm      — Root Mean Square layer normalization (no mean subtraction)
  FeedForward  — Position-wise 2-layer FFN with GELU activation
  TransformerBlock — One GPT layer: Attention + FFN with pre-norm residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .attention import GroupedQueryAttention, make_causal_mask


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler than LayerNorm — no mean subtraction, just RMS scaling.
    Used in LLaMA/Mistral; trains slightly faster and uses less memory.

    y = x / RMS(x) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along last dim; cast to float32 for numerical stability
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Standard GPT FFN: Linear → GELU → Linear → Dropout
    d_model -> d_ff -> d_model

    GELU (Gaussian Error Linear Unit) outperforms ReLU for language tasks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2  = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    One transformer layer with pre-normalization (Pre-LN) residuals.

    Pre-LN (normalize BEFORE attention/FFN) vs Post-LN (after):
      Pre-LN: more stable training, used in GPT-2+, LLaMA.
      Computation:  x = x + Attention(Norm(x))
                    x = x + FFN(Norm(x))

    Supports gradient checkpointing to trade memory for compute — essential
    when training on 8 GB RAM.
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int,
        d_ff:       int,
        dropout:    float = 0.1,
        bias:       bool  = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = GroupedQueryAttention(d_model, n_heads, n_kv_heads,
                                           dropout=dropout, bias=bias)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FeedForward(d_model, d_ff, dropout=dropout, bias=bias)

    def _attn_block(self, x: torch.Tensor, mask) -> torch.Tensor:
        """Attention sub-layer (wrapped for gradient checkpointing)."""
        return self.attn(self.norm1(x), mask)

    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """FFN sub-layer."""
        return self.ffn(self.norm2(x))

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        # Pre-norm + residual for attention
        if use_checkpoint and self.training:
            # gradient_checkpointing: don't store activations for backward pass.
            # Instead, recompute them during backward. ~2× slower but halves
            # activation memory — makes 15M model trainable on 8 GB.
            attn_out = grad_checkpoint(self._attn_block, x, mask, use_reentrant=False)
        else:
            attn_out = self._attn_block(x, mask)

        x = x + attn_out

        # Pre-norm + residual for FFN
        if use_checkpoint and self.training:
            ffn_out = grad_checkpoint(self._ffn_block, x, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(x)

        x = x + ffn_out
        return x
