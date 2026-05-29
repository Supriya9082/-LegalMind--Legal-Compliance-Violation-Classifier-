"""
Grouped Query Attention (GQA) — the attention mechanism used in LegalMind.

Standard Multi-Head Attention (MHA): n_heads query, key, AND value heads.
  Memory for KV cache: 2 * n_heads * d_head * seq_len per layer.

Grouped Query Attention (GQA): n_heads query heads, but only n_kv_heads
  key/value heads. Multiple queries share each KV pair.
  Memory saved: n_heads / n_kv_heads × reduction.

With n_heads=8, n_kv_heads=2:
  - 4 query heads share each K/V head
  - KV memory is 4× smaller than MHA
  - Minimal quality loss (empirically shown in LLaMA 2, Mistral)

Shapes used throughout:
  B  = batch size
  T  = sequence length (context)
  C  = d_model (embedding dimension)
  H  = n_heads (query heads)
  G  = n_kv_heads (KV heads)
  D  = d_head = C // H
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with causal masking.

    Args:
        d_model    : Model dimension (C)
        n_heads    : Number of query heads (H)
        n_kv_heads : Number of key/value heads (G). Must divide n_heads.
        dropout    : Attention dropout probability
        bias       : Whether to add bias to linear projections
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int,
        dropout:    float = 0.0,
        bias:       bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads     # queries per KV head
        self.d_head     = d_model // n_heads
        self.dropout    = dropout

        # Query projects to full n_heads × d_head
        self.q_proj  = nn.Linear(d_model, n_heads    * self.d_head, bias=bias)
        # Key/Value projects to smaller n_kv_heads × d_head
        self.k_proj  = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj  = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model,   bias=bias)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match number of query heads by repeating.

        Input:  (B, G, T, D)
        Output: (B, H, T, D)  where H = G * n_rep
        """
        B, G, T, D = x.shape
        if self.n_rep == 1:
            return x
        # Unsqueeze → expand → reshape
        return (
            x.unsqueeze(2)                                     # (B, G, 1, T, D)
             .expand(B, G, self.n_rep, T, D)                   # (B, G, n_rep, T, D)
             .reshape(B, G * self.n_rep, T, D)                 # (B, H, T, D)
        )

    def forward(
        self,
        x:    torch.Tensor,             # (B, T, C)
        mask: Optional[torch.Tensor] = None,   # (1, 1, T, T) causal mask
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # q: (B, H, T, D)    k, v: (B, G, T, D)

        # Expand K, V to match H query heads
        k = self._repeat_kv(k)   # (B, H, T, D)
        v = self._repeat_kv(v)   # (B, H, T, D)

        # Scaled dot-product attention
        # Use PyTorch 2.0 flash-attention path when available (auto-selects)
        if hasattr(F, "scaled_dot_product_attention"):
            # is_causal=True applies the causal mask automatically (faster)
            is_causal = mask is None
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask if not is_causal else None,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = is_causal,
            )
        else:
            # Manual fallback (PyTorch < 2.0)
            scale = 1.0 / math.sqrt(self.d_head)
            attn  = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out  = torch.matmul(attn, v)                           # (B,H,T,D)

        # Re-assemble heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)      # (B,T,C)
        out = self.resid_drop(self.out_proj(out))
        return out


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention bias mask.
    Positions in the upper triangle get -inf so softmax ignores them.

    Returns shape: (1, 1, seq_len, seq_len)  — broadcast over B and H.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)   # upper triangle = -inf
    return mask.unsqueeze(0).unsqueeze(0) # add batch + head dims
