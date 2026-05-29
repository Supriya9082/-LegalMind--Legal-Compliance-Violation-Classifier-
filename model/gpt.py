"""
LegalMind GPT — 15M parameter transformer for legal compliance.

Architecture:
  Token Embedding + Learned Positional Embedding
  → N × TransformerBlock (GQA + FFN + RMSNorm)
  → Final RMSNorm
  → LM Head (weight-tied to token embedding)

Two modes:
  pretrain  → forward() returns logits over vocab (cross-entropy LM loss)
  finetune  → forward() with classification head returns class logits
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .layers import TransformerBlock, RMSNorm
from .attention import make_causal_mask


class LegalMindGPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Args:
        vocab_size     : BPE vocabulary size
        context_length : Maximum sequence length
        n_layers       : Number of transformer blocks
        n_heads        : Query attention heads
        n_kv_heads     : Key/Value heads (GQA)
        d_model        : Hidden dimension
        d_ff           : Feed-forward inner dimension
        dropout        : Dropout probability
        bias           : Bias in linear layers
        num_classes    : If > 0, adds a classification head for fine-tuning
    """

    def __init__(
        self,
        vocab_size:     int,
        context_length: int   = 256,
        n_layers:       int   = 6,
        n_heads:        int   = 8,
        n_kv_heads:     int   = 2,
        d_model:        int   = 512,
        d_ff:           int   = 1024,
        dropout:        float = 0.1,
        bias:           bool  = False,
        num_classes:    int   = 0,   # 0 = pretrain only
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model        = d_model
        self.num_classes    = num_classes

        # ── Embeddings ────────────────────────────────────────────────
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # ── Transformer blocks ────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, bias)
            for _ in range(n_layers)
        ])

        # ── Output normalization ──────────────────────────────────────
        self.norm = RMSNorm(d_model)

        # ── Language model head (weight-tied to tok_emb) ──────────────
        # Weight tying: lm_head shares weights with tok_emb.
        # This reduces parameters by vocab_size * d_model (~4M for vocab=8000)
        # and empirically improves perplexity (Press & Wolf, 2017).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tie weights

        # ── Classification head (fine-tuning only) ───────────────────
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )
        else:
            self.cls_head = None

        # ── Weight initialization ─────────────────────────────────────
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) — GPT-2 trick
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        print(f"[GPT] Parameters: {self.count_params():,}")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:       torch.Tensor,            # (B, T)
        attention_mask:  Optional[torch.Tensor] = None,  # (B, T) 1=real 0=pad
        labels:          Optional[torch.Tensor] = None,  # (B, T) LM or (B,) cls
        use_checkpoint:  bool = False,
        mode:            str  = "pretrain",       # "pretrain" | "finetune"
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Returns: (loss, logits)
          pretrain : loss = cross-entropy LM loss;  logits shape (B, T, vocab)
          finetune : loss = cross-entropy cls loss; logits shape (B, num_classes)
        """
        B, T = input_ids.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}"
        )

        # ── Embeddings ────────────────────────────────────────────────
        positions = torch.arange(T, device=input_ids.device)
        x = self.emb_drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        # ── Causal mask ───────────────────────────────────────────────
        # Only needed for pretraining (autoregressive).
        # For classification fine-tuning, we use the full context.
        mask = None
        if mode == "pretrain":
            mask = make_causal_mask(T, input_ids.device)

        # ── Transformer blocks ────────────────────────────────────────
        for block in self.blocks:
            x = block(x, mask=mask, use_checkpoint=use_checkpoint)

        x = self.norm(x)   # (B, T, C)

        # ── Pretrain: LM head ─────────────────────────────────────────
        if mode == "pretrain":
            logits = self.lm_head(x)   # (B, T, vocab_size)
            loss   = None
            if labels is not None:
                # Flatten for cross-entropy: ignore padding (label = -100)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return loss, logits

        # ── Finetune: classification head ─────────────────────────────
        if mode == "finetune":
            assert self.cls_head is not None, (
                "num_classes must be > 0 to use finetune mode"
            )
            # Pool over sequence: use mean of non-padding tokens
            if attention_mask is not None:
                mask_f = attention_mask.unsqueeze(-1).float()  # (B,T,1)
                pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)   # (B, C)

            logits = self.cls_head(pooled)   # (B, num_classes)
            loss   = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, logits

        raise ValueError(f"Unknown mode: {mode}. Use 'pretrain' or 'finetune'.")

    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt_ids:  torch.Tensor,   # (1, T_prompt)
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k:       int   = 50,
    ) -> torch.Tensor:
        """
        Autoregressive text generation (greedy / top-k sampling).

        Args:
            prompt_ids    : Tokenized prompt, shape (1, T)
            max_new_tokens: Number of new tokens to generate
            temperature   : >1 = more random, <1 = more peaked distribution
            top_k         : Sample from top-k logits only (0 = greedy)
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to context window if needed
            ctx   = ids[:, -self.context_length:]
            _, lm_logits = self.forward(ctx, mode="pretrain")

            # Take logits at the last position
            logits = lm_logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                # Zero out everything except top-k
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float("-inf")

            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    # ─────────────────────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        """Freeze all parameters except the classification head (for fine-tuning)."""
        for name, p in self.named_parameters():
            if "cls_head" not in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[GPT] Backbone frozen. Trainable params: {trainable:,}")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────
#  Factory helpers                                                      #
# ─────────────────────────────────────────────────────────────────────

def build_model(cfg, num_classes: int = 0) -> LegalMindGPT:
    """
    Build LegalMindGPT from a ModelConfig dataclass.
    num_classes=0 for pretrain, num_classes=2 for finetune.
    """
    return LegalMindGPT(
        vocab_size     = cfg.vocab_size,
        context_length = cfg.context_length,
        n_layers       = cfg.n_layers,
        n_heads        = cfg.n_heads,
        n_kv_heads     = cfg.n_kv_heads,
        d_model        = cfg.d_model,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        bias           = cfg.bias,
        num_classes    = num_classes,
    )


def save_checkpoint(model, optimizer, step: int, loss: float, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step":       step,
        "loss":       loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    print(f"[GPT] Checkpoint saved → {path}  (step={step}, loss={loss:.4f})")


def load_checkpoint(model, path: str, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[GPT] Checkpoint loaded ← {path}  (step={ckpt.get('step')}, loss={ckpt.get('loss'):.4f})")
    return ckpt.get("step", 0), ckpt.get("loss", float("inf"))
