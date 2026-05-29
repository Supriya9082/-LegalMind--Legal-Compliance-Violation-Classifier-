"""
Pretraining loop — next-token prediction on legal corpus.

Uses:
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Gradient checkpointing (memory-efficient training on 8 GB RAM)
  - bf16 autocast on supported hardware
  - Periodic evaluation + checkpointing
"""

import os
import math
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  LR Schedule: linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.
    Standard schedule for small transformer training.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device: torch.device,
             use_bf16: bool = False) -> float:
    """Run model on validation set; return mean loss."""
    model.eval()
    total_loss, n_batches = 0.0, 0

    dtype = torch.bfloat16 if (use_bf16 and device.type == "cpu") else torch.float32

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_bf16):
            loss, _ = model(input_ids, labels=labels, mode="pretrain")

        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def pretrain(
    model,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    cfg,                          # PretrainConfig dataclass
    device:        torch.device,
    resume_from:   Optional[str] = None,
) -> None:
    """
    Pretrain LegalMindGPT on legal corpus.

    Args:
        model        : LegalMindGPT instance (pretrain mode, no cls_head)
        train_loader : Training DataLoader (PretrainDataset)
        val_loader   : Validation DataLoader
        cfg          : PretrainConfig
        device       : torch.device (cpu / cuda / mps)
        resume_from  : Path to checkpoint to resume from
    """
    from model.gpt import save_checkpoint, load_checkpoint

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    model.to(device)
    model.train()

    # ── Optimizer ──────────────────────────────────────────────────
    # AdamW with weight decay on non-bias / non-norm parameters only
    decay_params  = [p for n, p in model.named_parameters()
                     if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=(0.9, 0.95), fused=False)

    # ── Resume from checkpoint if requested ────────────────────────
    start_step = 0
    best_val   = float("inf")
    if resume_from and os.path.exists(resume_from):
        start_step, _ = load_checkpoint(model, resume_from, optimizer, device)

    # ── bf16 autocast (CPU bf16 supported from PyTorch 2.1+) ──────
    use_bf16 = cfg.get("use_bf16", False) if hasattr(cfg, "get") else False
    # Detect bf16 support
    try:
        _t = torch.zeros(1, dtype=torch.bfloat16)
        use_bf16 = True
        logger.info("[Pretrain] bf16 autocast ENABLED")
    except Exception:
        use_bf16 = False
        logger.info("[Pretrain] bf16 not supported — using float32")

    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ── Training loop ──────────────────────────────────────────────
    step          = start_step
    total_loss    = 0.0
    t0            = time.time()
    train_iter    = iter(train_loader)

    logger.info(f"[Pretrain] Starting at step {step} / {cfg.max_steps}")
    logger.info(f"[Pretrain] Device={device} | batch={cfg.batch_size} | "
                f"seq={cfg.seq_len} | grad_ckpt={cfg.gradient_checkpointing}")

    while step < cfg.max_steps:

        # ── LR update ──────────────────────────────────────────────
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps,
                    cfg.learning_rate, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Get next batch (cycle through dataset infinitely) ──────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # ── Forward pass ────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_bf16):
            loss, _ = model(
                input_ids,
                labels          = labels,
                use_checkpoint  = cfg.gradient_checkpointing,
                mode            = "pretrain",
            )

        # ── Backward + gradient clip ─────────────────────────────
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        step       += 1

        # ── Logging ─────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            avg_loss = total_loss / cfg.log_interval
            elapsed  = time.time() - t0
            tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.log_interval) / elapsed
            ppl      = math.exp(min(avg_loss, 20))
            logger.info(
                f"  step={step:>6}  loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                f"lr={lr:.2e}  tok/s={tokens_per_sec:.0f}  "
                f"elapsed={elapsed:.1f}s"
            )
            total_loss = 0.0
            t0 = time.time()

        # ── Validation ───────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, use_bf16)
            val_ppl  = math.exp(min(val_loss, 20))
            logger.info(f"  [EVAL] step={step}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                os.path.join(cfg.checkpoint_dir, "best.pt"))

        # ── Periodic save ────────────────────────────────────────
        if step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, step, loss.item(),
                            os.path.join(cfg.checkpoint_dir, f"step_{step}.pt"))

    # Final save
    save_checkpoint(model, optimizer, step, loss.item(),
                    os.path.join(cfg.checkpoint_dir, "final.pt"))
    logger.info(f"[Pretrain] Complete. Best val_loss={best_val:.4f}")
