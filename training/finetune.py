"""
Fine-tuning loop — compliance violation classification.

Loads pretrained GPT weights, attaches classification head,
trains with weighted cross-entropy to handle class imbalance.

Strategy:
  Phase 1 (epoch 1-2) : Freeze backbone, train only cls_head (fast convergence)
  Phase 2 (epoch 3+)  : Unfreeze all, low LR (full fine-tuning)
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cls(
    model,
    loader:   DataLoader,
    device:   torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float, float]:
    """
    Evaluate classification model.

    Returns: (loss, accuracy, f1_macro, f1_violation)
    """
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        loss, logits = model(
            input_ids,
            attention_mask = attn_mask,
            labels         = labels,
            mode           = "finetune",
        )
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    model.train()

    avg_loss = total_loss / max(len(loader), 1)
    acc      = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_viol  = f1_score(all_labels, all_preds, average="binary",   zero_division=0)

    return avg_loss, acc, f1_macro, f1_viol


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def finetune(
    model,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    cfg,                           # FinetuneConfig
    device:        torch.device,
    class_weights: torch.Tensor = None,
) -> None:
    """
    Fine-tune LegalMindGPT for compliance violation classification.

    Args:
        model          : LegalMindGPT with num_classes=2
        train_loader   : Training DataLoader (ComplianceDataset)
        val_loader     : Validation DataLoader
        cfg            : FinetuneConfig dataclass
        device         : torch.device
        class_weights  : Optional per-class weights for imbalanced data
    """
    from model.gpt import save_checkpoint

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    model.to(device)

    # ── Loss function ───────────────────────────────────────────────
    if class_weights is not None and cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"[Finetune] Class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Phase 1: freeze backbone (train cls_head only) ──────────────
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate * 5,    # higher LR for head-only training
        weight_decay=cfg.weight_decay,
    )

    best_f1        = 0.0
    no_improve     = 0
    global_step    = 0

    logger.info(f"[Finetune] Starting | epochs={cfg.max_epochs} | "
                f"batch={cfg.batch_size} | device={device}")

    for epoch in range(1, cfg.max_epochs + 1):

        # ── Phase transition: unfreeze after epoch 2 ────────────────
        if epoch == 3:
            logger.info("[Finetune] Phase 2: unfreezing backbone (full fine-tune)")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,    # lower LR for backbone
                weight_decay=cfg.weight_decay,
            )

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            labels     = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, logits = model(
                input_ids,
                attention_mask = attn_mask,
                labels         = labels,
                mode           = "finetune",
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            if (step + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch} | step {step+1}/{len(train_loader)} | "
                            f"loss={loss.item():.4f}")

        # ── Epoch evaluation ─────────────────────────────────────────
        val_loss, acc, f1_macro, f1_viol = evaluate_cls(
            model, val_loader, device, criterion
        )

        logger.info(
            f"[Epoch {epoch}] val_loss={val_loss:.4f}  acc={acc:.3f}  "
            f"f1_macro={f1_macro:.3f}  f1_violation={f1_viol:.3f}"
        )

        # ── Save best model ──────────────────────────────────────────
        if f1_viol > best_f1:
            best_f1    = f1_viol
            no_improve = 0
            save_checkpoint(model, optimizer, global_step, val_loss,
                            os.path.join(cfg.checkpoint_dir, "best.pt"))
            logger.info(f"  ✓ Best model saved (f1_violation={best_f1:.3f})")
        else:
            no_improve += 1

        # ── Early stopping ───────────────────────────────────────────
        if no_improve >= cfg.early_stopping_patience:
            logger.info(f"[Finetune] Early stopping at epoch {epoch} "
                        f"(no improvement for {no_improve} epochs)")
            break

    logger.info(f"[Finetune] Complete. Best F1 (violation): {best_f1:.3f}")
