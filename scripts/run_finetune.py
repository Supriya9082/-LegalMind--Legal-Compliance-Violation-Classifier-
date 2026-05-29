"""
Script: Fine-tune pretrained GPT for compliance violation classification.

Usage:
  python scripts/run_finetune.py --data data/finetune/labeled.json
  python scripts/run_finetune.py --data data/finetune/labeled.json \
         --pretrain checkpoints/pretrain/best.pt

Labeled data format (JSON list):
  [{"text": "...", "label": 0}, {"text": "...", "label": 1}, ...]
  label: 0=compliant, 1=violation
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune.log"),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LegalMind for classification")
    parser.add_argument("--data",      required=True, help="Path to labeled JSON file")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--pretrain",  default="checkpoints/pretrain/best.pt")
    args = parser.parse_args()

    from config import MODEL_CFG, FINETUNE_CFG
    from tokenizer.bpe import BPETokenizer
    from model.gpt import build_model, load_checkpoint
    from data.dataset import ComplianceDataset, make_finetune_loaders
    from training.finetune import finetune

    device = torch.device("cpu")

    # ── Tokenizer ────────────────────────────────────────────────────
    tok = BPETokenizer.load(args.tokenizer)
    MODEL_CFG.vocab_size = len(tok)

    # ── Dataset ──────────────────────────────────────────────────────
    train_ds, val_ds = ComplianceDataset.from_json(
        path       = args.data,
        tokenizer  = tok,
        max_length = MODEL_CFG.context_length,
    )
    train_dl, val_dl = make_finetune_loaders(train_ds, val_ds, FINETUNE_CFG.batch_size)

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(MODEL_CFG, num_classes=MODEL_CFG.num_classes)

    # Load pretrained backbone
    if os.path.exists(args.pretrain):
        logger.info(f"[Script] Loading pretrained weights from {args.pretrain}")
        load_checkpoint(model, args.pretrain, device=device)
    else:
        logger.warning(f"[Script] Pretrain checkpoint not found at {args.pretrain}. "
                       "Training from scratch (not recommended).")

    # ── Class weights ────────────────────────────────────────────────
    class_weights = train_ds.get_class_weights() if FINETUNE_CFG.use_class_weights else None

    # ── Fine-tune ────────────────────────────────────────────────────
    finetune(model, train_dl, val_dl, FINETUNE_CFG, device, class_weights)
    logger.info("[Script] Fine-tuning complete ✓")


if __name__ == "__main__":
    main()
