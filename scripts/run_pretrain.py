"""
Script: Pretrain GPT on SEBI + GDPR legal corpus.

Usage:
  python scripts/run_pretrain.py
  python scripts/run_pretrain.py --resume checkpoints/pretrain/step_10000.pt
  python scripts/run_pretrain.py --max-steps 5000  # quick smoke test
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
        logging.FileHandler("pretrain.log"),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pretrain LegalMind GPT")
    parser.add_argument("--corpus",    default="data/processed/corpus.txt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    from config import MODEL_CFG, PRETRAIN_CFG
    from tokenizer.bpe import BPETokenizer
    from model.gpt import build_model
    from data.dataset import PretrainDataset, make_pretrain_loaders
    from training.pretrain import pretrain

    device = torch.device("cpu")
    logger.info(f"[Script] Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────
    if not os.path.exists(args.tokenizer):
        logger.error(f"Tokenizer not found at {args.tokenizer}. "
                     "Run scripts/train_tokenizer.py first.")
        sys.exit(1)

    tok = BPETokenizer.load(args.tokenizer)
    MODEL_CFG.vocab_size = len(tok)

    # ── Dataset ──────────────────────────────────────────────────────
    logger.info("[Script] Building dataset...")
    dataset = PretrainDataset.from_file(
        corpus_path = args.corpus,
        tokenizer   = tok,
        seq_len     = PRETRAIN_CFG.seq_len,
        cache_path  = "data/processed/tokens_cache.pt",
    )
    train_dl, val_dl = make_pretrain_loaders(
        dataset,
        batch_size = PRETRAIN_CFG.batch_size,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(MODEL_CFG, num_classes=0)

    # ── Override max_steps if specified ─────────────────────────────
    if args.max_steps:
        PRETRAIN_CFG.max_steps = args.max_steps

    logger.info(f"[Script] Pretraining for {PRETRAIN_CFG.max_steps} steps...")

    # ── Train ────────────────────────────────────────────────────────
    pretrain(model, train_dl, val_dl, PRETRAIN_CFG, device,
             resume_from=args.resume)

    logger.info("[Script] Pretraining complete ✓")


if __name__ == "__main__":
    main()
