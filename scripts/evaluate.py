"""
Script: Evaluate a fine-tuned model checkpoint.

Usage:
  python scripts/evaluate.py --data data/finetune/labeled.json
  python scripts/evaluate.py --data data/finetune/labeled.json \
         --checkpoint checkpoints/finetune/best.pt
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--checkpoint", default="checkpoints/finetune/best.pt")
    parser.add_argument("--tokenizer",  default="tokenizer.json")
    parser.add_argument("--output",     default="evaluation/results.json")
    args = parser.parse_args()

    from config import MODEL_CFG
    from tokenizer.bpe import BPETokenizer
    from model.gpt import build_model, load_checkpoint
    from data.dataset import ComplianceDataset
    from torch.utils.data import DataLoader
    from evaluation.metrics import run_evaluation

    tok = BPETokenizer.load(args.tokenizer)
    MODEL_CFG.vocab_size = len(tok)

    # Load full dataset as test set
    import json
    with open(args.data) as f:
        data = json.load(f)
    test_ds = ComplianceDataset(data, tok, MODEL_CFG.context_length)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_model(MODEL_CFG, num_classes=MODEL_CFG.num_classes)
    load_checkpoint(model, args.checkpoint, device=torch.device("cpu"))

    run_evaluation(model, test_dl, tok, torch.device("cpu"), save_path=args.output)


if __name__ == "__main__":
    main()
