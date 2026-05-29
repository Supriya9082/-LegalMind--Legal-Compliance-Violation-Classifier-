"""
Script: Train BPE tokenizer on legal corpus.

Usage:
  python scripts/train_tokenizer.py
  python scripts/train_tokenizer.py --corpus data/processed/corpus.txt
  python scripts/train_tokenizer.py --scrape --sebi-max 100
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LegalMind BPE tokenizer")
    parser.add_argument("--corpus",     default="data/processed/corpus.txt")
    parser.add_argument("--output",     default="tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--scrape",     action="store_true",
                        help="Scrape SEBI/GDPR corpus before training")
    parser.add_argument("--sebi-max",   type=int, default=100)
    args = parser.parse_args()

    from config import TOKENIZER_CFG
    from tokenizer.bpe import BPETokenizer
    from data.cleaner import clean_corpus

    # ── Step 1: build / load corpus ─────────────────────────────────
    if args.scrape or not Path(args.corpus).exists():
        logger.info("[Script] Building corpus from web...")
        from data.scraper import build_corpus
        corpus = build_corpus(
            output_path = args.corpus,
            sebi_max    = args.sebi_max,
            scrape      = args.scrape,
        )
    else:
        logger.info(f"[Script] Loading corpus from {args.corpus}")
        corpus = Path(args.corpus).read_text(encoding="utf-8")

    logger.info(f"[Script] Corpus size: {len(corpus):,} chars")

    # ── Step 2: clean ────────────────────────────────────────────────
    from data.cleaner import clean_document
    corpus = clean_document(corpus)

    # ── Step 3: train tokenizer ──────────────────────────────────────
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.train(corpus)

    # ── Step 4: save ─────────────────────────────────────────────────
    tok.save(args.output)

    # ── Step 5: sanity check ─────────────────────────────────────────
    sample = "The entity violated SEBI regulations by failing to disclose material information."
    ids    = tok.encode(sample)
    recon  = tok.decode(ids)
    logger.info(f"[Script] Test encode: '{sample}'")
    logger.info(f"[Script] Token IDs ({len(ids)}): {ids[:10]}...")
    logger.info(f"[Script] Decoded: '{recon}'")
    logger.info("[Script] Tokenizer training complete ✓")


if __name__ == "__main__":
    main()
