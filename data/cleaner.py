"""
Legal text cleaning pipeline.

Legal documents have specific noise patterns:
  - Page headers/footers ("Page 3 of 17", "SEBI/HO/MIRSD/...")
  - Reference numbers ("[Ref: SEBI/CIRCULAR/2023/01]")
  - Excessive whitespace and line breaks from PDF extraction
  - Roman numerals, section markers ("I.", "II.", "(a)", "(i)")

This module cleans raw scraped text for tokenizer training and model input.
"""

import re
from typing import List


# ------------------------------------------------------------------ #
#  Individual cleaning functions (composable pipeline)                #
# ------------------------------------------------------------------ #

def remove_page_artifacts(text: str) -> str:
    """
    Remove PDF extraction artifacts:
    - Page numbers: "Page 3 of 17", "- 3 -", "3 |"
    - Header/footer repetitions
    - Form feed characters
    """
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-—]\s*\d+\s*[-—]\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\f", "\n", text)         # form feed -> newline
    text = re.sub(r"\r\n?", "\n", text)      # normalize line endings
    return text


def remove_reference_codes(text: str) -> str:
    """
    Remove SEBI/GDPR reference codes that add noise without semantic content.
    Examples:  SEBI/HO/MIRSD/DOP/P/CIR/2023/145
               Ref: 2016/679/EU
    """
    text = re.sub(
        r"\bSEBI/[A-Z/\d]+\b",
        " <SEBI_REF> ",
        text,
    )
    text = re.sub(
        r"\b\d{4}/\d+/EU\b",
        " <EU_REF> ",
        text,
    )
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs; keep single newlines as sentence breaks."""
    text = re.sub(r"[ \t]+", " ", text)            # multiple spaces -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3+ newlines -> 2
    text = re.sub(r" \n ", "\n", text)
    return text.strip()


def remove_non_ascii_control(text: str) -> str:
    """Remove non-printable control characters (keep normal unicode)."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def lowercase_normalize(text: str) -> str:
    """Lower-case the entire document (consistent with BPE training)."""
    return text.lower()


def remove_very_short_lines(text: str, min_chars: int = 10) -> str:
    """
    Drop lines that are too short to carry semantic content
    (e.g. isolated page numbers, single letters).
    """
    lines = text.split("\n")
    kept  = [ln for ln in lines if len(ln.strip()) >= min_chars]
    return "\n".join(kept)


# ------------------------------------------------------------------ #
#  Composed pipeline                                                   #
# ------------------------------------------------------------------ #

def clean_document(text: str, for_inference: bool = False) -> str:
    """
    Full cleaning pipeline for a single legal document.

    Args:
        text          : Raw text from scraper or user input
        for_inference : If True, skip lowercasing (preserve original casing
                        for human-readable API responses).
    """
    text = remove_non_ascii_control(text)
    text = remove_page_artifacts(text)
    text = remove_reference_codes(text)
    text = normalize_whitespace(text)
    text = remove_very_short_lines(text, min_chars=10)
    if not for_inference:
        text = lowercase_normalize(text)
    return text


def clean_corpus(raw_texts: List[str]) -> str:
    """
    Clean and concatenate a list of documents into one training corpus string.
    Documents are separated by a blank line for BPE word-frequency counting.
    """
    cleaned = [clean_document(t) for t in raw_texts if t.strip()]
    return "\n\n".join(cleaned)


def sentence_split(text: str) -> List[str]:
    """
    Heuristic sentence splitter for legal text.
    Legal sentences often end with a period followed by a capital letter,
    or with semicolons that act as clause terminators.
    Returns a list of sentence strings.
    """
    # Split on ". " followed by a capital letter or a number
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """
    Split text into overlapping token-approximate chunks for pretraining.
    We use word count as a proxy for token count (roughly 1:1.3 ratio).

    Args:
        text       : Cleaned document text
        chunk_size : Target words per chunk
        overlap    : Words to repeat at chunk boundaries (context continuity)
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks
