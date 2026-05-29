"""
Legal document scrapers for SEBI and GDPR corpora.

Data sources:
  SEBI  — https://www.sebi.gov.in  (circulars, orders, enforcement)
  GDPR  — EUR-Lex (official EU law repository) + GDPR.eu text

Run once to build your local corpus:
    python scripts/train_tokenizer.py --scrape
"""

import os
import re
import time
import logging
import requests
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (LegalMind Research Scraper; "
        "contact: your@email.com)"
    )
}

# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #

def _get(url: str, timeout: int = 30, retries: int = 3) -> Optional[requests.Response]:
    """HTTP GET with retry on transient failures."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning(f"  Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(2 ** attempt)   # exponential back-off
    return None


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF byte string using PyPDF2."""
    if not PDF_OK:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


# ------------------------------------------------------------------ #
#  SEBI scraper                                                        #
# ------------------------------------------------------------------ #

# Public SEBI circular listing page
SEBI_CIRCULARS_URL = (
    "https://www.sebi.gov.in/sebiweb/other/OtherAction.do"
    "?doListingAll=yes&sid=3&ssid=15&smid=0"
)

SEBI_ENFORCEMENT_URL = (
    "https://www.sebi.gov.in/enforcement/orders.html"
)


def scrape_sebi_circulars(
    output_dir: str = "data/raw/sebi",
    max_docs: int = 200,
) -> List[str]:
    """
    Download SEBI circulars (HTML listing → individual PDF links → text).

    Returns a list of extracted text strings (one per document).
    Saves each text file under output_dir/ for caching.
    """
    if not BS4_OK:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    texts: List[str] = []

    logger.info(f"[SEBI] Fetching circular listing from {SEBI_CIRCULARS_URL}")
    resp = _get(SEBI_CIRCULARS_URL)
    if resp is None:
        logger.error("[SEBI] Could not reach SEBI website. Check your internet connection.")
        return texts

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    # SEBI listing tables contain <a href="...pdf"> or links to circular pages
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower() or "circular" in href.lower():
            full_url = urljoin("https://www.sebi.gov.in", href)
            links.append(full_url)

    links = list(dict.fromkeys(links))[:max_docs]   # deduplicate
    logger.info(f"[SEBI] Found {len(links)} document links")

    for i, url in enumerate(links):
        cache_path = Path(output_dir) / f"sebi_{i:04d}.txt"
        if cache_path.exists():
            texts.append(cache_path.read_text(encoding="utf-8"))
            continue

        logger.info(f"  [{i+1}/{len(links)}] {url}")
        doc_resp = _get(url)
        if doc_resp is None:
            continue

        text = ""
        ct = doc_resp.headers.get("Content-Type", "")
        if "pdf" in ct or url.lower().endswith(".pdf"):
            try:
                text = _pdf_to_text(doc_resp.content)
            except Exception as e:
                logger.warning(f"  PDF parse error: {e}")
        else:
            # HTML circular page
            doc_soup = BeautifulSoup(doc_resp.text, "html.parser")
            # Remove nav / footer boilerplate
            for tag in doc_soup.find_all(["nav", "footer", "script", "style"]):
                tag.decompose()
            text = doc_soup.get_text(separator="\n")

        if text.strip():
            cache_path.write_text(text, encoding="utf-8")
            texts.append(text)

        time.sleep(1.0)   # be polite

    logger.info(f"[SEBI] Scraped {len(texts)} documents")
    return texts


# ------------------------------------------------------------------ #
#  GDPR scraper                                                        #
# ------------------------------------------------------------------ #

# EUR-Lex: full text of GDPR (Regulation 2016/679)
GDPR_EURLEX_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:32016R0679"
)

# Also the well-structured GDPR.eu site
GDPR_EU_ARTICLES_URL = "https://gdpr.eu/tag/gdpr/"


def scrape_gdpr(output_dir: str = "data/raw/gdpr") -> List[str]:
    """
    Download full GDPR text from EUR-Lex and article summaries from gdpr.eu.
    Returns list of text strings.
    """
    if not BS4_OK:
        raise ImportError("beautifulsoup4 not installed.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    texts: List[str] = []

    # --- EUR-Lex full text ---
    cache = Path(output_dir) / "gdpr_eurlex.txt"
    if cache.exists():
        logger.info("[GDPR] Loading cached EUR-Lex text")
        texts.append(cache.read_text(encoding="utf-8"))
    else:
        logger.info(f"[GDPR] Fetching EUR-Lex full text from {GDPR_EURLEX_URL}")
        resp = _get(GDPR_EURLEX_URL)
        if resp:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["script", "style", "nav"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            if text.strip():
                cache.write_text(text, encoding="utf-8")
                texts.append(text)
            logger.info(f"[GDPR] EUR-Lex text: {len(text):,} chars")

    # --- GDPR.eu article pages ---
    logger.info("[GDPR] Fetching article list from gdpr.eu")
    resp = _get(GDPR_EU_ARTICLES_URL)
    if resp:
        soup = BeautifulSoup(resp.text, "html.parser")
        article_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "gdpr.eu/article" in href or "gdpr.eu/recital" in href:
                article_links.append(href)
        article_links = list(dict.fromkeys(article_links))[:50]

        for j, url in enumerate(article_links):
            acache = Path(output_dir) / f"gdpr_article_{j:03d}.txt"
            if acache.exists():
                texts.append(acache.read_text(encoding="utf-8"))
                continue

            logger.info(f"  [{j+1}/{len(article_links)}] {url}")
            ar = _get(url)
            if ar:
                asoup = BeautifulSoup(ar.text, "html.parser")
                for tag in asoup.find_all(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = asoup.get_text(separator="\n")
                if text.strip():
                    acache.write_text(text, encoding="utf-8")
                    texts.append(text)
            time.sleep(0.5)

    logger.info(f"[GDPR] Scraped {len(texts)} documents")
    return texts


# ------------------------------------------------------------------ #
#  Combined corpus builder                                             #
# ------------------------------------------------------------------ #

def build_corpus(
    output_path: str = "data/processed/corpus.txt",
    sebi_dir: str   = "data/raw/sebi",
    gdpr_dir: str   = "data/raw/gdpr",
    sebi_max: int   = 200,
    scrape: bool    = True,
) -> str:
    """
    Build and save a combined SEBI + GDPR text corpus.

    Args:
        output_path : Where to write the final corpus.txt
        sebi_dir    : Cache directory for SEBI docs
        gdpr_dir    : Cache directory for GDPR docs
        sebi_max    : Max SEBI circulars to download
        scrape      : If False, load from existing cache only

    Returns:
        Full corpus as a single string.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(output_path).exists() and not scrape:
        logger.info(f"[Corpus] Loading existing corpus from {output_path}")
        return Path(output_path).read_text(encoding="utf-8")

    sebi_texts = scrape_sebi_circulars(sebi_dir, max_docs=sebi_max) if scrape else []
    gdpr_texts = scrape_gdpr(gdpr_dir)                               if scrape else []

    # Also load any existing cached files even if scrape=True
    if not sebi_texts:
        sebi_texts = [p.read_text(encoding="utf-8")
                      for p in Path(sebi_dir).glob("*.txt") if p.exists()]
    if not gdpr_texts:
        gdpr_texts = [p.read_text(encoding="utf-8")
                      for p in Path(gdpr_dir).glob("*.txt") if p.exists()]

    all_texts = sebi_texts + gdpr_texts
    corpus    = "\n\n".join(all_texts)

    Path(output_path).write_text(corpus, encoding="utf-8")
    logger.info(f"[Corpus] Saved {len(corpus):,} chars → {output_path}")
    return corpus
