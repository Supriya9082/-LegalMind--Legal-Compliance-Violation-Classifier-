"""
Custom Byte Pair Encoding (BPE) Tokenizer — built from scratch.

Algorithm overview:
  1. Split corpus into words; represent each word as space-separated chars + </w>
  2. Count every adjacent symbol pair across all words
  3. Merge the most frequent pair into a single new symbol
  4. Repeat until vocab_size is reached

The </w> (end-of-word) marker lets the tokenizer distinguish tokens that
appear mid-word from tokens at word boundaries — crucial for legal terms
like "disclosure</w>" vs "disclosure" inside "non-disclosure".
"""

import re
import json
import collections
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer trained on legal vocabulary.

    Usage:
        tok = BPETokenizer(vocab_size=8000)
        tok.train(corpus_text)
        ids = tok.encode("The entity violated SEBI regulations.")
        text = tok.decode(ids)
        tok.save("tokenizer.json")
        tok = BPETokenizer.load("tokenizer.json")
    """

    SPECIAL_TOKENS = {
        "<PAD>":  0,   # Padding (for batching)
        "<UNK>":  1,   # Unknown token
        "<BOS>":  2,   # Beginning of sequence
        "<EOS>":  3,   # End of sequence
        "<CLS>":  4,   # Classification token (prepended for fine-tuning)
        "<SEP>":  5,   # Separator between segments
        "<MASK>": 6,   # Masked token (for future MLM tasks)
    }
    N_SPECIAL = len(SPECIAL_TOKENS)

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        # merge table: (a, b) -> ab  (ordered: insertion order = merge priority)
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[str, int] = {}        # token string -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token string

        # Convenient ID aliases
        self.pad_id  = self.SPECIAL_TOKENS["<PAD>"]
        self.unk_id  = self.SPECIAL_TOKENS["<UNK>"]
        self.bos_id  = self.SPECIAL_TOKENS["<BOS>"]
        self.eos_id  = self.SPECIAL_TOKENS["<EOS>"]
        self.cls_id  = self.SPECIAL_TOKENS["<CLS>"]
        self.sep_id  = self.SPECIAL_TOKENS["<SEP>"]
        self.mask_id = self.SPECIAL_TOKENS["<MASK>"]

    # ------------------------------------------------------------------ #
    #  Training helpers                                                    #
    # ------------------------------------------------------------------ #

    def _word_freq(self, corpus: str) -> Dict[str, int]:
        """
        Build a frequency table of character-segmented words.

        "sebi" -> "s e b i </w>" : 1
        Whitespace tokens are skipped; everything is lowercased.
        """
        freq: Dict[str, int] = collections.defaultdict(int)
        for word in re.findall(r"\S+", corpus.lower()):
            key = " ".join(list(word)) + " </w>"
            freq[key] += 1
        return dict(freq)

    def _pair_freq(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count every adjacent (A, B) pair weighted by word frequency."""
        pairs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        for word, count in word_freq.items():
            syms = word.split()
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += count
        return pairs

    def _apply_merge(
        self,
        pair: Tuple[str, str],
        word_freq: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Replace every occurrence of (A B) with AB inside word_freq.

        We guard with spaces so "a b" inside "x a b c" becomes "x ab c"
        and NOT accidentally merged with neighbouring tokens.
        """
        a, b = pair
        target      = a + " " + b
        replacement = a + b
        new_freq: Dict[str, int] = {}
        for word, count in word_freq.items():
            # Pad with spaces so edges are handled uniformly
            padded = " " + word + " "
            merged = padded.replace(" " + target + " ", " " + replacement + " ")
            # Some words have consecutive same-pair occurrences; loop until stable
            while target in merged:
                merged = merged.replace(target, replacement)
            new_freq[merged.strip()] = count
        return new_freq

    # ------------------------------------------------------------------ #
    #  Public training entry-point                                        #
    # ------------------------------------------------------------------ #

    def train(self, corpus: str, verbose: bool = True) -> None:
        """
        Learn BPE merge rules from a text corpus.

        Args:
            corpus  : Raw concatenated text (SEBI + GDPR documents)
            verbose : Show a tqdm progress bar
        """
        print(f"[BPE] Training | corpus={len(corpus):,} chars | target_vocab={self.vocab_size}")

        # --- Step 1: seed vocab with special tokens ----------------------
        self.vocab = dict(self.SPECIAL_TOKENS)

        # --- Step 2: add every unique character as a base token ----------
        word_freq = self._word_freq(corpus)
        base_chars: set = set()
        for word in word_freq:
            base_chars.update(word.split())

        for ch in sorted(base_chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        print(f"[BPE] Base vocab (chars + specials): {len(self.vocab)}")

        # --- Step 3: iterative merging -----------------------------------
        n_merges = self.vocab_size - len(self.vocab)
        for i in tqdm(range(n_merges), desc="[BPE] Merging", disable=not verbose):
            pairs = self._pair_freq(word_freq)
            if not pairs:
                print(f"[BPE] Corpus exhausted at merge {i}. Done.")
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.N_SPECIAL:  # ignore extremely rare pairs
                print(f"[BPE] Stopping: best pair freq={pairs[best_pair]} < threshold.")
                break

            new_tok = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_tok
            word_freq = self._apply_merge(best_pair, word_freq)

            if new_tok not in self.vocab:
                self.vocab[new_tok] = len(self.vocab)

        # Build reverse lookup
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"[BPE] Done. Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------ #
    #  Encoding                                                           #
    # ------------------------------------------------------------------ #

    def _encode_word(self, word: str) -> List[str]:
        """
        Apply learned merge rules to a single word (greedy, in merge order).

        Start: ['s', 'e', 'b', 'i', '</w>']
        After merges: ['sebi</w>']  (if learned as a whole unit)
        """
        syms = list(word) + ["</w>"]
        # Walk through merge table in insertion (priority) order
        for (a, b), merged in self.merges.items():
            i = 0
            new_syms: List[str] = []
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
            if len(syms) == 1:
                break
        return syms

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode a string to a list of integer token IDs.

        Args:
            text               : Input text
            add_special_tokens : Wrap with BOS / EOS
            max_length         : Truncate to this length (inclusive of special tokens)
            padding            : Pad to max_length with pad_id
        """
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for word in re.findall(r"\S+", text.lower()):
            for tok in self._encode_word(word):
                ids.append(self.vocab.get(tok, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
                if add_special_tokens:
                    ids[-1] = self.eos_id   # keep EOS at the truncated end
            if padding:
                ids += [self.pad_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to human-readable text."""
        special = set(self.SPECIAL_TOKENS.values())
        parts: List[str] = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            parts.append(self.inverse_vocab.get(i, "<UNK>"))
        return "".join(parts).replace("</w>", " ").strip()

    def batch_encode(
        self,
        texts: List[str],
        max_length: int,
        add_special_tokens: bool = True,
    ) -> Dict[str, list]:
        """
        Encode a list of texts into padded input_ids + attention_mask tensors.

        Returns dict with keys: 'input_ids', 'attention_mask'
        Both are lists of lists (convert to torch.Tensor in the Dataset class).
        """
        all_ids, all_mask = [], []
        for text in texts:
            ids  = self.encode(text, add_special_tokens=add_special_tokens,
                               max_length=max_length, padding=True)
            mask = [1 if x != self.pad_id else 0 for x in ids]
            all_ids.append(ids)
            all_mask.append(mask)
        return {"input_ids": all_ids, "attention_mask": all_mask}

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serialize tokenizer state to JSON (vocab + merge rules)."""
        state = {
            "vocab_size": self.vocab_size,
            "vocab":      self.vocab,
            "merges":     [[a, b, m] for (a, b), m in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[BPE] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Deserialize a tokenizer from JSON."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        tok = cls(vocab_size=state["vocab_size"])
        tok.vocab         = {k: int(v) for k, v in state["vocab"].items()}
        tok.inverse_vocab = {int(v): k for k, v in state["vocab"].items()}
        tok.merges        = {(row[0], row[1]): row[2] for row in state["merges"]}
        print(f"[BPE] Loaded ← {path}  (vocab={len(tok.vocab)}, merges={len(tok.merges)})")
        return tok

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab_size={len(self.vocab)}, "
                f"merges={len(self.merges)}, "
                f"special_tokens={self.N_SPECIAL})")
