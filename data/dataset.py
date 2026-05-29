"""
PyTorch Dataset classes for LegalMind.

PretrainDataset  — sliding window over a tokenized corpus (next-token prediction)
ComplianceDataset — labeled (text, label) pairs for violation classification

Labels:
  0 = compliant
  1 = violation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path


# ------------------------------------------------------------------ #
#  Pretraining dataset                                                 #
# ------------------------------------------------------------------ #

class PretrainDataset(Dataset):
    """
    Sliding window dataset for causal language model pretraining.

    Given a flat list of token IDs, returns (input_ids, target_ids) pairs
    where target_ids = input_ids shifted by 1 (the standard LM objective).

    Example with seq_len=4, stride=2:
      token stream: [1, 2, 3, 4, 5, 6, 7, 8]
      windows:      [1,2,3,4], [3,4,5,6], [5,6,7,8]
      targets:      [2,3,4,5], [4,5,6,7], [6,7,8,?]
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_len: int = 256,
        stride: Optional[int] = None,   # default: stride = seq_len (no overlap)
    ):
        self.ids    = token_ids
        self.seq_len = seq_len
        self.stride  = stride or seq_len

        # Pre-compute starting indices of all windows
        self.starts = list(range(0, len(token_ids) - seq_len, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.starts[idx]
        end   = start + self.seq_len + 1           # +1 to get target
        chunk = self.ids[start : end]

        input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:],  dtype=torch.long)

        return {"input_ids": input_ids, "labels": target_ids}

    @classmethod
    def from_file(
        cls,
        corpus_path: str,
        tokenizer,                         # BPETokenizer instance
        seq_len: int = 256,
        stride: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> "PretrainDataset":
        """
        Build dataset from a plain text corpus file.
        Tokenizes the file and caches the token ID list as a .pt file
        to avoid re-tokenizing on every run.
        """
        if cache_path and Path(cache_path).exists():
            print(f"[Dataset] Loading cached token IDs from {cache_path}")
            ids = torch.load(cache_path).tolist()
        else:
            print(f"[Dataset] Tokenizing corpus: {corpus_path}")
            text = Path(corpus_path).read_text(encoding="utf-8")
            # Encode without special tokens (they'd fragment the sliding window)
            ids  = tokenizer.encode(text, add_special_tokens=False)
            print(f"[Dataset] Total tokens: {len(ids):,}")
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.tensor(ids, dtype=torch.long), cache_path)

        return cls(ids, seq_len=seq_len, stride=stride)


# ------------------------------------------------------------------ #
#  Fine-tuning dataset                                                 #
# ------------------------------------------------------------------ #

class ComplianceDataset(Dataset):
    """
    Classification dataset: (text snippet → violation/compliant label).

    Expected JSON format (one object per line, or a list):
    [
      {"text": "The entity failed to disclose...", "label": 1},
      {"text": "Consent was obtained prior to...", "label": 0},
      ...
    ]

    Labels: 0 = compliant, 1 = violation
    """

    LABEL_MAP = {"compliant": 0, "violation": 1, "0": 0, "1": 1}

    def __init__(
        self,
        samples: List[Dict],            # list of {"text": str, "label": int}
        tokenizer,                      # BPETokenizer instance
        max_length: int = 256,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.texts:  List[str] = []
        self.labels: List[int] = []

        for s in samples:
            text  = s["text"].strip()
            label = s["label"]
            if isinstance(label, str):
                label = self.LABEL_MAP.get(label.lower(), 0)
            self.texts.append(text)
            self.labels.append(int(label))

        print(f"[Dataset] Loaded {len(self.texts)} samples  |  "
              f"violations={sum(self.labels)}  compliant={len(self.labels)-sum(self.labels)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
        )
        attention_mask = [1 if x != self.tokenizer.pad_id else 0 for x in ids]

        return {
            "input_ids":      torch.tensor(ids,            dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer,
        max_length: int = 256,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> Tuple["ComplianceDataset", "ComplianceDataset"]:
        """
        Load from a JSON file and split into train / validation datasets.

        Returns: (train_dataset, val_dataset)
        """
        import random
        random.seed(seed)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        random.shuffle(data)
        split = int(len(data) * (1 - val_split))
        train_data = data[:split]
        val_data   = data[split:]

        train_ds = cls(train_data, tokenizer, max_length)
        val_ds   = cls(val_data,   tokenizer, max_length)

        print(f"[Dataset] Train={len(train_ds)}  Val={len(val_ds)}")
        return train_ds, val_ds

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights to handle label imbalance.
        Returns a weight tensor of shape [num_classes].
        Used with nn.CrossEntropyLoss(weight=...).
        """
        n_total   = len(self.labels)
        n_classes = max(self.labels) + 1
        counts    = [self.labels.count(c) for c in range(n_classes)]
        weights   = [n_total / (n_classes * max(c, 1)) for c in counts]
        return torch.tensor(weights, dtype=torch.float)


# ------------------------------------------------------------------ #
#  DataLoader factories                                               #
# ------------------------------------------------------------------ #

def make_pretrain_loaders(
    dataset: PretrainDataset,
    batch_size: int = 8,
    val_fraction: float = 0.05,
    num_workers: int = 0,        # 0 = main process (safe on Windows/low RAM)
) -> Tuple[DataLoader, DataLoader]:
    """Split PretrainDataset into train/val and return DataLoaders."""
    val_size   = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    print(f"[DataLoader] train={len(train_ds)} | val={len(val_ds)} | "
          f"batch={batch_size}")
    return train_dl, val_dl


def make_finetune_loaders(
    train_ds: ComplianceDataset,
    val_ds: ComplianceDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return train / val DataLoaders for fine-tuning."""
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    return train_dl, val_dl
