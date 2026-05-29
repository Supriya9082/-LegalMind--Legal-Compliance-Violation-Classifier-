"""
Inference engine for LegalMind — CPU-optimised with bf16 conversion.

Loads a fine-tuned checkpoint and exposes a simple predict() interface
used by both the CLI and the FastAPI server.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

LABEL_MAP = {0: "compliant", 1: "violation"}
LABEL_EMOJI = {0: "✅", 1: "⚠️"}


class CompliancePredictor:
    """
    Wraps a fine-tuned LegalMindGPT for single and batch inference.

    Args:
        model_path     : Path to fine-tuned checkpoint (.pt)
        tokenizer_path : Path to tokenizer JSON
        use_bf16       : Convert model weights to bfloat16 for faster CPU inference
        max_length     : Token sequence length (must match training config)
    """

    def __init__(
        self,
        model_path:     str,
        tokenizer_path: str,
        use_bf16:       bool = True,
        max_length:     int  = 256,
    ):
        from tokenizer.bpe import BPETokenizer
        from model.gpt import LegalMindGPT, load_checkpoint
        from config import MODEL_CFG

        self.device     = torch.device("cpu")
        self.max_length = max_length

        # ── Load tokenizer ──────────────────────────────────────────
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        # ── Build model ─────────────────────────────────────────────
        self.model = LegalMindGPT(
            vocab_size     = MODEL_CFG.vocab_size,
            context_length = MODEL_CFG.context_length,
            n_layers       = MODEL_CFG.n_layers,
            n_heads        = MODEL_CFG.n_heads,
            n_kv_heads     = MODEL_CFG.n_kv_heads,
            d_model        = MODEL_CFG.d_model,
            d_ff           = MODEL_CFG.d_ff,
            dropout        = 0.0,          # no dropout at inference
            bias           = MODEL_CFG.bias,
            num_classes    = MODEL_CFG.num_classes,
        )

        load_checkpoint(self.model, model_path, device=self.device)
        self.model.eval()

        # ── bf16 conversion ─────────────────────────────────────────
        # bfloat16 on modern CPUs (AVX-512 BF16 support) is 1.5–2× faster
        # than float32 for inference. Falls back to float32 silently.
        if use_bf16:
            try:
                self.model = self.model.to(torch.bfloat16)
                self._dtype = torch.bfloat16
                print("[Predictor] Using bf16 inference")
            except Exception:
                self._dtype = torch.float32
                print("[Predictor] bf16 unavailable, using float32")
        else:
            self._dtype = torch.float32

        print(f"[Predictor] Ready | device={self.device} | dtype={self._dtype}")

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Predict compliance violation for a single text snippet.

        Returns:
            {
              "label"      : "violation" or "compliant",
              "label_id"   : 0 or 1,
              "confidence" : float (0-1),
              "scores"     : {"compliant": float, "violation": float},
              "latency_ms" : float,
            }
        """
        t0 = time.perf_counter()

        ids = self.tokenizer.encode(
            text,
            add_special_tokens = True,
            max_length         = self.max_length,
            padding            = True,
        )
        input_tensor = torch.tensor([ids], dtype=torch.long)
        attn_mask    = torch.tensor(
            [[1 if x != self.tokenizer.pad_id else 0 for x in ids]],
            dtype=torch.long,
        )

        if self._dtype == torch.bfloat16:
            input_tensor = input_tensor  # long stays long; model handles internally
        
        _, logits = self.model(
            input_tensor,
            attention_mask = attn_mask,
            mode           = "finetune",
        )

        probs     = F.softmax(logits.float(), dim=-1)[0]
        label_id  = probs.argmax().item()
        confidence = probs[label_id].item()

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "label":       LABEL_MAP[label_id],
            "label_id":    label_id,
            "confidence":  round(confidence, 4),
            "scores": {
                "compliant": round(probs[0].item(), 4),
                "violation": round(probs[1].item(), 4),
            },
            "latency_ms": round(latency_ms, 2),
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for a list of texts. More efficient than calling predict() N times."""
        return [self.predict(t) for t in texts]

    def predict_verbose(self, text: str) -> None:
        """Pretty-print prediction result (for CLI use)."""
        result = self.predict(text)
        emoji  = LABEL_EMOJI[result["label_id"]]
        print(f"\n{'─'*60}")
        print(f"  Text      : {text[:120]}{'...' if len(text)>120 else ''}")
        print(f"  Prediction: {emoji} {result['label'].upper()}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Scores    : compliant={result['scores']['compliant']:.4f}  "
              f"violation={result['scores']['violation']:.4f}")
        print(f"  Latency   : {result['latency_ms']:.1f} ms")
        print(f"{'─'*60}\n")
