"""
Real-time input drift monitoring.

Drift = the distribution of incoming texts shifts away from training data.
When drift is detected, model predictions become unreliable.

Method: compare token-frequency distributions using KL divergence.
  - Reference window: first N predictions after server start (assumed in-distribution)
  - Live window: rolling last N predictions
  - Alert when KL(live || reference) > threshold
"""

import math
import logging
from collections import deque, Counter
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def _token_dist(token_lists: List[List[int]], vocab_size: int, smoothing: float = 1e-8):
    """
    Build a smoothed normalized token frequency distribution.

    Args:
        token_lists : List of token ID sequences
        vocab_size  : Total vocabulary size (for smoothing)
        smoothing   : Laplace smoothing to avoid log(0)

    Returns:
        dict {token_id: probability}
    """
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)

    total = sum(counts.values()) + smoothing * vocab_size
    return {tok: (cnt + smoothing) / total for tok, cnt in counts.items()}


def kl_divergence(p: dict, q: dict, vocab_size: int, smoothing: float = 1e-8) -> float:
    """
    KL divergence KL(P || Q).

    P = live distribution, Q = reference distribution.
    High KL means live inputs are very different from reference.
    """
    all_tokens = set(p.keys()) | set(q.keys())
    default_q  = smoothing / (sum(q.values()) + smoothing * vocab_size)
    default_p  = smoothing / (sum(p.values()) + smoothing * vocab_size)

    kl = 0.0
    for tok in all_tokens:
        p_val = p.get(tok, default_p)
        q_val = q.get(tok, default_q)
        if p_val > 0 and q_val > 0:
            kl += p_val * math.log(p_val / q_val)

    return max(kl, 0.0)


class DriftMonitor:
    """
    Sliding-window drift monitor for LegalMind API.

    Usage:
        monitor = DriftMonitor(window=500, threshold=0.1, vocab_size=8000)

        # On each request:
        monitor.add(token_ids)

        # Check for drift:
        alert = monitor.check()
        if alert["drift_detected"]:
            print(f"Drift! KL={alert['kl_divergence']:.4f}")
    """

    def __init__(
        self,
        window:     int   = 500,
        threshold:  float = 0.1,
        vocab_size: int   = 8000,
        warmup:     int   = 50,     # samples before drift checking begins
    ):
        self.window     = window
        self.threshold  = threshold
        self.vocab_size = vocab_size
        self.warmup     = warmup

        # Rolling buffer of token ID lists
        self._buffer: deque = deque(maxlen=window)
        # Reference distribution (set after warmup)
        self._reference: Optional[dict] = None
        self._n_seen = 0

        # Drift history
        self.kl_history: List[float] = []
        self.alert_count = 0

    def add(self, token_ids: List[int]) -> None:
        """Add a new sequence to the monitor."""
        self._buffer.append(token_ids)
        self._n_seen += 1

        # Capture reference distribution after warmup
        if self._n_seen == self.warmup:
            self._reference = _token_dist(list(self._buffer), self.vocab_size)
            logger.info(f"[Drift] Reference distribution captured "
                        f"({self.warmup} samples, {len(self._reference)} unique tokens)")

    def check(self) -> Dict:
        """
        Compute current KL divergence against reference.

        Returns dict with:
            drift_detected  : bool
            kl_divergence   : float
            samples_seen    : int
            alert_count     : int
            status          : str description
        """
        if self._reference is None:
            return {
                "drift_detected": False,
                "kl_divergence":  0.0,
                "samples_seen":   self._n_seen,
                "alert_count":    self.alert_count,
                "status":         f"warming up ({self._n_seen}/{self.warmup})",
            }

        if len(self._buffer) < 10:
            return {
                "drift_detected": False,
                "kl_divergence":  0.0,
                "samples_seen":   self._n_seen,
                "alert_count":    self.alert_count,
                "status":         "insufficient data",
            }

        live_dist = _token_dist(list(self._buffer), self.vocab_size)
        kl        = kl_divergence(live_dist, self._reference, self.vocab_size)
        self.kl_history.append(kl)

        detected = kl > self.threshold
        if detected:
            self.alert_count += 1
            logger.warning(f"[Drift] ALERT #{self.alert_count}: "
                           f"KL={kl:.4f} > threshold={self.threshold}")

        return {
            "drift_detected": detected,
            "kl_divergence":  round(kl, 6),
            "samples_seen":   self._n_seen,
            "alert_count":    self.alert_count,
            "status":         "drift detected" if detected else "ok",
        }

    def reset_reference(self) -> None:
        """Re-capture reference from current buffer (call after model update)."""
        if len(self._buffer) >= self.warmup:
            self._reference = _token_dist(list(self._buffer), self.vocab_size)
            self.kl_history.clear()
            self.alert_count = 0
            logger.info("[Drift] Reference distribution reset")

    @property
    def stats(self) -> Dict:
        """Summary statistics over all KL history."""
        if not self.kl_history:
            return {"mean_kl": 0.0, "max_kl": 0.0, "n_alerts": 0}
        return {
            "mean_kl":  round(sum(self.kl_history) / len(self.kl_history), 6),
            "max_kl":   round(max(self.kl_history), 6),
            "n_alerts": self.alert_count,
        }
