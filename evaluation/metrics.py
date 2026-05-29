"""
Evaluation metrics and reporting for LegalMind.

Computes: Accuracy, Precision, Recall, F1 (per-class + macro),
          Confusion matrix, ROC-AUC, and a full classification report.
"""

import json
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["compliant", "violation"]


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict:
    """
    Compute full classification metrics.

    Args:
        y_true : Ground-truth labels (0 or 1)
        y_pred : Predicted labels (0 or 1)
        y_prob : Predicted probabilities for class 1 (for AUC)

    Returns:
        Dictionary with all metrics.
    """
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "f1_macro":           f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_violation":       f1_score(y_true, y_pred, average="binary",   zero_division=0),
        "precision_violation":precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall_violation":   recall_score(y_true, y_pred, average="binary",    zero_division=0),
        "confusion_matrix":   confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def print_report(metrics: Dict, y_true: List[int], y_pred: List[int]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("  LegalMind — Evaluation Report")
    print("=" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  F1 (violation)    : {metrics['f1_violation']:.4f}  ← primary metric")
    print(f"  F1 (macro)        : {metrics['f1_macro']:.4f}")
    print(f"  Precision (viol.) : {metrics['precision_violation']:.4f}")
    print(f"  Recall (viol.)    : {metrics['recall_violation']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"]:
        print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    cm = np.array(metrics["confusion_matrix"])
    print("  Confusion Matrix:")
    print(f"                Pred:Compliant  Pred:Violation")
    print(f"  True:Compliant    {cm[0,0]:>6}          {cm[0,1]:>6}")
    print(f"  True:Violation    {cm[1,0]:>6}          {cm[1,1]:>6}")
    print("=" * 60)


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    tokenizer,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Run full evaluation on a DataLoader.
    Optionally saves results to a JSON file.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"]

        _, logits = model(input_ids, attention_mask=attn_mask, mode="finetune")
        probs     = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds     = logits.argmax(dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print_report(metrics, all_labels, all_preds)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[Eval] Metrics saved → {save_path}")

    return metrics
