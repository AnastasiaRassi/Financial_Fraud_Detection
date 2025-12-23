"""
Evaluation Module
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_sklearn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a classical sklearn model.

    Args:
        y_true: Ground truth labels (binary 0/1)
        y_pred: Predicted labels (binary 0/1)
        y_proba: Predicted probabilities for the positive class

    Returns:
        Dictionary of evaluation metrics.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_proba = y_proba.ravel()

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1
        else 0.0,
        "pr_auc": average_precision_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1
        else 0.0,
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Binary classification
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])

    return metrics


def print_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a readable format.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    if 'tp' in metrics:
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        print(f"  True Positives:  {metrics['tp']}")
    print("="*50 + "\n")

