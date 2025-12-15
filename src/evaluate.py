"""
Evaluation Module
Paper: https://arxiv.org/pdf/2208.11900

Evaluates model using metrics matching the paper.
Per project rules: Use the same metrics and evaluation protocol as the paper.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import FraudDetectionModel


def evaluate(
    model: FraudDetectionModel,
    dataloader: DataLoader,
    config: Dict,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Per project rules output expectations.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Dictionary of evaluation metrics
        
    NOTE: Metrics should match those reported in the paper.
    TODO: Verify exact metrics and evaluation protocol from paper.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probs = model.predict_proba(X)
            
            # Convert probabilities to predictions (threshold = 0.5)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    probs = np.array(all_probs).flatten()
    
    # Compute metrics
    # TODO: Verify which metrics the paper reports
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, zero_division=0),
        'recall': recall_score(targets, predictions, zero_division=0),
        'f1': f1_score(targets, predictions, zero_division=0),
        'roc_auc': roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0,
        'pr_auc': average_precision_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0,
    }
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    if cm.size == 4:  # Binary classification
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    
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

