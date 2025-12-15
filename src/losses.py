"""
Loss Functions Module
Paper: https://arxiv.org/pdf/2208.11900

IMPORTANT: Loss functions must match the paper exactly.
TODO: Verify exact loss function formulation from the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CrossEntropyLoss(nn.Module):
    """
    Binary cross-entropy loss.
    
    NOTE: This is a standard implementation. Verify if paper uses any
    modifications (e.g., class weights, focal loss, etc.)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize loss function.
        
        Args:
            config: Configuration dictionary with loss parameters
        """
        super(CrossEntropyLoss, self).__init__()
        self.config = config or {}
        loss_params = config.get('training', {}).get('loss_params', {}) if config else {}
        
        # TODO: Check if paper uses class weights for imbalanced data
        pos_weight = loss_params.get('pos_weight', None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.use_logits = True  # If False, expects probabilities instead
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions (logits if use_logits=True, else probabilities)
            targets: True labels (0 or 1)
            
        Returns:
            Loss value
        """
        # Ensure targets are float and have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        
        # If predictions are probabilities, convert to logits
        if not self.use_logits:
            # Clamp to avoid numerical issues
            predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)
            predictions = torch.log(predictions / (1 - predictions))
        
        return self.criterion(predictions, targets)


def compute_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    config: Dict
) -> torch.Tensor:
    """
    Compute loss function.
    
    This is the main loss computation function as specified in project rules.
    
    Args:
        predictions: Model predictions
        targets: True labels
        config: Configuration dictionary
        
    Returns:
        Loss value
        
    NOTE: Verify exact loss function from paper. Current implementation uses
    binary cross-entropy, but paper may specify different formulation.
    """
    loss_name = config.get('training', {}).get('loss_function', 'cross_entropy')
    
    if loss_name == 'cross_entropy' or loss_name == 'bce':
        criterion = CrossEntropyLoss(config)
        return criterion(predictions, targets)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                        "TODO: Implement paper-specific loss if different.")

