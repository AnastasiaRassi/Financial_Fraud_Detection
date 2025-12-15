"""
Model Architecture Module
Paper: https://arxiv.org/pdf/2208.11900

IMPORTANT: This implementation must match the paper's architecture exactly.
TODO: Verify all architecture details from the paper:
- Layer sizes and order
- Activation functions
- Normalization layers
- Dropout rates
- Initialization schemes
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FraudDetectionModel(nn.Module):
    """
    Neural network model for fraud detection.
    
    NOTE: Architecture parameters must match paper exactly.
    Per project rules:
    - Match layer order, dimensions, activations, and normalization
    - No refactoring or optimization unless explicitly requested
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model architecture.
        
        Args:
            config: Configuration dictionary with model parameters
            
        NOTE: All parameters should match the paper exactly.
        Current defaults are placeholders that need verification.
        """
        super(FraudDetectionModel, self).__init__()
        
        model_config = config.get('model', {})
        self.input_dim = model_config.get('input_dim')
        hidden_dims = model_config.get('hidden_dims', [64, 32])
        self.output_dim = model_config.get('output_dim', 1)
        activation_name = model_config.get('activation', 'relu')
        dropout_rate = model_config.get('dropout', 0.0)
        use_batch_norm = model_config.get('use_batch_norm', False)
        
        # Validate input_dim is set
        if self.input_dim is None:
            raise ValueError("input_dim must be specified in config")
        
        # Select activation function
        # TODO: Verify exact activation function from paper
        if activation_name.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_name.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation_name.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")
        
        # Build layers
        # TODO: Verify exact layer structure, order, and dimensions from paper
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (if specified in paper)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (if specified in paper)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        # TODO: Verify if paper uses sigmoid or linear output
        layers.append(nn.Linear(prev_dim, self.output_dim))
        # Typically binary classification uses sigmoid, but verify with paper
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        # TODO: Verify initialization scheme from paper
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights.
        
        NOTE: Initialization scheme should match paper if specified.
        Current implementation uses PyTorch defaults.
        TODO: Verify if paper specifies particular initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # PyTorch default: uniform initialization
                # TODO: Check if paper specifies different initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor (already includes sigmoid in forward)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def build_model(config: Dict, input_dim: Optional[int] = None) -> FraudDetectionModel:
    """
    Build model from configuration.
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (will override config if provided)
        
    Returns:
        Initialized model
    """
    if input_dim is not None:
        config = config.copy()
        config['model'] = config.get('model', {}).copy()
        config['model']['input_dim'] = input_dim
    
    return FraudDetectionModel(config)

