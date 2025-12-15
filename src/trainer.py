"""
Training Module
Paper: https://arxiv.org/pdf/2208.11900

Handles model training loop, loss computation, optimizer and scheduler logic,
and MLflow logging per project rules.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.losses import compute_loss
from src.model import FraudDetectionModel


class Trainer:
    """
    Handles model training.
    
    Per project rules:
    - Use optimizer, learning rate schedule, batch size, and epochs exactly as in paper
    - Replicate loss functions mathematically
    - Log intermediate values required to verify correctness
    - Track all experiments using MLflow
    """
    
    def __init__(self, model: FraudDetectionModel, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize Trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: PyTorch device (CPU or CUDA)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        # TODO: Verify optimizer type and parameters from paper
        training_config = config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        learning_rate = training_config.get('learning_rate', 0.001)
        optimizer_params = training_config.get('optimizer_params', {})
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                **optimizer_params
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                **optimizer_params
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup scheduler (if specified)
        # TODO: Verify if paper uses learning rate scheduler
        scheduler_config = training_config.get('scheduler')
        self.scheduler = None
        if scheduler_config:
            scheduler_type = scheduler_config.get('type', 'step')
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 10)
                )
        
        self.num_epochs = training_config.get('num_epochs', 100)
        self.batch_size = training_config.get('batch_size', 256)
        
        # MLflow setup
        mlflow_config = config.get('mlflow', {})
        if mlflow_config.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'fraud_detection'))
    
    def train_one_epoch(
        self, 
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss = compute_loss(predictions, y, self.config)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'epoch': epoch
        }
    
    def validate(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                predictions = self.model(X)
                loss = compute_loss(predictions, y, self.config)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            self._log_config_to_mlflow()
            
            for epoch in range(1, self.num_epochs + 1):
                # Train
                train_metrics = self.train_one_epoch(train_loader, epoch)
                history['train_loss'].append(train_metrics['train_loss'])
                
                # Validate
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['val_loss'])
                    
                    # Update scheduler if ReduceLROnPlateau
                    if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    val_metrics = {}
                    if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                
                # Log to MLflow
                mlflow.log_metrics({
                    'train_loss': train_metrics['train_loss'],
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)
                
                # Print progress
                if epoch % 10 == 0 or epoch == 1:
                    print(f"Epoch {epoch}/{self.num_epochs} - "
                          f"Train Loss: {train_metrics['train_loss']:.4f} - "
                          f"Val Loss: {val_metrics.get('val_loss', 'N/A')}")
            
            # Log model artifact
            if self.config.get('mlflow', {}).get('log_models', True):
                mlflow.pytorch.log_model(self.model, "model")
        
        return history
    
    def _log_config_to_mlflow(self):
        """Log configuration to MLflow."""
        # Log hyperparameters
        training_params = self.config.get('training', {})
        mlflow.log_params({
            'batch_size': training_params.get('batch_size'),
            'learning_rate': training_params.get('learning_rate'),
            'num_epochs': training_params.get('num_epochs'),
            'optimizer': training_params.get('optimizer'),
            'loss_function': training_params.get('loss_function'),
        })
        
        # Log model architecture
        model_params = self.config.get('model', {})
        mlflow.log_params({
            'hidden_dims': str(model_params.get('hidden_dims')),
            'activation': model_params.get('activation'),
            'dropout': model_params.get('dropout', 0),
        })
        
        # Log system info
        import sys
        import platform
        mlflow.log_params({
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
        })


def train_one_epoch(
    model: FraudDetectionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Pure function to train for one epoch.
    
    Per project rules output expectations.
    
    Args:
        model: Model to train
        dataloader: Training DataLoader
        optimizer: Optimizer
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Dictionary of metrics
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(X)
        loss = compute_loss(predictions, y, config)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0
    }

