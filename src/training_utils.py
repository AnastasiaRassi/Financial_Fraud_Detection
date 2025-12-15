# Helper functions for data loading, evaluation, and reproducibility.

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import random
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_seeds(config: Dict) -> None:
    """
    Set all random seeds for reproducibility.
        
    Args:
        config: Configuration dictionary with reproducibility settings
    """
    repro_config = config.get('reproducibility', {})
    
    seed = repro_config.get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(repro_config.get('torch_seed', seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(repro_config.get('torch_seed', seed))
        if repro_config.get('cudnn_deterministic', True):
            torch.backends.cudnn.deterministic = True
        if repro_config.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = False
    
    # NumPy seed
    np.random.seed(repro_config.get('numpy_seed', seed))
    
    os.environ['PYTHONHASHSEED'] = str(repro_config.get('python_seed', seed))


def load_data(config: Dict) -> pd.DataFrame:
    """
    Load data from specified path.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loaded DataFrame
    """
    from src.data_ingestion.ingest import DataIngestion
    
    ingestion = DataIngestion(config)
    return ingestion.load_data()


def split_data(
    df: pd.DataFrame, 
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Full dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    """
    from sklearn.model_selection import train_test_split
    
    data_config = config.get('data', {})
    target_column = data_config.get('target_column', 'Class')
    train_split = data_config.get('train_test_split', 0.8)
    val_split = data_config.get('validation_split', 0.2)
    shuffle = data_config.get('shuffle', True)
    random_seed = config.get('reproducibility', {}).get('random_seed', 42)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[[target_column]]
    
    # Initial train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=1 - train_split,
        shuffle=shuffle,
        random_state=random_seed,
        stratify=y  # Maintain class distribution
    )
    
    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        shuffle=shuffle,
        random_state=random_seed,
        stratify=y_train
    )
    
    # Recombine features and targets
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders from DataFrames.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import Dataset, DataLoader
    
    class FraudDataset(Dataset):
        """PyTorch Dataset for fraud detection."""
        
        def __init__(self, df: pd.DataFrame, target_column: str = 'Class'):
            self.target_column = target_column
            X = df.drop(columns=[target_column]).values
            y = df[[target_column]].values
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    target_column = config.get('data', {}).get('target_column', 'Class')
    batch_size = config.get('training', {}).get('batch_size', 256)
    
    train_dataset = FraudDataset(train_df, target_column)
    val_dataset = FraudDataset(val_df, target_column)
    test_dataset = FraudDataset(test_df, target_column)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

