# Helper functions for data loading, evaluation, and reproducibility.

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import random
import os
from pathlib import Path
import sys
from src.ingest import DataIngestion
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_seeds(config: Dict) -> None:
    """
    Set all random seeds for reproducibility.
        
    Args:
        config: Configuration dictionary with reproducibility settings
    """
    reproducibility_config = config.get('reproducibility', {})
    
    seed = reproducibility_config.get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(reproducibility_config.get('numpy_seed', seed))
    os.environ['PYTHONHASHSEED'] = str(reproducibility_config.get('python_seed', seed))     # To ensure no changes in the seed

def load_data(config: Dict) -> pd.DataFrame:
    """
    Load data from specified path.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loaded DataFrame
    """
    
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
    # set configuration
    data_config = config.get('data', {})
    target_column = data_config.get('target_column', 'Class')
    test_split = data_config.get('train_test_split', 0.8)
    val_split = data_config.get('validation_split', 0.2)
    shuffle = data_config.get('shuffle', True)
    random_seed = config.get('reproducibility', {}).get('random_seed', 42)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[[target_column]]
    
    # Initial train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size= test_split,
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