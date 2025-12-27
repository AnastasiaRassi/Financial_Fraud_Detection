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
    reproducibility_config = config.get('reproducibility', {})
    seed = reproducibility_config.get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(reproducibility_config.get('numpy_seed', seed))
    os.environ['PYTHONHASHSEED'] = str(reproducibility_config.get('python_seed', seed))     # To ensure no changes in the seed

def load_data(config: Dict) -> pd.DataFrame:
    ingestion = DataIngestion(config)
    return ingestion.load_data()


def split_data(df: pd.DataFrame, config: Dict):
    data_config = config.get('data', {})
    target_column = data_config.get('target_column', 'Class')
    test_split = data_config.get('train_test_split', 0.8)
    val_split = data_config.get('validation_split', 0.2)
    shuffle = data_config.get('shuffle', True)
    random_seed = config.get('reproducibility', {}).get('random_seed', 42)
    
    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size= test_split,
        shuffle=shuffle,
        random_state=random_seed,
        stratify=y 
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        shuffle=shuffle,
        random_state=random_seed,
        stratify=y_train
    )
    
    # Join features  & targets back together to return the full dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df