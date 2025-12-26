"""
Unit tests for Transformer
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transform import Transformer


def test_transformer_fit_transform():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    config = {'data': {'target_column': 'Class'}}
    
    transformer = Transformer(config)
    X_transformed = transformer.fit_transform(X)
    
    assert X_transformed.shape == X.shape
    assert list(X_transformed.columns) == list(X.columns)
    # Check that values are scaled (mean should be close to 0, std close to 1)
    assert abs(X_transformed.mean().mean()) < 0.1
    assert abs(X_transformed.std().mean() - 1.0) < 0.1


def test_transformer_save_load():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    config = {'data': {'target_column': 'Class'}}
    
    transformer = Transformer(config)
    transformer.fit(X)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
        temp_path = f.name
    
    try:
        transformer.save(temp_path)
        loaded_transformer = Transformer.load(temp_path, config)
        X_new = pd.DataFrame({
            'feature1': [2, 3],
            'feature2': [5, 6]
        })
        result = loaded_transformer.transform(X_new)
        
        assert result.shape == X_new.shape
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

