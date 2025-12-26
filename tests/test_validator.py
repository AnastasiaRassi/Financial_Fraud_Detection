"""
Unit tests for Validator
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validate import Validator


def test_validator_schema_check():
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    config = {'data': {'target_column': 'Class'}}
    
    with pytest.raises(ValueError, match="Target column"):
        validator = Validator(df, config)
        validator.validate()


def test_validator_null_check():
    df = pd.DataFrame({
        'feature1': [1, 2, None],
        'feature2': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    config = {'data': {'target_column': 'Class'}}
    
    with pytest.raises(ValueError, match="null values"):
        validator = Validator(df, config)
        validator.validate()


def test_validator_valid_data():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    config = {'data': {'target_column': 'Class'}}
    
    validator = Validator(df, config)
    result = validator.validate()
    
    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)


def test_validator_target_values():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'Class': [0, 1, 2]  })
    config = {'data': {'target_column': 'Class'}}
    
    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validator = Validator(df, config)
        validator.validate()

