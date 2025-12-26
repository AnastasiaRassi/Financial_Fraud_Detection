import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import CustomException


class Validator:
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        if not isinstance(df, pd.DataFrame):  
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy()
        self.config = config
        self.target_column =  self.config.get('data', {}).get('target_column', 'Class') if config else 'Class'
        self.features =  self.config.get('data', {}).get('features', None) if config else None
    
    def validate(self) -> pd.DataFrame:
        try:
            self._validate_schema()
            self._validate_nulls()
            self._validate_shape()
            self._validate_target()
            return self.df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _validate_schema(self):
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data. Available columns: {list(self.df.columns)}")
        missing_features = [col for col in self.features if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Features incomplete. Missing columns: {missing_features}. Available columns: {list(self.df.columns)}")

    def _validate_nulls(self):
        null_counts = self.df.isnull().sum()
        nulls_ratio = (null_counts.sum())/len(self.df)
        
        if nulls_ratio > 0.5:
            raise ValueError(f"With {null_counts.sum()} nulls, {nulls_ratio} of the data is null, invalidating it.")

    
    def _validate_shape(self):
        if self.df.shape[0] == 0:
            raise ValueError("DataFrame is empty")
        
    def _validate_target(self):
        target = self.df[self.target_column]
        
        unique_values = sorted(target.unique())
        if not (set(unique_values) <= {0, 1}):
            raise ValueError(f"Target column must contain only 0 and 1. Found values: {unique_values}")
        
        class_counts = target.value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Target column must contain both classes (0 and 1). Found only: {class_counts.to_dict()}")


def validate_data(dataset: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    validator = Validator(dataset, config)
    return validator.validate()

