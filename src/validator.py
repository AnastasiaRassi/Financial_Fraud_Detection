# Performs schema checks, null checks, shape consistency validation.
# In case a dataset is totally invalid, we cannot move to the pocessing stage.
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import CustomException


class Validator:
    """
    validates the input data at the start of the pipeline.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize Validator.
        
        Args:
            df: DataFrame to validate
            config: Configuration dictionary. If None, uses minimal validation
        """
        if not isinstance(df, pd.DataFrame):  
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy()
        self.config = config
        self.target_column =  self.config.get('data', {}).get('target_column', 'Class') if config else 'Class'
        self.features =  self.config.get('data', {}).get('features', None) if config else None
    
    def validate(self) -> pd.DataFrame:
        """
        Perform all validation checks.
        
        Returns:
            Validated DataFrame
            
        Raises:
            CustomException: If validation fails
        """
        try:
            self._validate_schema()
            self._validate_nulls()
            self._validate_shape()
            self._validate_target()
            return self.df, self.results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _validate_schema(self) -> None:
        """Validate that required columns exist."""
        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in data. "
                f"Available columns: {list[Any](self.df.columns)}"
            )
        if self.features not in self.df.columns:
            raise ValueError(
                f"Features incomplete "
                f"Available columns: {list[Any](self.df.columns)}"
                f"Needed columns: {list[Any](self.features)}"
            )
    
    def _validate_nulls(self) -> None:
        """
        Check for null values of an excessive ratio.
        """
        null_counts = self.df.isnull().sum()
        nulls_ratio = (null_counts.sum())/len(self.df)
        
        if nulls_ratio > 0.5:
            raise ValueError(f"With {null_counts.sum()} nulls, {nulls_ratio} of the data is null, invalidating it.")

    
    def _validate_shape(self) -> None:
        """Validate data shape consistency."""
        if self.df.shape[0] == 0:
            raise ValueError("DataFrame is empty")
        
    def _validate_target(self) -> None:
        """Validate target column values."""
        target = self.df[self.target_column]
        
        # Check that target is binary (0/1)
        unique_values = sorted(target.unique())
        if not (set(unique_values) <= {0, 1}):
            raise ValueError(
                f"Target column must contain only 0 and 1. Found values: {unique_values}"
            )
        # Check that both classes are present
        class_counts = target.value_counts()
        if len(class_counts) < 2:
            raise ValueError(
                f"Target column must contain both classes (0 and 1). "
                f"Found only: {class_counts.to_dict()}"
            )


def validate_data(dataset: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Pure function to validate data.
    
    Args:
        dataset: DataFrame to validate
        config: Configuration dictionary
        
    Returns:
        Validated DataFrame
    """
    validator = Validator(dataset, config)
    return validator.validate()

