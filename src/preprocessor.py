"""
Orchestrates validation and transformation.
The dataset was already subject to a PCA transformation, maintaining the features' predictive value but making 
the feature labels ambigious, so creative feature engineering is off the table.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator import Validator
from src.transformer import Transformer
from general_utils.general_utils import CustomException


class Preprocessor:
    """
    Orchestrates data preprocessing pipeline.
    Handles missing numeric values and chooses scaling based on skewness.
    """
    def __init__(self, config: Optional[Dict] = None, random_seed: int = 42):
        self.config = config or {}
        self.random_seed = random_seed
        self.validator = Validator(pd.DataFrame(), config)
        self.transformer = Transformer(config, random_seed=random_seed)
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        fit_scaler
    ) -> Tuple[pd.DataFrame, Transformer]:
        """
        Preprocess data: validate -> impute -> scale.
        """
        try:
            validator = Validator(df,  self.config)
            validated_df = validator.validate()
            
            target_column = self.config.get('data', {}).get('target_column', 'Class')
            X = validated_df.drop(columns=[target_column])
            y = validated_df[[target_column]]
            
            numeric_cols = X.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    median = X[col].median()
                    X[col] = X[col].fillna(median)

            fit_scaler =  self.config.get('data', {}).get('fit_scaler', True)

            if fit_scaler:
                X_scaled = self.transformer.fit_transform(X)
            else:
                X_scaled = self.transformer.transform(X)
            
            # Combine features with target
            preprocessed_df = pd.concat([X_scaled, y], axis=1)
            
            # Verify no dimensions were changed
            assert preprocessed_df.shape[0] == df.shape[0], "Row count changed during preprocessing"
            assert preprocessed_df.shape[1] == df.shape[1], "Column count changed during preprocessing"
            
            return preprocessed_df, self.transformer
        
        except Exception as e:
            raise CustomException(e, sys)


def preprocess_data(dataset: pd.DataFrame, config: Optional[Dict] = None, random_seed: int = 42) -> Tuple[pd.DataFrame, Transformer]:
    """
    Pure function to preprocess data.
    """
    preprocessor = Preprocessor(config, random_seed=random_seed)
    return preprocessor.preprocess(dataset, fit_scaler=True)