"""
Handles StandardScaler fitting and transformation.
Reproducible state saving/loading per project rules.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Optional, Tuple
import joblib
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import CustomException


class Transformer:
    """
    Handles scaling with automatic choice of StandardScaler vs RobustScaler
    based on feature skewness. Also stores fitted scalers for reproducibility.
    """
    def __init__(self, config: Optional[Dict] = None, random_seed: int = 42):
        self.config = config or {}
        self.random_seed = random_seed
        self.scaler = None
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            np.random.seed(self.random_seed)
            # Decide scaler type depending on feature skewness
            cols = self.config["data"]["features"]
            # Compute skewness for each numeric column
            skewed_cols = X[cols].apply(lambda col: col.skew())
            # Filter columns with absolute skew > 0.5
            skewed_cols = skewed_cols[skewed_cols.abs() > 0.5].index.tolist()
            if skewed_cols:
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            if not self.is_fitted:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                self.is_fitted = True
                
            return X_scaled
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled

    def save(self, path: str):
        """Saves the fitted scaler to disk."""
        if not self.is_fitted:
            raise ValueError("No scaler fitted to save")
        joblib.dump(self.scaler, path)

    def load(self, path: str):
        """Load a scaler from disk."""
        self.scaler = joblib.load(path)
        self.is_fitted = True
