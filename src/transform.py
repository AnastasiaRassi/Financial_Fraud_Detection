
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Optional, Tuple
import joblib, os , sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import CustomException


class Transformer:
    def __init__(self, config, random_seed: int = 42):
        self.config = config or {}
        self.random_seed = random_seed
        self.scaler = None
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            np.random.seed(self.random_seed)
            # I will decide scaler type depending on feature skewness
            cols = self.config["data"]["features"]
            skewed_cols = X[cols].apply(lambda col: col.skew())

            #  columns with |skew| > 0.5 have  skewed data 
            skewed_cols = skewed_cols[skewed_cols.abs() > 0.5].index.tolist()
            if skewed_cols:
                # robust scaler is 'robust' to outliers, it uses the median not the mean
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            if not self.is_fitted: # we do not fit more than once, unless we are retraining due to data drift
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
        if not self.is_fitted:
            raise ValueError("No scaler fitted to save")
        joblib.dump(self.scaler, path)

    def load(self, path: str):
        self.scaler = joblib.load(path)
        self.is_fitted = True
