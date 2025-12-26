"""
Loads raw data from specified path. Data is assumed to be already PCA-transformed,
with feature labels unclear for security reasons.
"""
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import CustomException, load_config


class DataIngestion:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        
        # Get input path from config
        input_path = config['paths']['input_raw_data']
        if not os.path.isabs(input_path):
            input_path = project_root / 'FRAUD_DETECTION' /input_path
        self.input_path = str(input_path)
        
        assert self.input_path != '', "Input path cannot be empty"
        assert os.path.exists(self.input_path), f"Input file not found: {self.input_path}"

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            assert df.shape[0] >= 10000, "Dataset not large enough (min 10000 rows required)."
            return df
        except Exception as e:
            raise CustomException(e, sys)

# for quick testing
if __name__ == '__main__':
    # Quick test
    obj = DataIngestion()
    df = obj.load_data()
    print(f"Data shape: {df.shape}")
    print(df.head(5))