import pandas as pd , sys
import os                  
sys.path.insert(0,r'C:\Users\User\Documents\FRAUD_DETECTION')
from utils.general_utils import CustomException

class DataIngestion:
    def __init__(self, input_path):
        self.input_path = input_path
        assert self.input_path!=''

    def load_and_store(self):
        try:
            df = pd.read_csv(self.input_path)
            assert df.shape[0]>=10000, "Dataset not large enough."
            return df
        except Exception as e:
            raise CustomException(e,sys)

# quick test
if __name__ == '__main__':
    inst = DataIngestion('data/raw/raw_fin_data.csv')
    df = inst.load_and_store()
    print(df.head(5))