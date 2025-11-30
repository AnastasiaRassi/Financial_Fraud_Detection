import logging, os, yaml
from datetime import datetime
from sklearn.metrics import f1_score, classification_report, roc_auc_score, matthews_corrcoef
import pandas as pd

class CustomException(Exception):
    """
    Custom exception handling class
    """
    def __init__(self, e, sys):
        self.error_message = e
        _,_,exc_tb = sys.exc_info()
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message))


def load_config():
    """
    loads config.yaml irrespective of current file path
    """
    with open(r'C:\Users\User\Documents\FRAUD_DETECTION\config.yaml') as f:
        return yaml.safe_load(f)
    
def setup_logger():
    """
    sets up the logger
    """
    config = load_config()

    log_folder = config["logger"]["destination"]
    log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_file_path = os.path.join(log_folder, log_file_name)
    os.makedirs(log_file_path, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ]  %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)  
    return logger


