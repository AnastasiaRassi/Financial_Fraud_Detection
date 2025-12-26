import sys, logging
import os
import pytest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

@pytest.fixture(autouse=True) 
def patch_logger(monkeypatch): 
    """ Replace the real setup_logger function with a no-op logger
      so tests don't create files or write to the real logging handlers. 
      This runs automatically for every test (autouse=True). """ 
    logger = logging.getLogger("test_logger") 
    logger.handlers = [] 
    logger.addHandler(logging.NullHandler()) 
    monkeypatch.setattr(gu, "setup_logger", lambda *a, **k: logger) 
    return logger    
