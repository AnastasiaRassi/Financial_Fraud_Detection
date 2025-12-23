# Fraud Detection ML Pipeline

**Project Overview**: Implementation of a machine learning pipeline for fraud detection using the Credit Card Fraud Detection dataset.  
The pipeline is inspired by [arxiv:2208.11900](https://arxiv.org/pdf/2208.11900) in the sense that I examined the models they used to confirm that the dataset is predictable. *This project focuses on building a reproducible, fully end-to-end ML pipeline without sampling or altering the dataset.*

## Project Goals

- Validate and preprocess the full dataset
- Handle missing values robustly
- Apply scaling with consideration for skewed features
- Build and train multiple ML models for fraud detection
- Track experiments with MLflow for full reproducibility

## Dataset Assumptions

- **PCA-transformed features**: Original features have been transformed using Principal Component Analysis (V1-V28). Feature names are not semantically meaningful.
- **No feature engineering**: For security reasons, features are ambigious, thus only scaling is performed. No additional feature creation or selection.
- **Null values**: Handled defensively (median imputation) though none are expected.
- **Scaling**: StandardScaler or RobustScaler is applied depending on feature skew.

## Implementation Details

### Preprocessing Pipeline

1. **Validation**: Schema checks, null checks, shape consistency, target value validation.
2. **Imputation**: Median for numeric nulls.
3. **Scaling**: StandardScaler or RobustScaler depending on skew.
4. **No feature engineering**: Dimensions preserved throughout.

### Model Pipeline

- Multiple classical ML models are used to confirm predictability.
- Model configurations are informed by the original paper but **not copied word-for-word**.
- Hyperparameters are configurable via `config.yaml`.

### Project Structure
```text
FRAUD_DETECTION/
├── config.yaml           # Configuration file
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── README.md             
├── data/
│   ├── raw/              # Raw data (provided externally)
│   ├── interim/          # Intermediate validated data
│   └── processed/        # Final processed data
├── src/
│   ├── ingest.py         # Training loop + MLflow logging
│   ├── preprocessor.py   # Orchestrates validation + transformation
│   ├── validator.py      # Data validation class
│   ├── transformer.py    # Scaling (Standard/Robust)
│   ├── model.py          # ML models
│   ├── evaluate.py       # Evaluation metrics
│   ├── training_utils.py # Training utility functions
│   └── train.py          # Training pipeline + MLflow logging
├── artifacts/            # Saved models, transformers (Such as scalers...)
├── logs/                 # Log files
└── tests/                # Unit tests
```

## Reproducibility Instructions

- **Python 3.8+**
- Install dependencies: `pip install -r requirements.txt`
- Run training: `python main.py --mode train`
- Track experiments: `mlflow ui` → http://localhost:5000

All randomness is controlled via seeds in `config.yaml`, and fitted scalers/models are saved for reproducibility.

## Key Notes

- This project **does not claim full replication** of the original paper.  
- The paper was consulted to estimate which models performed best on this dataset WITHOUT sampling.  
- No sampling was applied — the **entire dataset** is used for training and evaluation.  

## Code Quality

- Type hints and docstrings throughout
- Comprehensive error handling
- Unit tests included
- Logging and MLflow tracking for all experiments









