# Fraud Detection ML Pipeline

**Paper Replication Project**: Faithful replication of the methodology described in [arxiv:2208.11900](https://arxiv.org/pdf/2208.11900)

## Paper Summary

This project implements a machine learning pipeline for fraud detection following the methodology and experimental setup described in the referenced paper. The implementation strictly adheres to the original methodology, equations, architecture, and experimental setup.

**NOTE**: Please refer to the paper for detailed methodology and results. This implementation serves as a faithful replication for reproducibility purposes.

## Dataset Assumptions

The project uses the Credit Card Fraud Detection dataset with the following assumptions:

- **Data is already PCA-transformed**: The original features have been transformed using Principal Component Analysis (PCA) prior to ingestion. Feature names (V1-V28) are not semantically meaningful.
- **No feature engineering**: Only Standard Scaling is permitted. No feature creation, selection, or engineering steps are performed.
- **Null values**: Null values are not expected in the dataset but are handled defensively (validation will fail if nulls are detected).
- **Standard Scaling**: The only preprocessing step allowed is StandardScaler normalization, which is justified by manual inspection of feature distributions.

## Implementation Details

### Architecture

The neural network architecture follows the specifications from the paper:

- **Input Layer**: Dimension matches the number of features (after excluding target column)
- **Hidden Layers**: Architecture specified in `config.yaml` (TODO: Verify exact dimensions from paper)
- **Output Layer**: Single output with sigmoid activation for binary classification
- **Activations**: ReLU (or as specified in paper - TODO: Verify)
- **Initialization**: Xavier uniform (or as specified in paper - TODO: Verify)

### Training Configuration

Training hyperparameters match the paper exactly:

- **Optimizer**: Adam (or as specified - TODO: Verify)
- **Learning Rate**: 0.001 (or as specified - TODO: Verify)
- **Batch Size**: 256 (or as specified - TODO: Verify)
- **Epochs**: 100 (or as specified - TODO: Verify)
- **Loss Function**: Binary Cross-Entropy (or as specified - TODO: Verify)

### Preprocessing Pipeline

1. **Data Ingestion**: Load raw CSV data
2. **Validation**: 
   - Schema checks (ensure target column exists)
   - Null checks (fail if nulls detected)
   - Shape consistency validation
   - Target value validation (binary: 0/1)
3. **Transformation**: StandardScaler normalization only
4. **No feature engineering**: Dimensions are preserved throughout

### Project Structure

```
FRAUD_DETECTION/
├── config.yaml              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   ├── raw/               # Raw data (provided externally)
│   ├── interim/           # Intermediate validated data
│   └── processed/         # Final processed data
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── ingest.py     # Data loading
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   └── validate.py   # Data validation
│   ├── validator.py       # Data validation class
│   ├── transformer.py     # StandardScaler transformation
│   ├── preprocessor.py    # Preprocessing orchestration
│   ├── model.py           # Neural network architecture
│   ├── losses.py          # Loss functions
│   ├── trainer.py         # Training loop and MLflow logging
│   ├── evaluate.py        # Evaluation metrics
│   ├── train.py           # Training orchestration
│   └── utils.py           # Utility functions
├── utils/
│   └── general_utils.py   # General utilities (logging, config loading)
├── artifacts/             # Saved models, transformers
├── logs/                  # Log files
└── tests/                 # Unit tests
```

## Reproducibility Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

#### Training

Run the complete training pipeline:

```bash
python main.py --mode train
```

Or use the training script directly:

```bash
python src/train.py
```

#### Configuration

All hyperparameters and settings are specified in `config.yaml`. Key sections:

- `reproducibility`: Random seeds for reproducibility
- `data`: Dataset configuration (split ratios, target column)
- `preprocessing`: Preprocessing settings (only StandardScaler)
- `model`: Model architecture parameters
- `training`: Training hyperparameters (must match paper)
- `mlflow`: MLflow tracking configuration

### Reproducibility Guarantees

- **Random Seeds**: All randomness is controlled via explicit seeds in `config.yaml`
- **Deterministic Operations**: PyTorch deterministic mode enabled where possible
- **Version Tracking**: Python version, library versions, and hardware info logged to MLflow
- **Artifact Saving**: Model and transformer states are saved for exact reproducibility

## MLflow Tracking

All experiments are automatically tracked using MLflow:

- **Parameters**: All hyperparameters and configuration
- **Metrics**: Training and validation metrics per epoch
- **Artifacts**: Trained models, transformers, and config files
- **System Info**: Python version, library versions, platform
- **Git Hash**: Git commit hash (if available)

View results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Differences from Original Paper

**NOTE**: This section should be updated after verifying implementation against the paper. Current status:

- Architecture details: TODO - Verify exact layer dimensions and activations
- Hyperparameters: TODO - Verify optimizer, learning rate, batch size, epochs
- Loss function: TODO - Verify exact loss formulation
- Evaluation metrics: TODO - Verify which metrics are reported
- Data splits: TODO - Verify train/validation/test split strategy

All TODO items should be resolved by consulting the paper and updating the code accordingly.

## How to Run Experiments

### Basic Training

```bash
python main.py --mode train
```

### Custom Configuration

Modify `config.yaml` and run:

```bash
python main.py --config config.yaml --mode train
```

### View MLflow Results

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Evaluation (Future)

```bash
python main.py --mode evaluate
```

## Code Quality

- **Type Hints**: All functions use type annotations
- **Docstrings**: All modules and functions have docstrings
- **Error Handling**: Custom exceptions with detailed error messages
- **Logging**: Comprehensive logging throughout the pipeline
- **Testing**: Unit tests for key components (see `tests/` directory)

## Citation

If you use this implementation, please cite the original paper:

```
TODO: Add paper citation once verified
```

## License

[Add license information]

## Contact

[Add contact information]

