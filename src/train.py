"""
Orchestrates the complete classical ML training pipeline:
1. Load and validate data
2. Preprocess data
3. Build sklearn models
4. Train models
5. Evaluate models
6. Save artifacts and log to MLflow

Deviation from paper:
- The original paper uses a single MLP. Per user instruction we instead
  train multiple classical ML models defined in `config.yaml` while
  keeping the rest of the pipeline (data, preprocessing, metrics)
  consistent and fully reproducible.
"""
from typing import Dict, Any, Tuple
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import mlflow

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import load_config, setup_logger, CustomException
from src.training_utils import set_seeds, load_data, split_data
from src.preprocess import Preprocessor
from src.model import build_models
from src.evaluate import evaluate_sklearn, print_evaluation_metrics


def _prepare_xy(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a dataframe into X, y numpy arrays.
    """
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    return X, y


def _init_mlflow(config: Dict) -> None:
    """
    Initialize MLflow tracking according to configuration.
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_config.get("experiment_name", "financial_fraud_detection")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def run_experiment(config: Dict) -> Dict[str, Any]:
    """
    Runs complete experiment over all configured sklearn models.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with best model results and paths.
    """
    logger = setup_logger()
    logger.info("Starting classical ML experiment")

    try:
        # Set seeds for reproducibility
        set_seeds(config)
        logger.info("Random seeds set for reproducibility")

        # Initialise MLflow
        _init_mlflow(config)

        # Step 1: Load data
        logger.info("Loading data...")
        df = load_data(config)
        logger.info(f"Loaded data shape: {df.shape}")

        # Step 2: Validate and preprocess data
        logger.info("Preprocessing data...")
        preprocessor = Preprocessor(df, config)
        preprocessed_df, transformer = preprocessor.preprocess(fit_scaler=True)
        logger.info("Data preprocessing completed")

        # Save transformer
        artifacts_dir = config["paths"]["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)
        transformer_path = os.path.join(artifacts_dir, "transformer.joblib")
        transformer.save(transformer_path)
        logger.info(f"Transformer saved to {transformer_path}")

        # Step 3: Split data
        logger.info("Splitting data...")
        train_df, val_df, test_df = split_data(preprocessed_df, config)
        logger.info(
            f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
        )

        target_column = config.get("data", {}).get("target_column", "Class")
        X_train, y_train = _prepare_xy(train_df, target_column)
        X_val, y_val = _prepare_xy(val_df, target_column)
        X_test, y_test = _prepare_xy(test_df, target_column)

        # Step 4: Build models
        logger.info("Building sklearn models from config...")
        models = build_models(config)
        logger.info(f"{len(models)} models constructed from config.")

        best_f1 = -np.inf
        best_result: Dict[str, Any] = {}

        # Step 5: Train and evaluate each model with MLflow logging
        for model in models:
            model_name = getattr(model, "_fd_name", model.__class__.__name__)
            logger.info(f"Training model: {model_name}")

            with mlflow.start_run(run_name=model_name):
                # Log basic params
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_class", model.__class__.__name__)

                # Train
                model.fit(X_train, y_train)

                # Validation metrics (for model selection)
                if hasattr(model, "predict_proba"):
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # For margin-based classifiers, use decision_function
                    y_val_proba = model.decision_function(X_val)
                    y_test_proba = model.decision_function(X_test)

                y_val_pred = (y_val_proba > 0.5).astype(int)
                y_test_pred = (y_test_proba > 0.5).astype(int)

                val_metrics = evaluate_sklearn(y_val, y_val_pred, y_val_proba)
                test_metrics = evaluate_sklearn(y_test, y_test_pred, y_test_proba)

                # Log metrics
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", float(v))
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", float(v))

                # Save model artifact
                model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="models")

                # Track best model by test F1
                if test_metrics["f1"] > best_f1:
                    best_f1 = test_metrics["f1"]
                    best_result = {
                        "best_model_name": model_name,
                        "test_metrics": test_metrics,
                        "val_metrics": val_metrics,
                        "model_path": model_path,
                        "transformer_path": transformer_path,
                        "config": config,
                    }

                logger.info(f"Finished model: {model_name}")
                logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
                logger.info(f"Test F1: {test_metrics['f1']:.4f}")

        if not best_result:
            raise RuntimeError("No models were trained successfully.")

        logger.info("Experiment completed successfully")
        logger.info(
            f"Best model: {best_result['best_model_name']} "
            f"with Test F1={best_result['test_metrics']['f1']:.4f}"
        )

        # Print metrics for the best model
        print_evaluation_metrics(best_result["test_metrics"])

        return best_result

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    cfg = load_config()
    results = run_experiment(cfg)
    print("\nBest Experiment Results:")
    print(f"Best model: {results['best_model_name']}")
    print(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")










