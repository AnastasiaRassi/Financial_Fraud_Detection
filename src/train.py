from typing import Dict, Any, Tuple
import os, mlflow, sys, joblib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# we will need to utilize the classes & methods of other files, for this reason I will insert the root path for any time we test 
from general_utils.general_utils import load_config, setup_logger, CustomException
from src.training_utils import set_seeds, load_data, split_data
from src.preprocess import Preprocessor
from src.model import build_models
from src.evaluate import evaluate_model, print_evaluation_metrics
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# method for splitting the target & features apart
def _prepare_xy(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    return X, y

# method to initialize MLFlow
def _init_mlflow(config: Dict, logger) -> None:
    try: 
        mlflow_config = config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
        experiment_name = mlflow_config.get("experiment_name", "financial_fraud_detection")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise CustomException(e, sys)

# method to orchestrate entire process of training & registering models with MLFlow
def run_experiment(config: Dict) -> Dict[str, Any]:
    logger = setup_logger()
    logger.info("Starting ML experiment")

    try:
        set_seeds(config)
        logger.info("Random seeds set for reproducibility")

        _init_mlflow(config, logger)

        logger.info("Loading data...")
        df = load_data(config) 
        logger.info(f"Loaded data shape: {df.shape}, now preprocessing data...")
        preprocessor = Preprocessor(df, config) # begin preprocessing (validating is embedded here)
        preprocessed_df, transformer = preprocessor.preprocess(fit_scaler=True)
        logger.info("Data preprocessing completed")

        artifacts_dir = config["paths"]["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)
        transformer_path = os.path.join(artifacts_dir, "transformer.joblib")
        transformer.save(transformer_path)
        logger.info(f"Transformer saved to {transformer_path}")

        # create test,  train and val splits.
        logger.info("Splitting data...")
        train_df, val_df, test_df = split_data(preprocessed_df, config)
        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

        target_column = config.get("data", {}).get("target_column", "Class")
        X_train, y_train = _prepare_xy(train_df, target_column)
        X_val, y_val = _prepare_xy(val_df, target_column)
        X_test, y_test = _prepare_xy(test_df, target_column)

        logger.info("Building sklearn models from config...")
        models = build_models(config)
        logger.info(f"{len(models)} models constructed from config.")

        best_f1 = 0
        best_result= {}

        # train and evaluate each model with MLflow logging
        for model in models:
            model_name = getattr(model, "_fd_name", model.__class__.__name__)
            logger.info(f"Training model: {model_name}")

            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_class", model.__class__.__name__)

                model.fit(X_train, y_train)

                if hasattr(model, "predict_proba"):
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # for margin based classifiers, we use decision_function
                    y_val_proba = model.decision_function(X_val)
                    y_test_proba = model.decision_function(X_test)

                y_val_pred = (y_val_proba > 0.5).astype(int)
                y_test_pred = (y_test_proba > 0.5).astype(int)

                val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba)
                test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba)

                for metric, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric}", float(value))
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric}", float(value))

                # save model artifact to reuse later on.
                model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="models")
                mlflow.sklearn.log_model( model, artifact_path="model",
                                            registered_model_name="FraudDetectionModel")

                if test_metrics["f1"] > best_f1:
                    best_f1 = test_metrics["f1"]
                    best_result = { "best_model_name": model_name,
                                    "test_metrics": test_metrics,
                                    "val_metrics": val_metrics,
                                    "model_path": model_path,
                                    "transformer_path": transformer_path,
                                    "config": config}

                logger.info(f"Finished model: {model_name}")
                logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
                logger.info(f"Test F1: {test_metrics['f1']:.4f}")

        if not best_result:
            raise RuntimeError("No models were trained successfully.")

        logger.info("Experiment completed successfully")
        logger.info( f"Best model: {best_result['best_model_name']} with Test F1={best_result['test_metrics']['f1']:.4f}")

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










