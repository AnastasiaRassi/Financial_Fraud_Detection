from typing import Dict, Any, Tuple
import os, mlflow, sys, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import load_config, setup_logger, CustomException
from src.training_utils import set_seeds, load_data, split_data
from src.preprocess import Preprocessor
from src.model import build_models
from src.evaluate import evaluate_model, print_evaluation_metrics


def _prepare_xy(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    return X, y


def _tune_threshold(y_true, y_proba, metric_name="f1", start=0.1, end=0.9, step=0.01):
    best_threshold = 0.5
    best_score = 0.0
    
    thresholds = np.arange(start, end + step, step)
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

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
        preprocessor = Preprocessor(df, config)
        preprocessed_df, transformer = preprocessor.preprocess(fit_scaler=True)
        logger.info("Data preprocessing completed")

        artifacts_dir = config["paths"]["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)
        transformer_path = os.path.join(artifacts_dir, "transformer.joblib")
        transformer.save(transformer_path)
        logger.info(f"Transformer saved to {transformer_path}")

        logger.info("Splitting data...")
        train_df, val_df, test_df = split_data(preprocessed_df, config)
        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

        target_column = config.get("data", {}).get("target_column", "Class")
        X_train_df = train_df.drop(columns=[target_column])
        y_train_series = train_df[target_column]
        
        sampling_config = config.get("data_pipeline", {}).get("sampling", {})
        if sampling_config.get("enabled", False):
            X_train_df, y_train_series = transformer.apply_smote(X_train_df, y_train_series)
            logger.info(f"Applied SMOTE. New training shape: {X_train_df.shape}")
        
        X_train = X_train_df.to_numpy()
        y_train = y_train_series.to_numpy()
        X_val, y_val = _prepare_xy(val_df, target_column)
        X_test, y_test = _prepare_xy(test_df, target_column)

        logger.info("Building sklearn models from config...")
        models = build_models(config)
        logger.info(f"{len(models)} models constructed from config.")

        best_f1 = 0
        best_result = {}
        best_model = None

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

                model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="models")
                mlflow.sklearn.log_model(model, artifact_path="model",
                                         registered_model_name="FraudDetectionModel")

                if test_metrics["f1"] > best_f1:
                    best_f1 = test_metrics["f1"]
                    best_model = model
                    best_result = {
                        "best_model_name": model_name,
                        "test_metrics": test_metrics,
                        "val_metrics": val_metrics,
                        "model_path": model_path,
                        "transformer_path": transformer_path,
                        "config": config
                    }

                logger.info(f"Finished model: {model_name}")
                logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
                logger.info(f"Test F1: {test_metrics['f1']:.4f}")

        if not best_result:
            raise RuntimeError("No models were trained successfully.")

        post_training_config = config.get("model_selection", {}).get("post_training", {})
        if post_training_config.get("enable_threshold_tuning", False) and best_model is not None:
            logger.info("Tuning threshold for best model...")
            
            if hasattr(best_model, "predict_proba"):
                y_val_proba = best_model.predict_proba(X_val)[:, 1]
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_val_proba = best_model.decision_function(X_val)
                y_test_proba = best_model.decision_function(X_test)
            
            threshold_config = post_training_config.get("threshold_search_space", {})
            start = threshold_config.get("start", 0.1)
            end = threshold_config.get("end", 0.9)
            step = threshold_config.get("step", 0.01)
            
            best_threshold, tuned_f1 = _tune_threshold(y_val, y_val_proba, "f1", start, end, step)
            y_test_pred_tuned = (y_test_proba > best_threshold).astype(int)
            test_metrics_tuned = evaluate_model(y_test, y_test_pred_tuned, y_test_proba)
            
            logger.info(f"Best threshold: {best_threshold:.3f}, Tuned F1: {test_metrics_tuned['f1']:.4f}")
            
            if test_metrics_tuned["f1"] > best_result["test_metrics"]["f1"]:
                best_result["test_metrics"] = test_metrics_tuned
                best_result["optimal_threshold"] = best_threshold
                logger.info("Threshold tuning improved F1 score")
            else:
                logger.info("Default threshold (0.5) performed better")

        logger.info("Experiment completed successfully")
        logger.info(f"Best model: {best_result['best_model_name']} with Test F1={best_result['test_metrics']['f1']:.4f}")

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













