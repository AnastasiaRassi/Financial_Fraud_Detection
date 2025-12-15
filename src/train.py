"""
Training Orchestration Script
Paper: https://arxiv.org/pdf/2208.11900

Orchestrates the complete training pipeline:
1. Load and validate data
2. Preprocess data
3. Build model
4. Train model
5. Evaluate model
6. Save artifacts
"""
import pandas as pd
import torch
import joblib
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.general_utils import load_config, setup_logger, CustomException
from src.utils import set_seeds, load_data, split_data, create_data_loaders
from src.preprocessor import Preprocessor
from src.model import build_model
from src.trainer import Trainer
from src.evaluate import evaluate, print_evaluation_metrics


def run_experiment(config: Dict) -> Dict:
    """
    Run complete training experiment.
    
    Per project rules output expectations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with experiment results
    """
    logger = setup_logger()
    logger.info("Starting experiment")
    
    try:
        # Set seeds for reproducibility
        set_seeds(config)
        logger.info("Random seeds set for reproducibility")
        
        # Step 1: Load data
        logger.info("Loading data...")
        df = load_data(config)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Step 2: Validate and preprocess data
        logger.info("Preprocessing data...")
        preprocessor = Preprocessor(config)
        preprocessed_df, transformer = preprocessor.preprocess(df, fit_scaler=True)
        logger.info("Data preprocessing completed")
        
        # Save transformer
        artifacts_dir = config['paths']['artifacts_dir']
        os.makedirs(artifacts_dir, exist_ok=True)
        transformer_path = os.path.join(artifacts_dir, 'transformer.joblib')
        transformer.save(transformer_path)
        logger.info(f"Transformer saved to {transformer_path}")
        
        # Step 3: Split data
        logger.info("Splitting data...")
        train_df, val_df, test_df = split_data(preprocessed_df, config)
        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # Step 4: Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, config
        )
        
        # Step 5: Build model
        logger.info("Building model...")
        input_dim = train_df.shape[1] - 1  # Exclude target column
        config['model']['input_dim'] = input_dim
        model = build_model(config, input_dim)
        logger.info(f"Model built with input_dim={input_dim}")
        
        # Step 6: Train model
        logger.info("Training model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        trainer = Trainer(model, config, device)
        history = trainer.train(train_loader, val_loader)
        logger.info("Training completed")
        
        # Step 7: Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, config, device)
        print_evaluation_metrics(test_metrics)
        
        # Step 8: Save model
        model_path = os.path.join(artifacts_dir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Prepare results
        results = {
            'test_metrics': test_metrics,
            'train_history': history,
            'model_path': model_path,
            'transformer_path': transformer_path,
            'config': config
        }
        
        logger.info("Experiment completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == '__main__':
    config = load_config()
    results = run_experiment(config)
    print("\nExperiment Results:")
    print(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")

