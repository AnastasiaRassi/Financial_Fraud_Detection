"""
Main script to run the fraud detection ML pipeline.
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from general_utils.general_utils import load_config, setup_logger
from src.train import run_experiment


def main():
    """
    Main function to run the experiment.
    """
    parser = argparse.ArgumentParser(
        description='Fraud Detection ML Pipeline - Paper Replication'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate'],
        help='Mode: train or evaluate (default: train)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("="*60)
    logger.info("Fraud Detection ML Pipeline")
    logger.info("="*60)
    
    # Load configuration
    try:
        config = load_config()
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Run experiment
    if args.mode == 'train':
        logger.info("Starting training experiment...")
        try:
            results = run_experiment(config)
            logger.info("="*60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
            logger.info(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
            logger.info(f"Test Precision: {results['test_metrics']['precision']:.4f}")
            logger.info(f"Test Recall: {results['test_metrics']['recall']:.4f}")
            logger.info(f"Model saved to: {results['model_path']}")
            logger.info(f"Transformer saved to: {results['transformer_path']}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        logger.info("Evaluation mode not yet implemented")
        # TODO: Implement evaluation mode
        sys.exit(1)


if __name__ == '__main__':
    main()

