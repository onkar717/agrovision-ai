"""
Model Training Pipeline

Complete training workflow:
1. Generate or load synthetic dataset
2. Preprocess data
3. Build and compile CNN model
4. Train with early stopping
5. Evaluate on test set
6. Save model and metrics
"""

import os
import sys
import json
from datetime import datetime
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.synthetic_dataset import SyntheticDatasetGenerator
from data_pipeline.preprocessing import DataPreprocessor
from pest_model.cnn_architecture import PestCNNModel
from utils import load_config, setup_logging


def train_pest_model(
    dataset_path: str = "data/synthetic_pest_dataset.csv",
    generate_new_data: bool = False,
    num_samples: int = 2000,
    config_path: str = "config.yaml"
):
    """
    Complete training pipeline
    
    Args:
        dataset_path: Path to CSV dataset
        generate_new_data: Whether to generate new synthetic data
        num_samples: Number of samples to generate (if generate_new_data=True)
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    model_config = config['model']
    
    logger.info("="*60)
    logger.info("PEST PREDICTION MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Generate or load dataset
    if generate_new_data or not os.path.exists(dataset_path):
        logger.info(f"Generating new synthetic dataset: {num_samples} samples")
        generator = SyntheticDatasetGenerator(config_path=config_path)
        generator.generate_dataset(
            num_samples=num_samples,
            duration_days=7,
            output_path=dataset_path
        )
    else:
        logger.info(f"Using existing dataset: {dataset_path}")
    
    # Step 2: Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(
        sequence_length=model_config['sequence_length'],
        n_features=len(model_config['features'])
    )
    
    splits = preprocessor.preprocess_full_pipeline(
        csv_path=dataset_path,
        save_preprocessor=True
    )
    
    (X_train, X_val, X_test,
     y_train_growth, y_val_growth, y_test_growth,
     y_train_risk, y_val_risk, y_test_risk) = splits
    
    logger.info(f"Data splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Step 3: Build model
    logger.info("Building CNN model...")
    model = PestCNNModel(
        sequence_length=model_config['sequence_length'],
        n_features=len(model_config['features']),
        cnn_filters=model_config['cnn_filters'],
        kernel_sizes=model_config['kernel_sizes'],
        pool_size=model_config['pool_size'],
        dropout_rate=model_config['dropout_rate'],
        dense_units=model_config['dense_units'],
        n_risk_classes=3
    )
    
    model.compile_model(learning_rate=model_config['learning_rate'])
    
    logger.info("\nModel Architecture:")
    print("="*60)
    model.get_summary()
    print("="*60)
    
    # Step 4: Train model
    logger.info("\nTraining model...")
    history = model.train(
        X_train=X_train,
        y_train_growth=y_train_growth,
        y_train_risk=y_train_risk,
        X_val=X_val,
        y_val_growth=y_val_growth,
        y_val_risk=y_val_risk,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        model_save_path=model_config['model_save_path']
    )
    
    # Step 5: Evaluate on test set
    logger.info("\nEvaluating on test set...")
    metrics = model.evaluate(
        X_test=X_test,
        y_test_growth=y_test_growth,
        y_test_risk=y_test_risk
    )
    
    # Step 6: Save training results
    results = {
        'training_date': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'num_samples': {
            'train': int(X_train.shape[0]),
            'val': int(X_val.shape[0]),
            'test': int(X_test.shape[0])
        },
        'model_config': model_config,
        'test_metrics': {k: float(v) for k, v in metrics.items()},
        'best_epoch': len(history['loss'])
    }
    
    results_path = "models/training_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTraining results saved to {results_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model saved to: {model_config['model_save_path']}")
    logger.info(f"Test Growth MAE: {metrics.get('pest_growth_mae', 0):.4f}")
    logger.info(f"Test Risk Accuracy: {metrics.get('risk_level_accuracy', 0):.4f}")
    logger.info("="*60)
    
    return model, metrics, history


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train pest prediction model")
    parser.add_argument("--dataset", default="data/synthetic_pest_dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--generate", action="store_true",
                       help="Generate new synthetic dataset")
    parser.add_argument("--samples", type=int, default=2000,
                       help="Number of samples to generate")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    train_pest_model(
        dataset_path=args.dataset,
        generate_new_data=args.generate,
        num_samples=args.samples,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
