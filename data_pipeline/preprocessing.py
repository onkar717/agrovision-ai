"""
Data Preprocessing Pipeline

Prepares raw sensor data and synthetic datasets for CNN model training.
Handles:
- Sliding window creation for time-series
- Feature engineering and normalization
- Train/validation/test split
- Label encoding
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
from loguru import logger
import pickle
import os


class DataPreprocessor:
    """
    Preprocess time-series sensor data for CNN training
    
    Key functions:
    1. Load and reshape flattened CSV data into (samples, timesteps, features)
    2. Normalize features using StandardScaler
    3. Encode categorical labels (crop type, pest type, risk level)
    4. Create train/val/test splits with temporal awareness
    """
    
    def __init__(self, sequence_length: int = 168, n_features: int = 2):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Number of time steps (default: 168 for 7 days @ hourly)
            n_features: Number of features per timestep (default: 2 for temp + humidity)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        self.scaler = StandardScaler()
        self.label_encoder_risk = LabelEncoder()
        
        logger.info(f"Initialized preprocessor with sequence_length={sequence_length}, n_features={n_features}")
    
    def load_synthetic_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load synthetic dataset from CSV
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame
        """
        df = pd.DataFrame(pd.read_csv(csv_path))
        logger.info(f"Loaded dataset from {csv_path}: shape {df.shape}")
        return df
    
    def extract_features_and_labels(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature arrays and labels from DataFrame
        
        Args:
            df: DataFrame with flattened time series
            
        Returns:
            Tuple of (X, y_growth, y_risk)
            - X: shape (n_samples, sequence_length, n_features)
            - y_growth: shape (n_samples,) - continuous pest growth values
            - y_risk: shape (n_samples,) - categorical risk levels
        """
        # Extract feature columns (everything starting with 'feature_')
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features_flat = df[feature_cols].values
        
        # Reshape into (n_samples, sequence_length, n_features)
        n_samples = features_flat.shape[0]
        X = features_flat.reshape(n_samples, self.sequence_length, self.n_features)
        
        # Extract growth labels (continuous, 0-100)
        y_growth = df['pest_growth'].values / 100.0  # Normalize to [0, 1]
        
        # Extract risk labels (categorical)
        y_risk = df['risk_level'].values
        
        logger.info(f"Extracted features: X shape={X.shape}, y_growth shape={y_growth.shape}")
        
        return X, y_growth, y_risk
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler
        
        Args:
            X: Feature array (n_samples, sequence_length, n_features)
            fit: If True, fit scaler on this data; if False, use existing scaler
            
        Returns:
            Normalized features
        """
        original_shape = X.shape
        
        # Flatten to (n_samples * sequence_length, n_features) for scaling
        X_flat = X.reshape(-1, self.n_features)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_flat)
            logger.info("Fitted and transformed features")
        else:
            X_scaled = self.scaler.transform(X_flat)
            logger.info("Transformed features using existing scaler")
        
        # Reshape back to original
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def encode_risk_labels(self, y_risk: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode risk level labels as integers
        
        Args:
            y_risk: Risk level strings (Low, Medium, High)
            fit: If True, fit encoder; if False, use existing
            
        Returns:
            Encoded labels as integers
        """
        if fit:
            y_encoded = self.label_encoder_risk.fit_transform(y_risk)
            logger.info(f"Risk level classes: {self.label_encoder_risk.classes_}")
        else:
            y_encoded = self.label_encoder_risk.transform(y_risk)
        
        return y_encoded
    
    def prepare_train_test_split(
        self,
        X: np.ndarray,
        y_growth: np.ndarray,
        y_risk: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y_growth: Growth labels
            y_risk: Risk labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train_growth, y_val_growth, y_test_growth,
                     y_train_risk, y_val_risk, y_test_risk)
        """
        # First split: train+val vs test
        X_temp, X_test, y_growth_temp, y_growth_test, y_risk_temp, y_risk_test = train_test_split(
            X, y_growth, y_risk, test_size=test_size, random_state=random_state, stratify=y_risk
        )
        
        # Second split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_growth_train, y_growth_val, y_risk_train, y_risk_val = train_test_split(
            X_temp, y_growth_temp, y_risk_temp, test_size=val_relative_size,
            random_state=random_state, stratify=y_risk_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, X_val, X_test,
                y_growth_train, y_growth_val, y_growth_test,
                y_risk_train, y_risk_val, y_risk_test)
    
    def save_preprocessor(self, save_path: str = "models/preprocessor.pkl"):
        """Save scaler and encoders for later use"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder_risk': self.label_encoder_risk,
                'sequence_length': self.sequence_length,
                'n_features': self.n_features
            }, f)
        
        logger.info(f"Saved preprocessor to {save_path}")
    
    @staticmethod
    def load_preprocessor(load_path: str = "models/preprocessor.pkl") -> 'DataPreprocessor':
        """Load saved preprocessor"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = DataPreprocessor(
            sequence_length=data['sequence_length'],
            n_features=data['n_features']
        )
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoder_risk = data['label_encoder_risk']
        
        logger.info(f"Loaded preprocessor from {load_path}")
        return preprocessor
    
    def preprocess_full_pipeline(
        self,
        csv_path: str,
        save_preprocessor: bool = True
    ) -> Tuple:
        """
        Full preprocessing pipeline: load → extract → normalize → split
        
        Args:
            csv_path: Path to CSV dataset
            save_preprocessor: Whether to save scaler/encoders
            
        Returns:
            Train/val/test splits
        """
        # Load data
        df = self.load_synthetic_dataset(csv_path)
        
        # Extract features and labels
        X, y_growth, y_risk = self.extract_features_and_labels(df)
        
        # Normalize features
        X_normalized = self.normalize_features(X, fit=True)
        
        # Encode risk labels
        y_risk_encoded = self.encode_risk_labels(y_risk, fit=True)
        
        # Split data
        splits = self.prepare_train_test_split(X_normalized, y_growth, y_risk_encoded)
        
        # Save preprocessor
        if save_preprocessor:
            self.save_preprocessor()
        
        return splits


def main():
    """Test preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("--input", default="data/synthetic_pest_dataset.csv", help="Input CSV path")
    parser.add_argument("--sequence-length", type=int, default=168, help="Sequence length")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(sequence_length=args.sequence_length)
    splits = preprocessor.preprocess_full_pipeline(args.input)
    
    X_train, X_val, X_test = splits[0], splits[1], splits[2]
    logger.info(f"Preprocessing complete. Training samples: {X_train.shape[0]}")


if __name__ == "__main__":
    main()
