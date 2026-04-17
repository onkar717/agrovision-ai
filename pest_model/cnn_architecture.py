"""
CNN Model Architecture for Pest Growth Prediction

This module defines the CNN model for time-series pest prediction.

Why CNN instead of traditional ML (Random Forest, SVM, etc.)?
--------------------------------------------------------
1. **Temporal Pattern Recognition**: CNNs excel at detecting local patterns in sequential data.
   Conv1D kernels learn to identify specific environmental signatures (e.g., 3-day heat waves
   followed by high humidity) that correlate with pest outbreaks.

2. **Automatic Feature Learning**: Traditional ML requires manual feature engineering (rolling
   averages, lag features, etc.). CNNs automatically learn hierarchical features from raw data.

3. **Invariance to Time Shifts**: Convolutional layers detect patterns regardless of when they
   occur in the sequence. A "favorable growth window" triggers the same response whether it
   occurs on day 2 or day 5.

4. **Seasonal & Diurnal Cycles**: Multiple conv layers with different kernel sizes capture
   both short-term (hourly) and long-term (daily/weekly) patterns simultaneously.

5. **Lightweight for Edge Deployment**: CNNs are more compact than LSTMs/Transformers,
   making them suitable for Raspberry Pi inference with acceptable latency.

6. **Robustness to Missing Data**: Pooling layers and dropout provide resilience to
   occasional sensor failures or transmission gaps.

Model Architecture:
-------------------
Input: [batch_size, 168 timesteps, 2 features] → 7 days of hourly temp + humidity
    ↓
Conv1D(64 filters, kernel=5) + ReLU → Captures 5-hour patterns
    ↓
MaxPooling1D(2) → Downsamples to 84 timesteps
    ↓
Dropout(0.3) → Regularization
    ↓
Conv1D(128 filters, kernel=3) + ReLU → Captures refined 3-hour patterns
    ↓
MaxPooling1D(2) → Downsamples to 42 timesteps
    ↓
Dropout(0.3)
    ↓
Flatten → Dense(128) + ReLU
    ↓
Dropout(0.5)
    ↓
Output branches:
    1. Pest Growth: Dense(1, sigmoid) → Continuous value [0, 1]
    2. Risk Level: Dense(3, softmax) → {Low, Medium, High}
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from loguru import logger
import numpy as np
from typing import Tuple, Dict, Any
import os


class PestCNNModel:
    """
    CNN-based pest growth prediction model
    
    Multi-task learning:
    - Task 1: Regression for pest growth percentage (0-100)
    - Task 2: Classification for risk level (Low/Medium/High)
    """
    
    def __init__(
        self,
        sequence_length: int = 168,
        n_features: int = 2,
        cnn_filters: list = [64, 128],
        kernel_sizes: list = [5, 3],
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 128,
        n_risk_classes: int = 3
    ):
        """
        Initialize CNN model architecture
        
        Args:
            sequence_length: Number of time steps (168 = 7 days @ hourly)
            n_features: Number of features per timestep (2 = temp + humidity)
            cnn_filters: List of filter counts for each Conv1D layer
            kernel_sizes: List of kernel sizes for each Conv1D layer
            pool_size: MaxPooling size
            dropout_rate: Dropout rate
            dense_units: Units in dense layer
            n_risk_classes: Number of risk classes (3 = Low/Medium/High)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.n_risk_classes = n_risk_classes
        
        self.model = None
        self._build_model()
        
        logger.info("CNN model architecture created")
    
    def _build_model(self):
        """Build the CNN architecture"""
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='sensor_input')
        
        x = inputs
        
        # Convolutional layers
        for i, (filters, kernel_size) in enumerate(zip(self.cnn_filters, self.kernel_sizes)):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            
            x = layers.MaxPooling1D(
                pool_size=self.pool_size,
                name=f'maxpool_{i+1}'
            )(x)
            
            x = layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout_{i+1}'
            )(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layer
        x = layers.Dense(
            units=self.dense_units,
            activation='relu',
            name='dense'
        )(x)
        
        x = layers.Dropout(
            rate=0.5,
            name='dropout_dense'
        )(x)
        
        # Output branches
        # Branch 1: Pest growth (regression)
        growth_output = layers.Dense(
            units=1,
            activation='sigmoid',
            name='pest_growth'
        )(x)
        
        # Branch 2: Risk level (classification)
        risk_output = layers.Dense(
            units=self.n_risk_classes,
            activation='softmax',
            name='risk_level'
        )(x)
        
        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs={'pest_growth': growth_output, 'risk_level': risk_output},
            name='PestPredictionCNN'
        )
        
        logger.info(f"Model built: {self.model.name}")
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile model with loss functions and metrics
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'pest_growth': 'mean_squared_error',  # Regression loss
                'risk_level': 'sparse_categorical_crossentropy'  # Classification loss
            },
            loss_weights={
                'pest_growth': 1.0,
                'risk_level': 0.5  # Weight classification slightly less
            },
            metrics={
                'pest_growth': ['mae', 'mse'],
                'risk_level': ['accuracy']
            }
        )
        
        logger.info("Model compiled with multi-task losses")
    
    def get_summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train_growth: np.ndarray,
        y_train_risk: np.ndarray,
        X_val: np.ndarray,
        y_val_growth: np.ndarray,
        y_val_risk: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        model_save_path: str = "models/pest_prediction_model"
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train_growth: Training growth labels
            y_train_risk: Training risk labels
            X_val: Validation features
            y_val_growth: Validation growth labels
            y_val_risk: Validation risk labels
            epochs: Maximum epochs
            batch_size: Batch size
            model_save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare training data
        train_data = {
            'pest_growth': y_train_growth,
            'risk_level': y_train_risk
        }
        
        val_data = {
            'pest_growth': y_val_growth,
            'risk_level': y_val_risk
        }
        
        logger.info("Starting training...")
        
        # Train model
        history = self.model.fit(
            X_train,
            train_data,
            validation_data=(X_val, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
        return history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_growth: np.ndarray,
        y_test_risk: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test_growth: Test growth labels
            y_test_risk: Test risk labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        test_data = {
            'pest_growth': y_test_growth,
            'risk_level': y_test_risk
        }
        
        results = self.model.evaluate(X_test, test_data, verbose=0)
        
        # Parse results
        metrics = {}
        metric_names = self.model.metrics_names
        
        for name, value in zip(metric_names, results):
            metrics[name] = value
            logger.info(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (growth_predictions, risk_predictions)
        """
        predictions = self.model.predict(X, verbose=0)
        
        growth_pred = predictions['pest_growth'].flatten()
        risk_pred = predictions['risk_level']
        risk_classes = np.argmax(risk_pred, axis=1)
        
        return growth_pred, risk_classes
    
    def save(self, save_path: str = "models/pest_prediction_model"):
        """Save model"""
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    @staticmethod
    def load(load_path: str = "models/pest_prediction_model") -> 'PestCNNModel':
        """Load saved model"""
        loaded_model = keras.models.load_model(load_path)
        
        # Create instance and attach loaded model
        instance = PestCNNModel()
        instance.model = loaded_model
        
        logger.info(f"Model loaded from {load_path}")
        return instance


def main():
    """Test model architecture"""
    model = PestCNNModel(
        sequence_length=168,
        n_features=2,
        cnn_filters=[64, 128],
        kernel_sizes=[5, 3]
    )
    
    model.compile_model(learning_rate=0.001)
    print("\n" + "="*60)
    print("CNN MODEL ARCHITECTURE")
    print("="*60)
    model.get_summary()


if __name__ == "__main__":
    main()
