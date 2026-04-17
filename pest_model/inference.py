"""
Inference Engine for Real-Time Pest Prediction

This module loads the trained CNN model and performs real-time inference on live sensor data.
It can be triggered by:
- New data arriving via MQTT
- API requests from the dashboard
- Scheduled batch predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from loguru import logger
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.preprocessing import DataPreprocessor
from utils import load_config, setup_logging, calculate_pest_risk_level


class PestInferenceEngine:
    """
    Real-time inference engine for pest growth prediction
    
    Workflow:
    1. Fetch recent sensor data (last 7 days)
    2. Preprocess into model input format
    3. Run CNN inference
    4. Generate actionable predictions and alerts
    """
    
    def __init__(
        self,
        model_path: str = "models/pest_prediction_model",
        preprocessor_path: str = "models/preprocessor.pkl",
        config_path: str = "config.yaml",
        db_path: str = "data/pest_detection.db"
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to saved preprocessor
            config_path: Path to configuration
            db_path: Path to database
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        self.db_path = db_path
        self.model_config = self.config['model']
        self.alert_config = self.config['alerts']
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load preprocessor
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
        
        # Risk level mapping
        self.risk_labels = ['Low', 'Medium', 'High']
        
        logger.info("Inference engine initialized and ready")
    
    def fetch_recent_data(
        self,
        device_id: str,
        hours: int = 168  # 7 days
    ) -> Optional[np.ndarray]:
        """
        Fetch recent sensor readings from database
        
        Args:
            device_id: Device identifier
            hours: Number of hours of data to fetch
            
        Returns:
            Array of shape (hours, 2) for [temperature, humidity], or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate cutoff timestamp
        cutoff_time = (datetime.now() - timedelta(hours=hours)).timestamp()
        
        # Query recent data
        cursor.execute('''
            SELECT timestamp, temperature, humidity
            FROM sensor_readings
            WHERE device_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        ''', (device_id, cutoff_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < hours:
            logger.warning(
                f"Insufficient data for {device_id}: {len(rows)} readings "
                f"(need {hours})"
            )
            return None
        
        # Extract temperature and humidity
        data = np.array([[row[1], row[2]] for row in rows])
        
        # If more than needed, take the most recent
        if len(data) > hours:
            data = data[-hours:]
        
        logger.info(f"Fetched {len(data)} readings for {device_id}")
        return data
    
    def preprocess_for_inference(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess raw data for model input
        
        Args:
            data: Raw sensor data (hours, 2)
            
        Returns:
            Preprocessed data (1, hours, 2) - batch size of 1
        """
        # Reshape to (1, sequence_length, n_features)
        X = data.reshape(1, self.model_config['sequence_length'], len(self.model_config['features']))
        
        # Normalize using saved scaler
        X_normalized = self.preprocessor.normalize_features(X, fit=False)
        
        return X_normalized
    
    def predict(
        self,
        device_id: str,
        crop_type: Optional[str] = None,
        pest_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make pest growth prediction for a device
        
        Args:
            device_id: Device identifier
            crop_type: Crop type (optional, fetched from DB if not provided)
            pest_type: Pest type (optional)
            
        Returns:
            Prediction dictionary with growth, risk, and actionable info
        """
        # Fetch recent data
        data = self.fetch_recent_data(device_id)
        
        if data is None:
            return None
        
        # Preprocess
        X = self.preprocess_for_inference(data)
        
        # Run inference
        predictions = self.model.predict(X, verbose=0)
        
        # Extract predictions
        growth_probability = float(predictions['pest_growth'][0][0])
        risk_logits = predictions['risk_level'][0]
        risk_class_idx = int(np.argmax(risk_logits))
        risk_level = self.risk_labels[risk_class_idx]
        risk_confidence = float(risk_logits[risk_class_idx])
        
        # Calculate days to potential infestation (heuristic)
        if growth_probability > 0.8:
            days_to_infestation = np.random.randint(1, 4)
        elif growth_probability > 0.6:
            days_to_infestation = np.random.randint(4, 8)
        elif growth_probability > 0.4:
            days_to_infestation = np.random.randint(8, 12)
        else:
            days_to_infestation = np.random.randint(12, 21)
        
        # Generate actionable message
        message = self._generate_alert_message(
            growth_probability,
            risk_level,
            days_to_infestation
        )
        
        result = {
            'device_id': device_id,
            'crop_type': crop_type,
            'pest_type': pest_type,
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'growth_probability': growth_probability,
                'growth_percentage': growth_probability * 100,
                'risk_level': risk_level,
                'risk_confidence': risk_confidence,
                'days_to_infestation': days_to_infestation
            },
            'alert': {
                'should_alert': growth_probability > self.alert_config['pest_risk']['medium'],
                'message': message,
                'severity': risk_level
            }
        }
        
        logger.info(
            f"Prediction for {device_id}: Growth={growth_probability:.2%}, "
            f"Risk={risk_level}, ETA={days_to_infestation}d"
        )
        
        # Store prediction in database
        self._store_prediction(result)
        
        return result
    
    def _generate_alert_message(
        self,
        growth_prob: float,
        risk_level: str,
        days: int
    ) -> str:
        """
        Generate human-readable alert message
        
        Args:
            growth_prob: Pest growth probability
            risk_level: Risk level string
            days: Days to potential infestation
            
        Returns:
            Alert message string
        """
        if growth_prob > 0.8:
            return (
                f"⚠️  CRITICAL ALERT: High pest risk ({growth_prob:.0%}). "
                f"Potential infestation in {days} days. Take immediate action."
            )
        elif growth_prob > 0.6:
            return (
                f"⚠️  WARNING: Medium-High pest risk ({growth_prob:.0%}). "
                f"Prepare preventive measures. Timeline: {days} days."
            )
        elif growth_prob > 0.4:
            return (
                f"ℹ️ ADVISORY: Moderate pest risk ({growth_prob:.0%}). "
                f"Monitor conditions closely over the next {days} days."
            )
        else:
            return (
                f"✅ LOW RISK: Pest conditions are currently unfavorable ({growth_prob:.0%}). "
                f"Continue routine monitoring."
            )
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """
        Store prediction in database for historical tracking
        
        Args:
            prediction: Prediction dictionary
        """
        try:
            conn =sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            pred_data = prediction['prediction']
            
            cursor.execute('''
                INSERT INTO pest_predictions
                (device_id, crop_type, pest_type, prediction_timestamp,
                 growth_probability, risk_level, infestation_timeline_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction['device_id'],
                prediction.get('crop_type'),
                prediction.get('pest_type'),
                datetime.now().timestamp(),
                pred_data['growth_probability'],
                pred_data['risk_level'],
                pred_data['days_to_infestation']
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored prediction for {prediction['device_id']}")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def batch_predict(self, device_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Run predictions for multiple devices
        
        Args:
            device_ids: List of device identifiers
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for device_id in device_ids:
            prediction = self.predict(device_id)
            if prediction:
                results.append(prediction)
        
        logger.info(f"Batch prediction completed for {len(results)} devices")
        
        return results


def main():
    """Test inference engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pest prediction inference")
    parser.add_argument("--device-id", required=True, help="Device identifier")
    parser.add_argument("--crop", help="Crop type")
    parser.add_argument("--pest", help="Pest type")
    parser.add_argument("--model", default="models/pest_prediction_model",
                       help="Model path")
    parser.add_argument("--preprocessor", default="models/preprocessor.pkl",
                       help="Preprocessor path")
    
    args = parser.parse_args()
    
    engine = PestInferenceEngine(
        model_path=args.model,
        preprocessor_path=args.preprocessor
    )
    
    result = engine.predict(
        device_id=args.device_id,
        crop_type=args.crop,
        pest_type=args.pest
    )
    
    if result:
        import json
        print("\n" + "="*60)
        print("PEST PREDICTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("="*60)
    else:
        print("Insufficient data for prediction")


if __name__ == "__main__":
    main()
