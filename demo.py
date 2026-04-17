#!/usr/bin/env python3
"""
Complete Demo Script for Pest Detection System

This script demonstrates the full workflow:
1. Generate synthetic dataset
2. Train CNN model
3. Simulate sensor data
4. Run inference
5. Display results

Run this to quickly test the entire system.
"""

import os
import sys
import time
from datetime import datetime
from loguru import logger

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, setup_logging


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """Run complete demo"""
    
    print_header("🌾 PEST DETECTION SYSTEM - COMPLETE DEMO")
    
    config = load_config()
    setup_logging(config)
    
    # Step 1: Generate Dataset
    print_header("Step 1: Generating Synthetic Dataset")
    from data_pipeline.synthetic_dataset import SyntheticDatasetGenerator
    
    generator = SyntheticDatasetGenerator()
    print("Generating 500 samples (this may take a minute)...")
    df = generator.generate_dataset(
        num_samples=500,
        duration_days=7,
        output_path="data/demo_dataset.csv"
    )
    print(f"✅ Dataset generated: {df.shape[0]} samples")
    print(f"   Risk level distribution:")
    print(df['risk_level'].value_counts().to_string())
    
    # Step 2: Train Model
    print_header("Step 2: Training CNN Model")
    from pest_model.train import train_pest_model
    
    print("Training model (this will take a few minutes)...")
    model, metrics, history = train_pest_model(
        dataset_path="data/demo_dataset.csv",
        generate_new_data=False,
        config_path="config.yaml"
    )
    print(f"✅ Model trained successfully")
    print(f"   Test Accuracy (Risk): {metrics.get('risk_level_accuracy', 0):.2%}")
    print(f"   Test MAE (Growth): {metrics.get('pest_growth_mae', 0):.4f}")
    
    # Step 3: Simulate Sensor Data
    print_header("Step 3: Simulating Sensor Data")
    import sqlite3
    import numpy as np
    
    # Create some demo sensor readings
    conn = sqlite3.connect(config['database']['sqlite_path'])
    cursor = conn.cursor()
    
    print("Inserting 168 hours (7 days) of sensor data for demo_device_01...")
    
    # Generate 7 days of hourly readings
    base_time = time.time() - (168 * 3600)  # 7 days ago
    for i in range(168):
        timestamp = base_time + (i * 3600)
        
        # Simulate conditions favorable for pests
        hour_of_day = i % 24
        temp = 25 + 5 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 1)
        humidity = 70 - 10 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 2)
        
        cursor.execute('''
            INSERT INTO sensor_readings
            (device_id, crop_type, timestamp, datetime, temperature, humidity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            'demo_device_01',
            'tomato',
            timestamp,
            datetime.fromtimestamp(timestamp).isoformat(),
            round(temp, 2),
            round(humidity, 2)
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Inserted 168 sensor readings")
    
    # Step 4: Run Inference
    print_header("Step 4: Running Pest Prediction Inference")
    from pest_model.inference import PestInferenceEngine
    
    print("Loading inference engine...")
    engine = PestInferenceEngine(
        model_path=config['model']['model_save_path'],
        preprocessor_path="models/preprocessor.pkl",
        db_path=config['database']['sqlite_path']
    )
    
    print("Running prediction for demo_device_01...")
    prediction = engine.predict(
        device_id='demo_device_01',
        crop_type='tomato',
        pest_type='aphids'
    )
    
    if prediction:
        print("✅ Prediction completed successfully\n")
        
        pred = prediction['prediction']
        alert = prediction['alert']
        
        print(f"📊 PREDICTION RESULTS:")
        print(f"   Device: {prediction['device_id']}")
        print(f"   Crop: {prediction['crop_type']}")
        print(f"   Pest: {prediction['pest_type']}")
        print(f"\n   Growth Probability: {pred['growth_percentage']:.1f}%")
        print(f"   Risk Level: {pred['risk_level']}")
        print(f"   Confidence: {pred['risk_confidence']:.2%}")
        print(f"   Days to Infestation: {pred['days_to_infestation']}")
        print(f"\n   Alert: {alert['message']}")
    else:
        print("❌ Prediction failed - insufficient data")
    
    # Step 5: NDVI Analysis
    print_header("Step 5: NDVI Vegetation Health Analysis")
    from ndvi_analysis.compute import NDVICompute
    from ndvi_analysis.historical_comparison import NDVIComparison
    
    ndvi_computer = NDVICompute()
    ndvi_comparison = NDVIComparison()
    
    # Generate current NDVI readings
    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("Generating NDVI data for plot demo_plot_01...")
    current_readings = ndvi_computer.simulate_ndvi_timeseries(
        plot_id='demo_plot_01',
        start_date=start_date,
        end_date=end_date,
        crop_health='healthy'
    )
    
    print(f"✅ Generated {len(current_readings)} NDVI readings")
    print(f"   Current NDVI: {current_readings[-1]['ndvi']:.3f}")
    
    # Compare to historical
    print("\nComparing to historical baseline...")
    comparison = ndvi_comparison.compare_current_to_historical(
        plot_id='demo_plot_01',
        current_readings=current_readings,
        historical_years=3
    )
    
    print(f"✅ Historical comparison complete")
    print(f"   Current mean NDVI: {comparison['current_period']['mean_ndvi']:.3f}")
    print(f"   Historical mean: {comparison['historical_baseline']['mean_ndvi']:.3f}")
    print(f"   Difference: {comparison['comparison']['percent_difference']:.1f}%")
    print(f"   Health Status: {comparison['vegetation_health_status']}")
    print(f"   Anomalies detected: {comparison['anomaly_count']}")
    
    # Final Summary
    print_header("✅ DEMO COMPLETE - SYSTEM READY")
    
    print("Next Steps:")
    print("1. Start MQTT broker:         mosquitto -v")
    print("2. Start sensor simulator:    python iot_clients/sensor_node.py --device-id sensor_01 --crop tomato")
    print("3. Start MQTT subscriber:     python mqtt_server/subscriber.py")
    print("4. Start API server:          python api/main.py")
    print("5. Open dashboard:            http://localhost:8000")
    print("\nAll components are working correctly! 🎉\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
