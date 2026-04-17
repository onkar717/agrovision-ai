"""
Synthetic Dataset Generator

Generates training data for the CNN pest prediction model.
Creates realistic time-series patterns of temperature and humidity correlated with pest growth.

In production, this would be replaced with:
- Historical weather data from agricultural extension services
- Real pest infestation records from farmer logs
- Integrated pest management (IPM) databases
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Tuple, List
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging


class SyntheticDatasetGenerator:
    """
    Generates synthetic sensor data with pest growth labels
    
    The dataset simulates realistic scenarios:
    1. Favorable conditions → Gradual pest population increase
    2. Unfavorable conditions → Pest population decline or stasis
    3. Seasonal variations in temperature and humidity
    4. Random environmental noise
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize dataset generator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        self.crops = self.config['crops']
        self.pests = self.config['pests']
        
        logger.info("Synthetic dataset generator initialized")
    
    def generate_time_series(
        self,
        crop_type: str,
        pest_type: str,
        duration_days: int = 30,
        samples_per_day: int = 24,  # Hourly readings
        scenario: str = "favorable"
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a single time-series sample with pest growth
        
        Args:
            crop_type: Type of crop
            pest_type: Type of pest
            duration_days: Length of time series in days
            samples_per_day: Samples per day (24 = hourly)
            scenario: "favorable", "unfavorable", or "mixed"
            
        Returns:
            Tuple of (time_series_array, final_pest_growth_level)
            time_series_array shape: (duration_days * samples_per_day, 2) for [temp, humidity]
        """
        pest_info = self.pests.get(pest_type)
        if not pest_info:
            raise ValueError(f"Unknown pest type: {pest_type}")
        
        optimal_temp_range = pest_info['optimal_temp_range']
        optimal_humidity_range = pest_info['optimal_humidity_range']
        growth_cycle_days = pest_info['growth_cycle_days']
        
        num_samples = duration_days * samples_per_day
        hours = np.arange(num_samples) / samples_per_day
        
        # Generate temperature time series
        if scenario == "favorable":
            # Temperature stays in optimal range
            base_temp = np.mean(optimal_temp_range)
            temp_variation = 3.0
        elif scenario == "unfavorable":
            # Temperature outside optimal range
            base_temp = optimal_temp_range[0] - 5 if random.random() > 0.5 else optimal_temp_range[1] + 5
            temp_variation = 4.0
        else:  # mixed
            base_temp = random.uniform(optimal_temp_range[0] - 3, optimal_temp_range[1] + 3)
            temp_variation = 5.0
        
        # Diurnal cycle (daily temperature variation)
        diurnal_temp = temp_variation * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        
        # Seasonal drift
        seasonal_drift = 2 * np.sin(2 * np.pi * hours / (365 * 24))
        
        # Random noise
        noise_temp = np.random.normal(0, 1.0, num_samples)
        
        temperature = base_temp + diurnal_temp + seasonal_drift + noise_temp
        temperature = np.clip(temperature, 10, 40)  # Realistic bounds
        
        # Generate humidity (inversely correlated with temperature)
        if scenario == "favorable":
            base_humidity = np.mean(optimal_humidity_range)
            humidity_variation = 10.0
        elif scenario == "unfavorable":
            base_humidity = optimal_humidity_range[0] - 15 if random.random() > 0.5 else optimal_humidity_range[1] + 10
            humidity_variation = 12.0
        else:
            base_humidity = random.uniform(optimal_humidity_range[0] - 10, optimal_humidity_range[1] + 10)
            humidity_variation = 15.0
        
        # Inverse correlation with temperature
        temp_correlation = -0.8 * (temperature - base_temp)
        
        # Diurnal cycle (opposite phase to temperature)
        diurnal_humidity = humidity_variation * np.sin(2 * np.pi * hours / 24 + np.pi/2)
        
        # Random noise
        noise_humidity = np.random.normal(0, 3.0, num_samples)
        
        humidity = base_humidity + temp_correlation + diurnal_humidity + noise_humidity
        humidity = np.clip(humidity, 30, 95)  # Realistic bounds
        
        # Calculate pest growth level based on environmental favorability
        # Growth is cumulative over time when conditions are favorable
        pest_growth = 0.0
        growth_history = []
        
        for i in range(num_samples):
            temp = temperature[i]
            hum = humidity[i]
            
            # Check if conditions are favorable
            temp_favorable = optimal_temp_range[0] <= temp <= optimal_temp_range[1]
            humidity_favorable = optimal_humidity_range[0] <= hum <= optimal_humidity_range[1]
            
            # Daily growth rate (0-100% over growth_cycle_days)
            max_daily_growth = 100 / (growth_cycle_days * samples_per_day)
            
            if temp_favorable and humidity_favorable:
                # Optimal growth
                daily_growth = max_daily_growth * random.uniform(0.8, 1.0)
            elif temp_favorable or humidity_favorable:
                # Moderate growth
                daily_growth = max_daily_growth * random.uniform(0.3, 0.6)
            else:
                # Minimal or negative growth
                daily_growth = max_daily_growth * random.uniform(-0.2, 0.1)
            
            pest_growth = max(0, min(100, pest_growth + daily_growth))
            growth_history.append(pest_growth)
        
        # Stack temperature and humidity into feature array
        time_series = np.column_stack([temperature, humidity])
        
        final_pest_growth = pest_growth
        
        return time_series, final_pest_growth
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        duration_days: int = 7,
        output_path: str = "data/synthetic_pest_dataset.csv"
    ) -> pd.DataFrame:
        """
        Generate full dataset with multiple samples
        
        Args:
            num_samples: Number of time-series samples to generate
            duration_days: Length of each time series in days
            output_path: Path to save CSV
            
        Returns:
            DataFrame with time-series sequences and labels
        """
        logger.info(f"Generating {num_samples} samples with {duration_days}-day sequences...")
        
        samples = []
        samples_per_day = 24  # Hourly readings
        
        crop_types = list(self.crops.keys())
        
        for i in range(num_samples):
            # Random crop and pest selection
            crop_type = random.choice(crop_types)
            crop_info = self.crops[crop_type]
            pest_type = random.choice(crop_info['pests'])
            
            # Random scenario distribution
            scenario = random.choices(
                ["favorable", "unfavorable", "mixed"],
                weights=[0.4, 0.3, 0.3]
            )[0]
            
            # Generate time series
            time_series, pest_growth = self.generate_time_series(
                crop_type=crop_type,
                pest_type=pest_type,
                duration_days=duration_days,
                samples_per_day=samples_per_day,
                scenario=scenario
            )
            
            # Determine risk level
            if pest_growth < 30:
                risk_level = "Low"
            elif pest_growth < 60:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Estimate days to infestation (simplified)
            pest_info = self.pests[pest_type]
            if pest_growth > 70:
                days_to_infestation = random.randint(1, 3)
            elif pest_growth > 40:
                days_to_infestation = random.randint(4, 10)
            else:
                days_to_infestation = random.randint(11, 20)
            
            # Flatten time series for CSV storage
            # Each row: [crop, pest, temp_0, hum_0, temp_1, hum_1, ..., growth, risk]
            flattened = time_series.flatten()
            
            sample = {
                'crop_type': crop_type,
                'pest_type': pest_type,
                'scenario': scenario,
                'pest_growth': pest_growth,
                'risk_level': risk_level,
                'days_to_infestation': days_to_infestation
            }
            
            # Add time series features
            for j in range(len(flattened)):
                feature_name = f"feature_{j}"
                sample[feature_name] = flattened[j]
            
            samples.append(sample)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        df = pd.DataFrame(samples)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Risk level distribution:\n{df['risk_level'].value_counts()}")
        
        return df


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic pest prediction dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--days", type=int, default=7, help="Duration of each sequence in days")
    parser.add_argument("--output", default="data/synthetic_pest_dataset.csv", help="Output CSV path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    generator = SyntheticDatasetGenerator(config_path=args.config)
    generator.generate_dataset(
        num_samples=args.samples,
        duration_days=args.days,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
