"""
NDVI Computation Module

Computes Normalized Difference Vegetation Index (NDVI) from satellite imagery.
NDVI = (NIR - Red) / (NIR + Red)

NDVI is a key indicator of vegetation health:
- Values close to +1 → Dense, healthy vegetation
- Values around 0.5-0.7 → Moderate vegetation
- Values below 0.3 → Stressed or sparse vegetation
- Negative values → Water, clouds, or bare soil

This module integrates with Google Earth Engine API for Sentinel-2 data.
In production, this could also use Landsat-8 or local drone/satellite imagery.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging


class NDVICompute:
    """
    Compute NDVI from satellite imagery
    
    For this demonstration, we'll simulate NDVI values.
    In production, this would:
    1. Query Google Earth Engine API for Sentinel-2 imagery
    2. Filter by cloud coverage and date range
    3. Extract Red and NIR bands
    4. Calculate NDVI for the specified geospatial coordinates
    5. Return time-series NDVI data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize NDVI computer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        self.ndvi_config = self.config['ndvi']
        self.satellite_source = self.ndvi_config['satellite_source']
        self.cloud_threshold = self.ndvi_config['cloud_threshold']
        
        logger.info(f"NDVI computer initialized with source: {self.satellite_source}")
    
    def compute_ndvi(
        self,
        nir_band: np.ndarray,
        red_band: np.ndarray
    ) -> np.ndarray:
        """
        Calculate NDVI from NIR and Red bands
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            nir_band: Near-infrared band values
            red_band: Red band values
            
        Returns:
            NDVI values (range: -1 to +1)
        """
        # Avoid division by zero
        denominator = nir_band + red_band
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        ndvi = (nir_band - red_band) / denominator
        
        # Clamp to valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def simulate_ndvi_timeseries(
        self,
        plot_id: str,
        start_date: datetime,
        end_date: datetime,
        crop_health: str = "healthy"
    ) -> List[Dict[str, Any]]:
        """
        Simulate NDVI time series for demonstration
        
        In production, replace this with actual Earth Engine queries
        
        Args:
            plot_id: Plot identifier
            start_date: Start date
            end_date: End date
            crop_health: Simulated health ("healthy", "stressed", "pest_affected")
            
        Returns:
            List of NDVI readings with timestamps
        """
        # Generate dates (weekly intervals)
        current_date = start_date
        readings = []
        
        while current_date <= end_date:
            # Base NDVI value depends on crop health
            if crop_health == "healthy":
                base_ndvi = 0.75
                variation = 0.05
            elif crop_health == "stressed":
                base_ndvi = 0.55
                variation = 0.08
            elif crop_health == "pest_affected":
                base_ndvi = 0.45
                variation = 0.10
                # Add declining trend
                days_elapsed = (current_date - start_date).days
                decline = -0.002 * days_elapsed
                base_ndvi += decline
            else:
                base_ndvi = 0.65
                variation = 0.07
            
            # Add seasonal variation
            day_of_year = current_date.timetuple().tm_yday
            seasonal_component = 0.1 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add random noise
            noise = np.random.normal(0, variation)
            
            ndvi_value = base_ndvi + seasonal_component + noise
            ndvi_value = max(0.2, min(0.9, ndvi_value))  # Realistic bounds
            
            readings.append({
                'plot_id': plot_id,
                'date': current_date.isoformat(),
                'timestamp': current_date.timestamp(),
                'ndvi': round(ndvi_value, 3),
                'cloud_coverage': np.random.randint(0, 15),  # Low cloud coverage
                'data_quality': 'good'
            })
            
            # Move to next week
            current_date += timedelta(days=7)
        
        logger.info(f"Generated {len(readings)} NDVI readings for plot {plot_id}")
        
        return readings
    
    def fetch_current_ndvi(
        self,
        latitude: float,
        longitude: float,
        crop_health: str = "healthy"
    ) -> float:
        """
        Fetch current NDVI for coordinates
        
        Args:
            latitude: Plot latitude
            longitude: Plot longitude
            crop_health: Simulated health status
            
        Returns:
            Current NDVI value
        """
        # In production, this would query Earth Engine API
        # For simulation, generate a realistic value
        
        readings = self.simulate_ndvi_timeseries(
            plot_id=f"lat{latitude}_lon{longitude}",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            crop_health=crop_health
        )
        
        # Return most recent reading
        return readings[-1]['ndvi']
    
    def save_ndvi_data(self, plot_id: str, readings: List[Dict[str, Any]], output_path: str = "data/ndvi_data"):
        """
        Save NDVI data to file
        
        Args:
            plot_id: Plot identifier
            readings: List of NDVI readings
            output_path: Output directory
        """
        os.makedirs(output_path, exist_ok=True)
        
        filename = os.path.join(output_path, f"{plot_id}_ndvi.json")
        
        with open(filename, 'w') as f:
            json.dump(readings, f, indent=2)
        
        logger.info(f"Saved NDVI data to {filename}")


# Production implementation with Earth Engine (commented out)
"""
import ee

class EarthEngineNDVI:
    def __init__(self):
        try:
            ee.Initialize()
            logger.info("Earth Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.info("Run: earthengine authenticate")
    
    def get_sentinel2_ndvi(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        cloud_threshold: int = 20
    ) -> List[float]:
        # Define region of interest
        point = ee.Geometry.Point([longitude, latitude])
        roi = point.buffer(500)  # 500m buffer
        
        # Load Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
        
        def calculate_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        # Calculate NDVI for each image
        ndvi_collection = collection.map(calculate_ndvi)
        
        # Extract NDVI values
        ndvi_values = []
        for image_info in ndvi_collection.getInfo()['features']:
            ndvi = image_info['properties']['NDVI']
            date = image_info['properties']['system:time_start']
            ndvi_values.append({'date': date, 'ndvi': ndvi})
        
        return ndvi_values
"""


def main():
    """Test NDVI computation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute NDVI for a plot")
    parser.add_argument("--plot-id", required=True, help="Plot identifier")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history")
    parser.add_argument("--health", default="healthy",
                       choices=["healthy", "stressed", "pest_affected"],
                       help="Simulated crop health")
    
    args = parser.parse_args()
    
    ndvi_computer = NDVICompute()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    readings = ndvi_computer.simulate_ndvi_timeseries(
        plot_id=args.plot_id,
        start_date=start_date,
        end_date=end_date,
        crop_health=args.health
    )
    
    # Print summary
    print("\n" + "="*60)
    print(f"NDVI Time Series for Plot: {args.plot_id}")
    print("="*60)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Health Status: {args.health}")
    print(f"Number of readings: {len(readings)}")
    print(f"Latest NDVI: {readings[-1]['ndvi']:.3f}")
    print(f"Average NDVI: {np.mean([r['ndvi'] for r in readings]):.3f}")
    print("="*60)
    
    # Save data
    ndvi_computer.save_ndvi_data(args.plot_id, readings)


if __name__ == "__main__":
    main()
