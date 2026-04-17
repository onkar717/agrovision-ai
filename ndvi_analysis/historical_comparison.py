"""
NDVI Historical Comparison and Anomaly Detection

Compares current NDVI values with historical baselines to detect vegetation stress.
Key features:
- Statistical anomaly detection using z-scores
- Seasonal baseline comparison
- Trend analysis over time
- Correlation with pest predictions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging
from ndvi_analysis.compute import NDVICompute


class NDVIComparison:
    """
    Compare current NDVI with historical data to detect anomalies
    
    Methods:
    1. Load historical NDVI for the same season (previous years)
    2. Calculate statistical baseline (mean, std)
    3. Detect anomalies using threshold-based and z-score methods
    4. Correlate drops with pest risk predictions
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize NDVI comparison engine
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        self.ndvi_config = self.config['ndvi']
        self.historical_years = self.ndvi_config['historical_years']
        self.anomaly_threshold = self.ndvi_config['anomaly_threshold']
        
        self.ndvi_computer = NDVICompute(config_path)
        
        logger.info("NDVI comparison engine initialized")
    
    def load_historical_data(
        self,
        plot_id: str,
        years: int = 3
    ) -> pd.DataFrame:
        """
        Load or simulate historical NDVI data
        
        Args:
            plot_id: Plot identifier
            years: Number of historical years
            
        Returns:
            DataFrame with historical NDVI values
        """
        all_data = []
        
        # Simulate historical data for previous years
        for year_offset in range(1, years + 1):
            end_date = datetime.now() - timedelta(days=365 * year_offset)
            start_date = end_date - timedelta(days=365)
            
            readings = self.ndvi_computer.simulate_ndvi_timeseries(
                plot_id=plot_id,
                start_date=start_date,
                end_date=end_date,
                crop_health="healthy"  # Historical baseline is healthy
            )
            
            for reading in readings:
                reading['year'] = end_date.year
            
            all_data.extend(readings)
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        logger.info(f"Loaded {len(df)} historical NDVI readings for {plot_id}")
        
        return df
    
    def calculate_baseline(
        self,
        historical_df: pd.DataFrame,
        groupby: str = 'week_of_year'
    ) -> pd.DataFrame:
        """
        Calculate statistical baseline from historical data
        
        Args:
            historical_df: Historical NDVI DataFrame
            groupby: Temporal grouping ('week_of_year' or 'day_of_year')
            
        Returns:
            DataFrame with mean, std, min, max for each time period
        """
        baseline = historical_df.groupby(groupby)['ndvi'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        
        logger.info(f"Calculated baseline statistics grouped by {groupby}")
        
        return baseline
    
    def detect_anomaly(
        self,
        current_ndvi: float,
        baseline_mean: float,
        baseline_std: float,
        threshold: float = 1.5
    ) -> Tuple[bool, float]:
        """
        Detect if current NDVI is anomalous compared to baseline
        
        Uses z-score method: z = (x - μ) / σ
        
        Args:
            current_ndvi: Current NDVI value
            baseline_mean: Historical mean
            baseline_std: Historical standard deviation
            threshold: Number of standard deviations for anomaly
            
        Returns:
            Tuple of (is_anomaly, z_score)
        """
        if baseline_std == 0:
            z_score = 0
        else:
            z_score = (current_ndvi - baseline_mean) / baseline_std
        
        is_anomaly = z_score < -threshold  # Negative z-score indicates below-normal NDVI
        
        return is_anomaly, z_score
    
    def compare_current_to_historical(
        self,
        plot_id: str,
        current_readings: List[Dict[str, Any]],
        historical_years: int = 3
    ) -> Dict[str, Any]:
        """
        Compare current NDVI readings to historical baseline
        
        Args:
            plot_id: Plot identifier
            current_readings: List of recent NDVI readings
            historical_years: Years of historical data to use
            
        Returns:
            Comparison results with anomalies and trends
        """
        # Load historical data
        historical_df = self.load_historical_data(plot_id, years=historical_years)
        
        # Calculate baseline
        baseline = self.calculate_baseline(historical_df)
        
        # Analyze current readings
        anomalies = []
        
        for reading in current_readings:
            date = pd.to_datetime(reading['date'])
            week_of_year = date.isocalendar().week
            current_ndvi = reading['ndvi']
            
            # Get baseline for this week
            baseline_week = baseline[baseline['week_of_year'] == week_of_year]
            
            if len(baseline_week) > 0:
                baseline_mean = baseline_week.iloc[0]['mean']
                baseline_std = baseline_week.iloc[0]['std']
                
                is_anomaly, z_score = self.detect_anomaly(
                    current_ndvi,
                    baseline_mean,
                    baseline_std,
                    threshold=self.anomaly_threshold
                )
                
                if is_anomaly:
                    anomalies.append({
                        'date': reading['date'],
                        'current_ndvi': current_ndvi,
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'z_score': z_score,
                        'deviation': current_ndvi - baseline_mean,
                        'deviation_percent': ((current_ndvi - baseline_mean) / baseline_mean) * 100
                    })
        
        # Calculate overall comparison
        current_mean = np.mean([r['ndvi'] for r in current_readings])
        historical_mean = historical_df['ndvi'].mean()
        
        comparison = {
            'plot_id': plot_id,
            'analysis_date': datetime.now().isoformat(),
            'current_period': {
                'start': current_readings[0]['date'],
                'end': current_readings[-1]['date'],
                'mean_ndvi': round(current_mean, 3),
                'readings_count': len(current_readings)
            },
            'historical_baseline': {
                'years': historical_years,
                'mean_ndvi': round(historical_mean, 3),
                'readings_count': len(historical_df)
            },
            'comparison': {
                'absolute_difference': round(current_mean - historical_mean, 3),
                'percent_difference': round(((current_mean - historical_mean) / historical_mean) * 100, 2),
                'is_below_normal': current_mean < historical_mean
            },
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'vegetation_health_status': self._assess_health_status(current_mean, historical_mean, len(anomalies))
        }
        
        logger.info(f"NDVI comparison complete for {plot_id}: {len(anomalies)} anomalies detected")
        
        return comparison
    
    def _assess_health_status(
        self,
        current_mean: float,
        historical_mean: float,
        anomaly_count: int
    ) -> str:
        """
        Assess overall vegetation health status
        
        Args:
            current_mean: Current mean NDVI
            historical_mean: Historical mean NDVI
            anomaly_count: Number of anomalies detected
            
        Returns:
            Health status string
        """
        deviation_percent = ((current_mean - historical_mean) / historical_mean) * 100
        
        if deviation_percent < -15 or anomaly_count > 5:
            return "Critical - Severe vegetation stress detected"
        elif deviation_percent < -10 or anomaly_count > 3:
            return "Warning - Moderate vegetation stress detected"
        elif deviation_percent < -5 or anomaly_count > 1:
            return "Advisory - Mild vegetation stress detected"
        elif deviation_percent > 5:
            return "Excellent - Above-average vegetation health"
        else:
            return "Normal - Vegetation health within expected range"
    
    def correlate_with_pest_risk(
        self,
        ndvi_comparison: Dict[str, Any],
        pest_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Correlate NDVI drops with pest risk predictions
        
        Args:
            ndvi_comparison: NDVI comparison results
            pest_prediction: Pest prediction results
            
        Returns:
            Correlation analysis
        """
        ndvi_stress = ndvi_comparison['comparison']['is_below_normal']
        pest_risk_high = pest_prediction['prediction']['risk_level'] in ['High', 'Medium']
        
        correlation = {
            'ndvi_below_normal': ndvi_stress,
            'pest_risk_elevated': pest_risk_high,
            'potentially_correlated': ndvi_stress and pest_risk_high,
            'interpretation': ""
        }
        
        if ndvi_stress and pest_risk_high:
            correlation['interpretation'] = (
                "⚠️  CORRELATED STRESS: Both NDVI and pest risk are elevated. "
                "Vegetation stress may be due to pest activity. Immediate intervention recommended."
            )
        elif ndvi_stress and not pest_risk_high:
            correlation['interpretation'] = (
                "NDVI stress detected without high pest risk. Possible causes: water stress, "
                "nutrient deficiency, or early-stage pest infestation not yet detected by the model."
            )
        elif not ndvi_stress and pest_risk_high:
            correlation['interpretation'] = (
                "Pest risk elevated but vegetation appears healthy. Model predicting potential "
                "future infestation. Preventive measures recommended."
            )
        else:
            correlation['interpretation'] = (
                "✅ No significant stress indicators. Continue routine monitoring."
            )
        
        logger.info(f"Correlation analysis: {correlation['interpretation']}")
        
        return correlation


def main():
    """Test NDVI comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare current NDVI to historical baseline")
    parser.add_argument("--plot-id", required=True, help="Plot identifier")
    parser.add_argument("--days", type=int, default=30, help="Days of current data")
    parser.add_argument("--health", default="stressed",
                       choices=["healthy", "stressed", "pest_affected"],
                       help="Simulated current health")
    
    args = parser.parse_args()
    
    comparison_engine = NDVIComparison()
    
    # Generate current readings
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    current_readings = comparison_engine.ndvi_computer.simulate_ndvi_timeseries(
        plot_id=args.plot_id,
        start_date=start_date,
        end_date=end_date,
        crop_health=args.health
    )
    
    # Compare to historical
    comparison = comparison_engine.compare_current_to_historical(
        plot_id=args.plot_id,
        current_readings=current_readings,
        historical_years=3
    )
    
    # Print results
    import json
    print("\n" + "="*60)
    print("NDVI COMPARISON ANALYSIS")
    print("="*60)
    print(json.dumps(comparison, indent=2))
    print("="*60)


if __name__ == "__main__":
    main()
