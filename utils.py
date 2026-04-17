"""
Utility functions for the Pest Detection System
"""

import yaml
import os
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure logging based on config settings
    
    Args:
        config: Configuration dictionary (loads from file if None)
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', 'logs/pest_detection.log')
    log_level = log_config.get('level', 'INFO')
    rotation = log_config.get('rotation', '10 MB')
    retention = log_config.get('retention', '30 days')
    log_format = log_config.get('format', 
                                 '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}')
    
    # Ensure log directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True
    )
    
    logger.info("Logging initialized successfully")


def validate_sensor_reading(reading: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Validate sensor reading against configured bounds
    
    Args:
        reading: Sensor reading dictionary
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    sensor_config = config.get('sensors', {})
    
    # Validate temperature
    if 'temperature' in reading:
        temp = reading['temperature']
        temp_config = sensor_config.get('temperature', {})
        min_temp = temp_config.get('min_value', -10)
        max_temp = temp_config.get('max_value', 50)
        
        if not (min_temp <= temp <= max_temp):
            logger.warning(f"Temperature {temp}°C out of valid range [{min_temp}, {max_temp}]")
            return False
    
    # Validate humidity
    if 'humidity' in reading:
        humidity = reading['humidity']
        humidity_config = sensor_config.get('humidity', {})
        min_humidity = humidity_config.get('min_value', 0)
        max_humidity = humidity_config.get('max_value', 100)
        
        if not (min_humidity <= humidity <= max_humidity):
            logger.warning(f"Humidity {humidity}% out of valid range [{min_humidity}, {max_humidity}]")
            return False
    
    return True


def format_mqtt_topic(template: str, **kwargs) -> str:
    """
    Format MQTT topic template with variables
    
    Args:
        template: Topic template string (e.g., "field/{crop_type}/{device_id}/telemetry")
        **kwargs: Variables to substitute
        
    Returns:
        Formatted topic string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing key for topic formatting: {e}")
        raise


def calculate_pest_risk_level(probability: float, config: Dict[str, Any]) -> str:
    """
    Calculate pest risk level based on probability
    
    Args:
        probability: Pest growth probability (0-1)
        config: Configuration dictionary
        
    Returns:
        Risk level string: "Low", "Medium", or "High"
    """
    thresholds = config.get('alerts', {}).get('pest_risk', {})
    low_threshold = thresholds.get('low', 0.3)
    medium_threshold = thresholds.get('medium', 0.6)
    high_threshold = thresholds.get('high', 0.8)
    
    if probability < low_threshold:
        return "Low"
    elif probability < medium_threshold:
        return "Medium"
    elif probability < high_threshold:
        return "High"
    else:
        return "Critical"


def get_pest_info(pest_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get pest information from configuration
    
    Args:
        pest_type: Pest type identifier
        config: Configuration dictionary
        
    Returns:
        Pest information dictionary or None if not found
    """
    pests = config.get('pests', {})
    return pests.get(pest_type)


def get_crop_info(crop_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get crop information from configuration
    
    Args:
        crop_type: Crop type identifier
        config: Configuration dictionary
        
    Returns:
        Crop information dictionary or None if not found
    """
    crops = config.get('crops', {})
    return crops.get(crop_type)


def timestamp_to_string(timestamp: float) -> str:
    """
    Convert Unix timestamp to readable string
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted datetime string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def string_to_timestamp(date_string: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> float:
    """
    Convert date string to Unix timestamp
    
    Args:
        date_string: Date string
        format_str: Format of the date string
        
    Returns:
        Unix timestamp
    """
    dt = datetime.strptime(date_string, format_str)
    return dt.timestamp()


# Environment-specific paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
