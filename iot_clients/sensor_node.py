"""
IoT Sensor Node Simulator

This module simulates field sensors (temperature and humidity) that would be deployed
in agricultural plots. In production, this would interface with real DHT22/BME280 sensors.

The simulator generates realistic environmental data with:
- Diurnal (daily) temperature and humidity cycles
- Seasonal variations
- Random noise to simulate real-world conditions
- Correlation between temperature and humidity
- Pest-favorable conditions based on crop and pest type
"""

import json
import time
import random
import math
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging, format_mqtt_topic


class SensorNode:
    """
    Simulates an IoT sensor node deployed in an agricultural field
    
    In production, this would:
    - Interface with DHT22 (temperature/humidity sensor)
    - Optionally read soil moisture and leaf wetness sensors
    - Handle sensor calibration and error detection
    - Implement local buffering for offline operation
    """
    
    def __init__(
        self,
        device_id: str,
        crop_type: str,
        pest_type: Optional[str] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize sensor node
        
        Args:
            device_id: Unique identifier for this sensor node
            crop_type: Type of crop in this field (e.g., "tomato", "rice")
            pest_type: Target pest to simulate growth conditions (optional)
            config_path: Path to configuration file
        """
        self.device_id = device_id
        self.crop_type = crop_type
        self.pest_type = pest_type
        
        # Load configuration
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        # MQTT configuration
        mqtt_config = self.config['mqtt']
        self.broker_host = mqtt_config['broker_host']
        self.broker_port = mqtt_config['broker_port']
        self.keepalive = mqtt_config['keepalive']
        self.qos = mqtt_config['qos']
        
        # Topic configuration
        self.telemetry_topic = format_mqtt_topic(
            mqtt_config['topics']['telemetry'],
            crop_type=crop_type,
            device_id=device_id
        )
        
        # Sensor configuration
        sensor_config = self.config['sensors']
        self.reading_interval = sensor_config['reading_interval']
        
        # Get pest information for simulation
        self.pest_info = None
        if pest_type:
            self.pest_info = self.config['pests'].get(pest_type)
            if self.pest_info:
                logger.info(f"Simulating {pest_type} growth conditions")
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"sensor_{device_id}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        # Simulation state
        self.start_time = time.time()
        self.running = False
        
        logger.info(f"Initialized sensor node {device_id} for {crop_type} crop")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker. Return code: {rc}")
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logger.debug(f"Message {mid} published successfully")
    
    def _generate_realistic_reading(self) -> Dict[str, Any]:
        """
        Generate realistic sensor readings with temporal patterns
        
        Returns:
            Dictionary containing temperature, humidity, and metadata
        """
        # Time-based simulation
        elapsed_hours = (time.time() - self.start_time) / 3600
        hour_of_day = (elapsed_hours % 24)
        day_of_year = (elapsed_hours / 24) % 365
        
        # Base temperature with diurnal and seasonal cycles
        # Diurnal: lower at night (4 AM), higher in afternoon (2 PM)
        diurnal_temp = 10 * math.sin((hour_of_day - 6) * math.pi / 12)
        
        # Seasonal: lower in winter, higher in summer (simplified for demo)
        seasonal_temp = 5 * math.sin((day_of_year - 80) * 2 * math.pi / 365)
        
        # Base temperature (average for region)
        base_temp = 25.0
        
        # If simulating pest growth, bias toward optimal conditions
        if self.pest_info:
            optimal_temp = sum(self.pest_info['optimal_temp_range']) / 2
            # Gradually drift toward optimal temperature
            drift = (optimal_temp - base_temp) * 0.3
            base_temp += drift
        
        # Add random noise
        noise = random.gauss(0, 1.5)
        
        temperature = base_temp + diurnal_temp + seasonal_temp + noise
        temperature = round(max(10, min(40, temperature)), 2)  # Clamp to reasonable range
        
        # Humidity (inversely correlated with temperature)
        # Higher humidity at night and in cooler weather
        base_humidity = 70.0
        humidity_temp_correlation = -1.5 * (temperature - 25)  # Inverse relationship
        diurnal_humidity = 15 * math.sin((hour_of_day - 18) * math.pi / 12)  # Peak at night
        
        if self.pest_info:
            optimal_humidity = sum(self.pest_info['optimal_humidity_range']) / 2
            drift_humidity = (optimal_humidity - base_humidity) * 0.3
            base_humidity += drift_humidity
        
        humidity_noise = random.gauss(0, 3)
        humidity = base_humidity + humidity_temp_correlation + diurnal_humidity + humidity_noise
        humidity = round(max(30, min(95, humidity)), 2)  # Clamp to reasonable range
        
        return {
            "device_id": self.device_id,
            "crop_type": self.crop_type,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "temperature": temperature,
            "humidity": humidity,
            "unit_temp": "celsius",
            "unit_humidity": "percent"
        }
    
    def publish_reading(self) -> None:
        """Generate and publish a sensor reading to MQTT"""
        try:
            reading = self._generate_realistic_reading()
            payload = json.dumps(reading)
            
            result = self.client.publish(
                self.telemetry_topic,
                payload,
                qos=self.qos
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(
                    f"Published reading: {reading['temperature']}°C, "
                    f"{reading['humidity']}% to {self.telemetry_topic}"
                )
            else:
                logger.error(f"Failed to publish reading. Error code: {result.rc}")
                
        except Exception as e:
            logger.error(f"Error publishing reading: {e}")
    
    def connect(self) -> None:
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            time.sleep(1)  # Wait for connection
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def run(self, duration_seconds: Optional[int] = None) -> None:
        """
        Run the sensor node, continuously publishing readings
        
        Args:
            duration_seconds: How long to run (None for infinite)
        """
        self.running = True
        self.connect()
        
        logger.info(f"Starting sensor node {self.device_id}. Publishing every {self.reading_interval}s")
        
        start = time.time()
        
        try:
            while self.running:
                self.publish_reading()
                
                # Check duration
                if duration_seconds and (time.time() - start) >= duration_seconds:
                    logger.info(f"Reached duration limit of {duration_seconds}s")
                    break
                
                # Wait for next reading interval
                time.sleep(self.reading_interval)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the sensor node"""
        self.running = False
        self.disconnect()
        logger.info("Sensor node stopped")


def main():
    """
    Main function for standalone sensor node execution
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="IoT Sensor Node Simulator")
    parser.add_argument("--device-id", required=True, help="Unique device identifier")
    parser.add_argument("--crop", required=True, help="Crop type (e.g., tomato, rice, wheat)")
    parser.add_argument("--pest", help="Pest type to simulate (e.g., aphids, whitefly)")
    parser.add_argument("--duration", type=int, help="Duration to run in seconds (default: infinite)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    sensor = SensorNode(
        device_id=args.device_id,
        crop_type=args.crop,
        pest_type=args.pest,
        config_path=args.config
    )
    
    sensor.run(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
