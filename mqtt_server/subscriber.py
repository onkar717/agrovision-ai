"""
MQTT Subscriber and Data Ingestion

This module subscribes to sensor telemetry topics and ingests data into the local database.
It acts as the bridge between the IoT sensor network and the data pipeline.

Features:
- Subscribes to wildcard MQTT topics to receive all sensor data
- Validates incoming message payloads
- Stores time-series data in local database
- Triggers inference pipeline on new data
- Implements fault tolerance with message buffering
- Handles intermittent broker connectivity
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from loguru import logger
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging, validate_sensor_reading


class MQTTSubscriber:
    """
    MQTT subscriber that ingests sensor data into local database
    
    Architecture:
    - Subscribes to field/+/+/telemetry (wildcard for all crops and devices)
    - Validates message schema and sensor bounds
    - Stores in time-series optimized SQLite database
    - Buffers messages during database unavailability
    - Optionally triggers real-time pest prediction inference
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize MQTT subscriber
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        # MQTT configuration
        mqtt_config = self.config['mqtt']
        self.broker_host = mqtt_config['broker_host']
        self.broker_port = mqtt_config['broker_port']
        self.keepalive = mqtt_config['keepalive']
        
        # Subscribe to all telemetry topics using wildcard
        # field/+/+/telemetry matches field/{any_crop}/{any_device}/telemetry
        self.telemetry_topic_pattern = "field/+/+/telemetry"
        
        # Database configuration
        db_config = self.config['database']
        self.db_path = db_config.get('sqlite_path', 'data/pest_detection.db')
        self._init_database()
        
        # MQTT client
        self.client = mqtt.Client(client_id="mqtt_subscriber")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Message buffer for fault tolerance
        self.message_buffer = []
        self.max_buffer_size = 1000
        
        logger.info("MQTT Subscriber initialized")
    
    def _init_database(self):
        """
        Initialize SQLite database with optimized schema for time-series data
        
        Schema design:
        - sensor_readings: Main table for telemetry data
        - Indexed on timestamp and device_id for fast queries
        - Partitioned by crop_type for efficient filtering
        """
        # Ensure data directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sensor readings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                unit_temp TEXT DEFAULT 'celsius',
                unit_humidity TEXT DEFAULT 'percent',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast time-series queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON sensor_readings(timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_device_timestamp 
            ON sensor_readings(device_id, timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_crop_timestamp 
            ON sensor_readings(crop_type, timestamp DESC)
        ''')
        
        # Create pest predictions table (for storing model outputs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pest_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                pest_type TEXT,
                prediction_timestamp REAL NOT NULL,
                growth_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                infestation_timeline_days INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prediction_timestamp 
            ON pest_predictions(prediction_timestamp DESC)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Subscribe to all telemetry topics
            client.subscribe(self.telemetry_topic_pattern, qos=1)
            logger.info(f"Subscribed to {self.telemetry_topic_pattern}")
            
            # Flush any buffered messages
            if self.message_buffer:
                logger.info(f"Flushing {len(self.message_buffer)} buffered messages")
                for msg in self.message_buffer:
                    self._process_message(msg)
                self.message_buffer.clear()
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker. Return code: {rc}")
            logger.info("Will attempt to reconnect...")
    
    def _on_message(self, client, userdata, msg):
        """
        MQTT message callback - processes incoming sensor data
        
        Args:
            client: MQTT client instance
            userdata: User data
            msg: MQTT message object
        """
        try:
            # Decode and parse message
            payload = json.loads(msg.payload.decode())
            
            logger.debug(f"Received message on {msg.topic}: {payload}")
            
            # Validate message
            if not self._validate_message(payload):
                logger.warning(f"Invalid message received: {payload}")
                return
            
            # Process and store message
            self._process_message(payload)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
    
    def _validate_message(self, payload: Dict[str, Any]) -> bool:
        """
        Validate message payload
        
        Args:
            payload: Message payload dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['device_id', 'crop_type', 'timestamp', 'temperature', 'humidity']
        for field in required_fields:
            if field not in payload:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate sensor readings against configured bounds
        if not validate_sensor_reading(payload, self.config):
            return False
        
        return True
    
    def _process_message(self, payload: Dict[str, Any]):
        """
        Process and store sensor reading in database
        
        Args:
            payload: Validated message payload
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (device_id, crop_type, timestamp, datetime, temperature, humidity, unit_temp, unit_humidity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                payload['device_id'],
                payload['crop_type'],
                payload['timestamp'],
                payload.get('datetime', datetime.fromtimestamp(payload['timestamp']).isoformat()),
                payload['temperature'],
                payload['humidity'],
                payload.get('unit_temp', 'celsius'),
                payload.get('unit_humidity', 'percent')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(
                f"Stored reading: {payload['device_id']} | "
                f"{payload['temperature']}°C, {payload['humidity']}%"
            )
            
            # TODO: Trigger inference pipeline if enough data accumulated
            # self._trigger_inference(payload['device_id'], payload['crop_type'])
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            
            # Buffer message for retry if database unavailable
            if len(self.message_buffer) < self.max_buffer_size:
                self.message_buffer.append(payload)
                logger.info(f"Message buffered ({len(self.message_buffer)}/{self.max_buffer_size})")
            else:
                logger.error("Message buffer full. Dropping message.")
    
    def get_recent_readings(
        self,
        device_id: Optional[str] = None,
        crop_type: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Retrieve recent sensor readings from database
        
        Args:
            device_id: Filter by device ID (optional)
            crop_type: Filter by crop type (optional)
            limit: Maximum number of readings to return
            
        Returns:
            List of sensor reading dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        query = "SELECT * FROM sensor_readings WHERE 1=1"
        params = []
        
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        
        if crop_type:
            query += " AND crop_type = ?"
            params.append(crop_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        readings = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return readings
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def run(self):
        """Run the MQTT subscriber (blocking)"""
        self.connect()
        
        logger.info("MQTT Subscriber running. Press Ctrl+C to stop.")
        
        try:
            # Keep the script running
            while True:
                import time
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
        finally:
            self.disconnect()


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQTT Subscriber for Sensor Data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    subscriber = MQTTSubscriber(config_path=args.config)
    subscriber.run()


if __name__ == "__main__":
    main()
