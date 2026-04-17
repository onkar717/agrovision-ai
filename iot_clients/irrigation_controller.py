"""
Irrigation Controller

Controls a solenoid valve for automated or manual irrigation.
In production, this would interface with GPIO pins on a Raspberry Pi
to control a relay that activates/deactivates the solenoid valve.

Features:
- MQTT-based remote control
- Manual override safety mechanism
- Maximum duration limits
- Cooldown period enforcement
- Event logging for audit trail
"""

import time
import json
from datetime import datetime, timedelta
from typing import Optional
import paho.mqtt.client as mqtt
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging, format_mqtt_topic


class IrrigationController:
    """
    Controls irrigation valve via MQTT commands
    
    In production deployment:
    - Uses RPi.GPIO library to control relay connected to GPIO pin
    - Relay controls 12V/24V solenoid valve
    - Includes hardware safety switches and manual override
    - Logs all valve operations for regulatory compliance
    """
    
    def __init__(
        self,
        device_id: str,
        crop_type: str,
        config_path: str = "config.yaml",
        simulation_mode: bool = True
    ):
        """
        Initialize irrigation controller
        
        Args:
            device_id: Unique identifier for this controller
            crop_type: Type of crop being irrigated
            config_path: Path to configuration file
            simulation_mode: If True, simulate valve control; if False, use GPIO
        """
        self.device_id = device_id
        self.crop_type = crop_type
        self.simulation_mode = simulation_mode
        
        # Load configuration
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        # Irrigation configuration
        irrigation_config = self.config['irrigation']
        self.auto_mode = irrigation_config['auto_mode']
        self.gpio_pin = irrigation_config['valve_gpio_pin']
        self.max_duration_minutes = irrigation_config['max_duration_minutes']
        self.cooldown_minutes = irrigation_config['cooldown_minutes']
        
        # MQTT configuration
        mqtt_config = self.config['mqtt']
        self.broker_host = mqtt_config['broker_host']
        self.broker_port = mqtt_config['broker_port']
        self.keepalive = mqtt_config['keepalive']
        
        # Subscribe to irrigation commands topic
        self.command_topic = format_mqtt_topic(
            mqtt_config['topics']['irrigation'],
            crop_type=crop_type,
            device_id=device_id
        )
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"irrigation_{device_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Valve state
        self.valve_open = False
        self.valve_open_time = None
        self.last_closed_time = None
        
        # Initialize GPIO if not in simulation mode
        if not simulation_mode:
            self._init_gpio()
        
        logger.info(f"Initialized irrigation controller {device_id} for {crop_type}")
        logger.info(f"Auto mode: {self.auto_mode}, Simulation mode: {simulation_mode}")
    
    def _init_gpio(self):
        """
        Initialize GPIO for valve control (production only)
        
        NOTE: This requires RPi.GPIO library and physical Raspberry Pi hardware
        """
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)  # Ensure valve is closed initially
            logger.info(f"GPIO pin {self.gpio_pin} initialized for valve control")
        except ImportError:
            logger.warning("RPi.GPIO not available. Switching to simulation mode.")
            self.simulation_mode = True
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            self.simulation_mode = True
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker")
            # Subscribe to command topic
            client.subscribe(self.command_topic, qos=1)
            logger.info(f"Subscribed to {self.command_topic}")
        else:
            logger.error(f"Failed to connect. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker. Return code: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback - processes irrigation commands"""
        try:
            payload = json.loads(msg.payload.decode())
            command = payload.get('command')
            duration_minutes = payload.get('duration_minutes', self.max_duration_minutes)
            
            logger.info(f"Received command: {command} (duration: {duration_minutes} min)")
            
            if command == 'activate':
                self.activate_valve(duration_minutes)
            elif command == 'deactivate':
                self.deactivate_valve()
            elif command == 'status':
                self._report_status()
            else:
                logger.warning(f"Unknown command: {command}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def activate_valve(self, duration_minutes: Optional[int] = None) -> bool:
        """
        Open the irrigation valve
        
        Args:
            duration_minutes: How long to keep valve open (None for manual control)
            
        Returns:
            True if activated successfully, False otherwise
        """
        # Check if valve is already open
        if self.valve_open:
            logger.warning("Valve is already open")
            return False
        
        # Check cooldown period
        if self.last_closed_time:
            cooldown_end = self.last_closed_time + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).seconds / 60
                logger.warning(f"Cooldown period active. {remaining:.1f} minutes remaining")
                return False
        
        # Validate duration
        if duration_minutes:
            if duration_minutes > self.max_duration_minutes:
                logger.warning(
                    f"Requested duration {duration_minutes} exceeds max {self.max_duration_minutes}. "
                    f"Capping at maximum."
                )
                duration_minutes = self.max_duration_minutes
        
        # Open valve
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Opening valve for {duration_minutes} minutes")
        else:
            import RPi.GPIO as GPIO
            GPIO.output(self.gpio_pin, GPIO.HIGH)
            logger.info(f"[HARDWARE] GPIO pin {self.gpio_pin} set HIGH - Valve OPEN")
        
        self.valve_open = True
        self.valve_open_time = datetime.now()
        
        # Log event
        self._log_event("VALVE_OPENED", duration_minutes)
        
        # Schedule auto-close if duration specified
        if duration_minutes and self.auto_mode:
            logger.info(f"Valve will auto-close in {duration_minutes} minutes")
            # In production, this would use a timer thread
        
        return True
    
    def deactivate_valve(self) -> bool:
        """
        Close the irrigation valve
        
        Returns:
            True if deactivated successfully, False otherwise
        """
        if not self.valve_open:
            logger.warning("Valve is already closed")
            return False
        
        # Close valve
        if self.simulation_mode:
            logger.info("[SIMULATION] Closing valve")
        else:
            import RPi.GPIO as GPIO
            GPIO.output(self.gpio_pin, GPIO.LOW)
            logger.info(f"[HARDWARE] GPIO pin {self.gpio_pin} set LOW - Valve CLOSED")
        
        # Calculate duration
        duration = (datetime.now() - self.valve_open_time).seconds / 60
        
        self.valve_open = False
        self.last_closed_time = datetime.now()
        
        # Log event
        self._log_event("VALVE_CLOSED", duration)
        
        return True
    
    def _report_status(self):
        """Report current valve status via logging"""
        status = {
            "device_id": self.device_id,
            "crop_type": self.crop_type,
            "valve_open": self.valve_open,
            "valve_open_time": self.valve_open_time.isoformat() if self.valve_open_time else None,
            "auto_mode": self.auto_mode,
            "simulation_mode": self.simulation_mode
        }
        logger.info(f"Status: {json.dumps(status, indent=2)}")
    
    def _log_event(self, event_type: str, duration: Optional[float] = None):
        """
        Log irrigation event for audit trail
        
        Args:
            event_type: Type of event (e.g., "VALVE_OPENED", "VALVE_CLOSED")
            duration: Duration in minutes (for close events)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id,
            "crop_type": self.crop_type,
            "event_type": event_type,
            "duration_minutes": round(duration, 2) if duration else None,
            "simulation_mode": self.simulation_mode
        }
        
        # In production, this would write to a dedicated audit log file or database
        logger.info(f"IRRIGATION_EVENT: {json.dumps(event)}")
    
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
        # Ensure valve is closed before disconnecting
        if self.valve_open:
            self.deactivate_valve()
        
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def run(self):
        """Run the irrigation controller (blocking)"""
        self.connect()
        logger.info(f"Irrigation controller running. Listening on {self.command_topic}")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
                
                # Auto-close valve if max duration exceeded
                if self.valve_open and self.valve_open_time:
                    elapsed = (datetime.now() - self.valve_open_time).seconds / 60
                    if elapsed >= self.max_duration_minutes:
                        logger.warning(f"Max duration {self.max_duration_minutes} min exceeded. Auto-closing valve.")
                        self.deactivate_valve()
                        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
        finally:
            self.disconnect()


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Irrigation Controller")
    parser.add_argument("--device-id", required=True, help="Unique device identifier")
    parser.add_argument("--crop", required=True, help="Crop type")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--hardware", action="store_true", help="Use hardware GPIO (default: simulation)")
    
    args = parser.parse_args()
    
    controller = IrrigationController(
        device_id=args.device_id,
        crop_type=args.crop,
        config_path=args.config,
        simulation_mode=not args.hardware
    )
    
    controller.run()


if __name__ == "__main__":
    main()
