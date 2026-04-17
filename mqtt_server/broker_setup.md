# Setting up MQTT Broker (Mosquitto)

This document provides installation and configuration instructions for the Mosquitto MQTT broker, which serves as the communication hub for the pest detection system.

## Installing Mosquitto

### macOS
```bash
brew install mosquitto

# Start broker
brew services start mosquitto

# Or run manually
/usr/local/opt/mosquitto/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# Start broker
sudo systemctl start mosquitto
sudo systemctl enable mosquitto  # Auto-start on boot
```

### Raspberry Pi (Production Deployment)
```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

## Basic Configuration

Default configuration file locations:
- **macOS**: `/usr/local/etc/mosquitto/mosquitto.conf`
- **Linux**: `/etc/mosquitto/mosquitto.conf`

### Minimal Configuration

Create or edit the configuration file:

```conf
# mosquitto.conf - Basic configuration for pest detection system

# Listener
listener 1883
protocol mqtt

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_dest stdout
log_type all
connection_messages true
timestamp_format %Y-%m-%dT%H:%M:%S

# Persistence (important for message retention)
persistence true
persistence_location /var/lib/mosquitto/

# Allow anonymous connections (disable in production)
allow_anonymous true

# Message size limits
message_size_limit 10240  # 10KB

# QoS settings
max_queued_messages 1000
```

### Production Configuration with Authentication

For production deployments, enable authentication:

```conf
# mosquitto.conf - Production configuration

listener 1883
protocol mqtt

# Disable anonymous access
allow_anonymous false

# Password file
password_file /etc/mosquitto/passwd

# ACL (Access Control List)
acl_file /etc/mosquitto/acl

# TLS/SSL (optional but recommended)
# listener 8883
# cafile /etc/mosquitto/ca_certificates/ca.crt
# certfile /etc/mosquitto/certs/server.crt
# keyfile /etc/mosquitto/certs/server.key

# Persistence
persistence true
persistence_location /var/lib/mosquitto/

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
log_type information
```

### Creating User Passwords

```bash
# Create password file and add user
sudo mosquitto_passwd -c /etc/mosquitto/passwd pest_system

# Add additional users
sudo mosquitto_passwd /etc/mosquitto/passwd sensor_node_01
sudo mosquitto_passwd /etc/mosquitto/passwd dashboard_user
```

### Access Control List (ACL)

Create `/etc/mosquitto/acl`:

```conf
# ACL for pest detection system

# Admin user - full access
user pest_system
topic readwrite #

# Sensor nodes - can only publish telemetry
user sensor_node_01
topic write field/+/sensor_node_01/telemetry

# Dashboard - read-only on telemetry, write on irrigation commands
user dashboard_user
topic read field/#
topic write field/+/+/irrigation
```

## Testing the Broker

### Start the broker
```bash
# macOS
mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf -v

# Linux
sudo mosquitto -c /etc/mosquitto/mosquitto.conf -v
```

### Test with mosquitto_sub and mosquitto_pub

Terminal 1 (Subscriber):
```bash
mosquitto_sub -h localhost -t "field/+/+/telemetry" -v
```

Terminal 2 (Publisher):
```bash
mosquitto_pub -h localhost -t "field/tomato/sensor_01/telemetry" \
  -m '{"device_id":"sensor_01","crop_type":"tomato","temperature":25.5,"humidity":70.2,"timestamp":1702540800}'
```

You should see the message appear in Terminal 1.

## Monitoring Broker

### View active connections
```bash
# Using mosquitto_sub to monitor system topics
mosquitto_sub -h localhost -t '$SYS/#' -v
```

### View logs
```bash
# macOS
tail -f /usr/local/var/log/mosquitto/mosquitto.log

# Linux
tail -f /var/log/mosquitto/mosquitto.log
```

## Network Configuration for Field Deployment

For multi-device deployments across a local network:

1. **Find broker IP address:**
   ```bash
   # macOS
   ifconfig | grep "inet "
   
   # Linux
   hostname -I
   ```

2. **Update config.yaml** on all sensor nodes:
   ```yaml
   mqtt:
     broker_host: "192.168.1.100"  # Replace with actual broker IP
     broker_port: 1883
   ```

3. **Firewall Configuration (if needed):**
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 1883/tcp
   
   # CentOS/RHEL
   sudo firewall-cmd --permanent --add-port=1883/tcp
   sudo firewall-cmd --reload
   ```

## Low-Connectivity Rural Deployment

For deployments with intermittent internet:

1. **Enable message persistence** (already in config above)
2. **Increase queue sizes:**
   ```conf
   max_queued_messages 10000
   max_inflight_messages 100
   ```

3. **Set up QoS levels** in client code (already implemented):
   - QoS 1: At least once delivery (recommended for sensor data)
   - QoS 2: Exactly once delivery (use for critical irrigation commands)

4. **Client-side buffering:** Sensor nodes buffer messages locally when broker is unavailable (implemented in `sensor_node.py`)

## Troubleshooting

### Broker won't start
```bash
# Check for port conflicts
sudo lsof -i :1883

# Check logs
tail -100 /var/log/mosquitto/mosquitto.log
```

### Clients can't connect
```bash
# Test broker is listening
telnet localhost 1883

# Check firewall
sudo ufw status  # Ubuntu
```

### Messages not being delivered
- Verify QoS settings match between publisher and subscriber
- Check ACL permissions
- Verify topic names match (wildcards: `+` for single level, `#` for multi-level)

## Security Recommendations for Production

1. **Always use authentication** (disable `allow_anonymous`)
2. **Implement ACLs** to restrict topic access
3. **Use TLS/SSL** for encrypted communication (especially over internet)
4. **Regular password rotation** for all users
5. **Separate VLANs** for IoT devices and main network
6. **Monitor unusual connection patterns** via logs
7. **Limit message sizes** to prevent DOS attacks

## Bridge Configuration (Optional)

For syncing local edge broker with cloud broker:

```conf
# Bridge to cloud MQTT broker
connection bridge-to-cloud
address cloud.mqtt.server:1883
topic field/# out 0
bridge_attempt_unsubscribe true
```

This allows local operation with periodic cloud sync when internet available.
