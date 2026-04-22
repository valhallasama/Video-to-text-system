# Robot Pose Integration via MQTT

Guide for recording robot 6D coordinates (position + orientation + joint angles) alongside ultrasound images using MQTT for local network communication.

## Overview

**Goal:** Train a model that takes ultrasound images + instructions as input and outputs optimal robot target coordinates.

**Communication Method:** MQTT (Message Queuing Telemetry Transport)
- Lightweight publish/subscribe protocol
- Low latency (~1-5ms on local network)
- Ideal for real-time sensor data
- No HTTP overhead

## Architecture

```
┌─────────────────┐         MQTT Broker          ┌──────────────────┐
│   ROS PC        │         (Port 1883)           │  Dataset PC      │
│                 │                               │                  │
│ UR5e Robot      │   Publish: test/position     │  realtime_ocr.py │
│ ROS2 Controller │   ──────────────────────►    │  + GE Ultrasound │
│                 │                               │                  │
│ robot_position_ │   {x,y,z,quat,euler_angles}  │  robot_pose_     │
│ mqtt_publisher  │                               │  mqtt_client     │
└─────────────────┘                               └──────────────────┘
```

## Installation

### Step 1: Install MQTT Broker (Choose One PC)

You can run the MQTT broker on either PC. For simplicity, run it on the ROS PC.

**On ROS PC (or Dataset PC):**
```bash
# Install Mosquitto MQTT broker
sudo apt update
sudo apt install -y mosquitto mosquitto-clients

# Start broker
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Verify broker is running
sudo systemctl status mosquitto

# Optional: Configure for higher message rate
# Edit /etc/mosquitto/mosquitto.conf and add:
# max_inflight_messages 100
# max_queued_messages 1000
```

**Test broker:**
```bash
# Terminal 1: Subscribe to test topic
mosquitto_sub -h localhost -t test/topic

# Terminal 2: Publish to test topic
mosquitto_pub -h localhost -t test/topic -m "Hello MQTT"

# You should see "Hello MQTT" in Terminal 1
```

### Step 2: Install Python MQTT Client

**On both PCs:**
```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Install paho-mqtt
pip install paho-mqtt

# Or install from requirements.txt
pip install -r requirements.txt
```

## Usage

### On ROS PC: Start MQTT Publisher

**Your ROS2 publisher script reads `tcp_link` position from TF and publishes to MQTT.**

```bash
# Source ROS2 workspace
source ~/ros2_ws/install/setup.bash

# Run the ROS2 MQTT publisher node
ros2 run <your_package_name> robot_position_mqtt_publisher

# Or with custom parameters
ros2 run <your_package_name> robot_position_mqtt_publisher \
    --ros-args \
    -p mqtt_broker:=192.168.56.2 \
    -p mqtt_port:=1883 \
    -p mqtt_topic:=test/position \
    -p publish_rate:=30.0
```

**ROS2 Parameters:**
- `mqtt_broker`: MQTT broker hostname/IP (default: 192.168.56.2)
- `mqtt_port`: MQTT broker port (default: 1883)
- `mqtt_topic`: MQTT topic to publish to (default: test/position)
- `publish_rate`: Publishing rate in Hz (default: 30.0)

**TF Frames:**
- **Source frame**: `base_link`
- **Target frame**: `tcp_link` (tool center point)

**Recommended Publishing Rates:**
- **10 FPS video**: 20-30 Hz (2-3 poses per frame)
- **30 FPS video**: 60-90 Hz (2-3 poses per frame)
- **Higher is better** for frame synchronization, but diminishing returns above 3x frame rate

**Expected output:**
```
[INFO] [robot_position_mqtt_publisher]: Connected to MQTT broker at 192.168.56.2:1883
[INFO] [robot_position_mqtt_publisher]: MQTT connection successful
[INFO] [robot_position_mqtt_publisher]: Publishing robot position to MQTT topic: test/position
```

### On Dataset PC: Run Real-time OCR with MQTT

```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Run real-time OCR with MQTT robot pose recording
# Use the IP of the MQTT broker PC (e.g., ROS PC at 192.168.56.2)
python3 scripts/realtime_ocr.py \
    --device /dev/video16 \
    --save-dataset \
    --dataset-dir robot_training_dataset \
    --robot-pose-mqtt mqtt://192.168.56.2:1883
```

**Important:** 
- Replace `192.168.56.2` with the IP address of the PC running the MQTT broker
- The MQTT topic is automatically set to `test/position` (matching your ROS2 publisher)

**If broker is on the same PC:**
```bash
python3 scripts/realtime_ocr.py \
    --device /dev/video16 \
    --save-dataset \
    --dataset-dir robot_training_dataset \
    --robot-pose-mqtt mqtt://localhost:1883
```

## Dataset Output

### CSV Format (With Robot Pose via MQTT)
```csv
frame_number,instruction_text,quality_score,image_path,robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw,roll_deg,pitch_deg,yaw_deg
0,make slow circular sweeps until moving anatomy appears,45,frames/frame_000000.png,0.500000,0.300000,0.400000,0.000000,0.000000,0.707107,0.707107,0.000000,0.000000,90.000000
1,make slow circular sweeps until moving anatomy appears,47,frames/frame_000001.png,0.501000,0.301000,0.401000,0.000000,0.000000,0.707107,0.707107,0.000000,0.000000,90.000000
```

**Columns:**
- `robot_x, robot_y, robot_z`: Cartesian position of tcp_link (meters)
- `robot_qx, robot_qy, robot_qz, robot_qw`: Orientation quaternion
- `roll_deg, pitch_deg, yaw_deg`: Euler angles (degrees)

### Directory Structure
```
robot_training_dataset/
├── dataset.csv                    # Main dataset with robot poses
├── frames/                        # Ultrasound ROI images
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
└── original_frames/               # Full frames with ROI annotations
    ├── original_000000.png
    ├── original_000001.png
    └── ...
```

## Network Configuration

### Find IP Address

**On ROS PC (broker host):**
```bash
# Find IP address
ip addr show | grep "inet "

# Example output:
#     inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
```

Use this IP address (`192.168.1.100`) in the MQTT URL on the dataset PC.

### Test Connectivity

**From Dataset PC:**
```bash
# Ping ROS PC
ping 192.168.1.100

# Test MQTT connection
mosquitto_sub -h 192.168.1.100 -p 1883 -t robot/pose

# You should see robot pose messages if publisher is running
```

### Firewall Configuration

If connection fails, allow MQTT port on the broker PC:

```bash
# On broker PC (ROS PC)
sudo ufw allow 1883/tcp
sudo ufw reload
```

## Troubleshooting

### Issue: "Cannot connect to MQTT broker"

**Solutions:**

1. **Check broker is running:**
   ```bash
   sudo systemctl status mosquitto
   ```

2. **Check firewall:**
   ```bash
   sudo ufw status
   sudo ufw allow 1883/tcp
   ```

3. **Test with mosquitto_sub:**
   ```bash
   mosquitto_sub -h 192.168.1.100 -p 1883 -t robot/pose
   ```

4. **Check IP address:**
   ```bash
   ip addr show
   ```

### Issue: "No robot pose data in CSV"

**Possible causes:**

1. **Publisher not running:**
   - Start `robot_pose_mqtt_publisher.py` on ROS PC

2. **Wrong broker URL:**
   - Verify IP address: `mqtt://192.168.1.100:1883`
   - Use `localhost` if broker is on same PC

3. **ROS topics not publishing:**
   ```bash
   # On ROS PC
   rostopic list
   rostopic echo /joint_states -n 1
   ```

### Issue: "paho-mqtt not installed"

**Solution:**
```bash
pip install paho-mqtt
```

## Performance

### Latency
- **MQTT latency:** ~1-5ms on local network (wired)
- **MQTT latency:** ~5-20ms on WiFi
- **Publishing rate:** 30 Hz default (configurable)
- **Message size:** ~200 bytes per pose

### Bandwidth
- **30 Hz publishing:** ~6 KB/s
- **60 Hz publishing:** ~12 KB/s
- **Negligible network impact**

### Frame Synchronization

**How it works:**
1. MQTT publisher sends robot pose at 30 Hz (every 33ms)
2. Dataset PC captures video at 10 FPS (every 100ms)
3. When saving a frame, the MQTT client returns the **most recent** robot pose
4. Result: Each frame gets the robot pose from within ~33ms of capture time

**Publishing Rate Guidelines:**
- **Minimum**: Match frame rate (10 Hz for 10 FPS)
- **Recommended**: 2-3x frame rate (20-30 Hz for 10 FPS)
- **Maximum**: Limited by ROS update rate (~100-200 Hz typical)

**Why higher is better:**
- Reduces maximum pose age (33ms @ 30Hz vs 100ms @ 10Hz)
- Better temporal accuracy for fast robot movements
- Minimal overhead (MQTT is very lightweight)

### Comparison with HTTP

| Method | Latency | Bandwidth | Complexity |
|--------|---------|-----------|------------|
| **MQTT** | 1-5ms | 2 KB/s | Low |
| HTTP REST | 10-30ms | 5 KB/s | Medium |

**MQTT is better for:**
- Real-time data streaming
- Low latency requirements
- Local network communication
- Publish/subscribe pattern

## Advanced Configuration

### Custom MQTT Topic

**Publisher:**
```python
# In robot_pose_mqtt_publisher.py
# Change topic name
self.mqtt_client.publish('my_robot/pose', pose_json, qos=0)
```

**Subscriber:**
```python
# In robot_pose_mqtt_client.py
# Change topic in __init__
self.topic = "my_robot/pose"
```

### QoS Levels

MQTT supports 3 Quality of Service levels:

- **QoS 0** (default): At most once delivery (fastest, may lose messages)
- **QoS 1**: At least once delivery (slower, guarantees delivery)
- **QoS 2**: Exactly once delivery (slowest, no duplicates)

For robot pose, **QoS 0** is recommended (real-time data, latest value matters most).

### Authentication

To secure MQTT broker:

**On broker PC:**
```bash
# Create password file
sudo mosquitto_passwd -c /etc/mosquitto/passwd username

# Edit mosquitto config
sudo nano /etc/mosquitto/mosquitto.conf

# Add:
allow_anonymous false
password_file /etc/mosquitto/passwd

# Restart broker
sudo systemctl restart mosquitto
```

**In publisher/subscriber:**
```python
self.mqtt_client.username_pw_set("username", "password")
```

## Summary

✅ **MQTT Setup Complete:**
- MQTT broker running on ROS PC (or Dataset PC)
- Publisher sending robot pose at 10 Hz
- Subscriber receiving pose in real-time
- Dataset includes synchronized robot coordinates

✅ **Advantages over HTTP:**
- Lower latency (1-5ms vs 10-30ms)
- Simpler setup (no web server needed)
- Real-time push updates (no polling)
- Lightweight protocol

✅ **Ready for Training:**
- Input: Ultrasound image + instruction text
- Output: Optimal robot pose (x, y, z, qx, qy, qz, qw)
- Model learns visual feedback → robot control mapping

For installation instructions, see [INSTALL.md](INSTALL.md).

For system overview, see [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).
