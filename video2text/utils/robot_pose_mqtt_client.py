"""
Robot Pose MQTT Client

Subscribes to robot pose from MQTT broker running on ROS PC.
Used by dataset generation to record robot coordinates for each frame.
"""

import json
import time
import threading
from typing import Optional, Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("WARNING: paho-mqtt not installed. Install with: pip install paho-mqtt")


class RobotPoseMQTTClient:
    """Client to receive robot pose from MQTT broker."""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883, 
                 topic: str = "robot/pose"):
        """
        Initialize robot pose MQTT client.
        
        Args:
            broker_host: MQTT broker hostname/IP
            broker_port: MQTT broker port (default: 1883)
            topic: MQTT topic to subscribe to (default: robot/pose)
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt not installed. Install with: pip install paho-mqtt")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.last_pose = None
        self.connection_ok = False
        self.lock = threading.Lock()
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id="robot_pose_subscriber")
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        self.mqtt_client.on_message = self._on_message
        
        # Connect to broker
        self._connect()
    
    def _connect(self):
        """Connect to MQTT broker."""
        try:
            self.mqtt_client.connect(self.broker_host, self.broker_port, 60)
            self.mqtt_client.loop_start()
            
            # Wait a bit for connection
            time.sleep(0.5)
            
            if self.connection_ok:
                print(f"✅ Robot pose MQTT client connected: {self.broker_host}:{self.broker_port}")
            else:
                print(f"⚠️  Cannot connect to MQTT broker: {self.broker_host}:{self.broker_port}")
                print(f"   Robot coordinates will not be recorded.")
                
        except Exception as e:
            self.connection_ok = False
            print(f"⚠️  Cannot connect to MQTT broker: {self.broker_host}:{self.broker_port}")
            print(f"   Error: {e}")
            print(f"   Robot coordinates will not be recorded.")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects."""
        if rc == 0:
            self.connection_ok = True
            # Subscribe to robot pose topic
            self.mqtt_client.subscribe(self.topic)
        else:
            self.connection_ok = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects."""
        if rc != 0:
            self.connection_ok = False
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        try:
            # Parse JSON message
            pose_data = json.loads(msg.payload.decode('utf-8'))
            
            # Update last pose with thread safety
            with self.lock:
                self.last_pose = pose_data
                
        except json.JSONDecodeError:
            pass  # Ignore malformed messages
        except Exception:
            pass  # Ignore other errors
    
    def get_pose(self) -> Optional[Dict[str, Any]]:
        """
        Get current robot pose.
        
        Returns:
            Dictionary with robot pose data, or None if unavailable:
            {
                'timestamp': float,
                'position': {'x': float, 'y': float, 'z': float},
                'orientation': {'x': float, 'y': float, 'z': float, 'w': float},
                'joint_positions': [float, ...]
            }
        """
        with self.lock:
            return self.last_pose
    
    def get_pose_string(self) -> str:
        """
        Get robot pose as CSV-compatible string.
        
        Returns:
            String format: "x,y,z,qx,qy,qz,qw,roll,pitch,yaw"
            or empty string if pose unavailable
        """
        pose = self.get_pose()
        
        if pose is None:
            return ""
        
        try:
            pos = pose['position']
            ori = pose['orientation']
            euler = pose.get('euler_angles', {})
            
            # Format: x,y,z,qx,qy,qz,qw,roll,pitch,yaw
            parts = [
                f"{pos['x']:.6f}",
                f"{pos['y']:.6f}",
                f"{pos['z']:.6f}",
                f"{ori['quat_x']:.6f}",
                f"{ori['quat_y']:.6f}",
                f"{ori['quat_z']:.6f}",
                f"{ori['quat_w']:.6f}",
            ]
            
            # Add euler angles if available
            if euler:
                parts.append(f"{euler.get('roll_deg', 0.0):.6f}")
                parts.append(f"{euler.get('pitch_deg', 0.0):.6f}")
                parts.append(f"{euler.get('yaw_deg', 0.0):.6f}")
            else:
                # Fallback if euler angles not provided
                parts.extend(["0.000000", "0.000000", "0.000000"])
            
            return ",".join(parts)
            
        except (KeyError, TypeError):
            return ""
    
    def is_connected(self) -> bool:
        """Check if connected to MQTT broker."""
        return self.connection_ok
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
