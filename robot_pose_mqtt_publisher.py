#!/usr/bin/env python3
"""
Robot Pose MQTT Publisher (Run on ROS PC)

This publisher sends the current robot pose via MQTT broker.
The dataset generation PC subscribes to receive real-time robot coordinates.

Usage on ROS PC:
    python3 robot_pose_mqtt_publisher.py --broker localhost --port 1883

The publisher will:
- Subscribe to ROS topics (/joint_states, /tf)
- Publish robot pose to MQTT topic: robot/pose
    
Message format (JSON):
    {
        "timestamp": 1234567890.123,
        "position": {"x": 0.5, "y": 0.3, "z": 0.4},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "joint_positions": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    }
"""

import argparse
import json
import time
import signal
import sys

# MQTT client
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("ERROR: paho-mqtt not installed. Install with: pip install paho-mqtt")
    sys.exit(1)

# Try to import ROS (only needed on ROS PC)
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from tf2_ros import Buffer, TransformListener
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("WARNING: ROS not available. Running in mock mode.")


class RobotPoseMQTTPublisher:
    """MQTT publisher that sends robot pose from ROS topics."""
    
    def __init__(self, broker_host='localhost', broker_port=1883, 
                 robot_frame='tool0', base_frame='base_link',
                 publish_rate=10.0):
        """
        Initialize MQTT publisher.
        
        Args:
            broker_host: MQTT broker hostname/IP
            broker_port: MQTT broker port (default: 1883)
            robot_frame: Robot end-effector frame
            base_frame: Robot base frame
            publish_rate: Publishing rate in Hz (default: 10.0)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.robot_frame = robot_frame
        self.base_frame = base_frame
        self.publish_rate = publish_rate
        self.running = True
        
        # Current robot state
        self.current_pose = {
            'timestamp': time.time(),
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'joint_positions': [0.0] * 6
        }
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id="robot_pose_publisher")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        # Initialize ROS if available
        if ROS_AVAILABLE:
            self._init_ros()
        else:
            print("⚠️  ROS not available - will publish mock data")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects."""
        if rc == 0:
            print(f"✅ Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
        else:
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects."""
        if rc != 0:
            print(f"⚠️  Unexpected MQTT disconnection. Return code: {rc}")
    
    def _init_ros(self):
        """Initialize ROS node and subscribers."""
        try:
            rospy.init_node('robot_pose_mqtt_publisher', anonymous=True)
            
            # Subscribe to joint states
            rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
            
            # TF listener for end-effector pose
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)
            
            print(f"✅ ROS initialized")
            print(f"   Listening to /joint_states")
            print(f"   TF frames: {self.base_frame} -> {self.robot_frame}")
            
        except Exception as e:
            print(f"❌ ROS initialization failed: {e}")
    
    def _joint_state_callback(self, msg):
        """Callback for joint state updates."""
        if len(msg.position) >= 6:
            self.current_pose['joint_positions'] = list(msg.position[:6])
            self.current_pose['timestamp'] = time.time()
    
    def _update_cartesian_pose(self):
        """Update Cartesian pose from TF."""
        if not ROS_AVAILABLE:
            return
        
        try:
            # Get transform from base to tool
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            
            # Update position
            self.current_pose['position'] = {
                'x': transform.transform.translation.x,
                'y': transform.transform.translation.y,
                'z': transform.transform.translation.z
            }
            
            # Update orientation (quaternion)
            self.current_pose['orientation'] = {
                'x': transform.transform.rotation.x,
                'y': transform.transform.rotation.y,
                'z': transform.transform.rotation.z,
                'w': transform.transform.rotation.w
            }
            
            self.current_pose['timestamp'] = time.time()
            
        except Exception:
            # TF not available yet, use last known pose
            pass
    
    def _publish_pose(self):
        """Publish current robot pose to MQTT."""
        # Update Cartesian pose from TF
        self._update_cartesian_pose()
        
        # Convert to JSON
        pose_json = json.dumps(self.current_pose)
        
        # Publish to MQTT topic
        result = self.mqtt_client.publish('robot/pose', pose_json, qos=0)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"⚠️  Failed to publish pose. Error code: {result.rc}")
    
    def run(self):
        """Start the MQTT publisher."""
        print()
        print("=" * 60)
        print("Robot Pose MQTT Publisher")
        print("=" * 60)
        print(f"MQTT Broker: {self.broker_host}:{self.broker_port}")
        print(f"MQTT Topic: robot/pose")
        print(f"Publish Rate: {self.publish_rate} Hz")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(self.broker_host, self.broker_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"❌ Cannot connect to MQTT broker: {e}")
            return
        
        # Publishing loop
        rate = 1.0 / self.publish_rate
        
        try:
            while self.running:
                self._publish_pose()
                time.sleep(rate)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping publisher...")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("✅ Publisher stopped")
    
    def stop(self):
        """Stop the publisher."""
        self.running = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⏹️  Received interrupt signal...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Robot Pose MQTT Publisher for dataset generation"
    )
    parser.add_argument('--broker', type=str, default='localhost',
                       help='MQTT broker hostname/IP (default: localhost)')
    parser.add_argument('--port', type=int, default=1883,
                       help='MQTT broker port (default: 1883)')
    parser.add_argument('--robot-frame', type=str, default='tool0',
                       help='Robot end-effector frame (default: tool0)')
    parser.add_argument('--base-frame', type=str, default='base_link',
                       help='Robot base frame (default: base_link)')
    parser.add_argument('--rate', type=float, default=30.0,
                       help='Publishing rate in Hz (default: 30.0, recommended: 20-50 for 10 FPS video)')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    publisher = RobotPoseMQTTPublisher(
        broker_host=args.broker,
        broker_port=args.port,
        robot_frame=args.robot_frame,
        base_frame=args.base_frame,
        publish_rate=args.rate
    )
    
    publisher.run()


if __name__ == '__main__':
    main()
