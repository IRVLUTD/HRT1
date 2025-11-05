#!/usr/bin/env python
import socket
import struct
from threading import Thread, Lock
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
import time
import tf
from tf.transformations import quaternion_matrix
import numpy as np
import yaml
import os

class Sensor:
    '''Class manager for ATI Force/Torque sensor via UDP/RDT with absolute and external force/torque computation.'''
    def __init__(self, ip="10.42.42.41"):
        # Initialization
        self.ip = ip
        self.port = 49152
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((ip, self.port))
        self.stream = False
        self.cpf = 1000000.0
        self.cpt = 1000000.0
        
        # ROS node and publishers
        rospy.init_node('ft_sensor', anonymous=True)
        self.pub_raw = rospy.Publisher('/gripper/ft_sensor/raw', WrenchStamped, queue_size=10)
        self.pub_absolute = rospy.Publisher('/gripper/ft_sensor/absolute', WrenchStamped, queue_size=10)
        self.pub_external_imu = rospy.Publisher('/gripper/ft_sensor/external_imu', WrenchStamped, queue_size=10)
        self.pub_external = rospy.Publisher('/gripper/ft_sensor/external', WrenchStamped, queue_size=10)
        self.data = None

        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # Get robot_type from ROS parameter server
        self.robot_type = rospy.get_param('~robot_type', 'fetch')  # Default to 'fetch'
        rospy.loginfo("Robot type: %s" % self.robot_type)

        # Load gripper configuration from YAML file
        self.load_gripper_config()

        # Gripper properties (set by load_gripper_config)
        self.gravity = 9.81  # m/s^2

        # Reset and capture bias
        self.send(0x0042)
        self.capture_initial_bias()

        # Subscribers
        self.sub_raw_absolute = rospy.Subscriber('/gripper/ft_sensor/raw', WrenchStamped, self.publish_absolute)
        self.sub_absolute_external_imu = rospy.Subscriber('/gripper/ft_sensor/absolute', WrenchStamped, self.publish_external_imu)
        self.sub_absolute_external = rospy.Subscriber('/gripper/ft_sensor/raw', WrenchStamped, self.publish_external)
        self.sub_imu = rospy.Subscriber('/gripper/imu', Imu, self.update_imu_data)
        
        # IMU data storage
        self.latest_imu = None
        #self.imu_lock = Lock() #Acceleration is continuous, therefore, partially updated data is better than slower data.

    def load_gripper_config(self):
        '''Load gripper mass and CoG from gripper_config.yaml based on robot_type.'''
        # Construct the path to the YAML file
        pkg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        yaml_path = os.path.join(pkg_path, 'config', 'gripper_config.yaml')
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if self.robot_type not in config:
                rospy.logerr("Robot type '%s' not found in %s. Using defaults." % (self.robot_type, yaml_path))
                self.gripper_mass = 1.56
                self.gripper_cog = np.array([0.0547, 0.0, 0.0])
            else:
                robot_config = config[self.robot_type]
                self.gripper_mass = float(robot_config['gripper_mass'])  # Ensure float
                self.gripper_cog = np.array(robot_config['gripper_cog'])  # Convert list to numpy array
                rospy.loginfo("Loaded gripper config - Mass: %s, CoG: %s" % (self.gripper_mass, self.gripper_cog))
        except Exception as e:
            rospy.logerr("Failed to load gripper config from %s: %s. Using defaults." % (yaml_path, e))
            self.gripper_mass = 1.56
            self.gripper_cog = np.array([0.0547, 0.0, 0.0])

    def capture_initial_bias(self):
        try:
            self.tf_listener.waitForTransform("base_link", "ati_link", rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "ati_link", rospy.Time(0))
            self.initial_rot = np.array(rot)
            rospy.loginfo("Captured initial pose of ati_link relative to base_link")
            
            rot_matrix = quaternion_matrix(self.initial_rot)[:3, :3]
            gravity_base = np.array([0.0, 0.0, -self.gripper_mass * self.gravity])
            self.bias_force = rot_matrix.T.dot(gravity_base)
            self.bias_torque = np.cross(self.gripper_cog, self.bias_force)
            rospy.loginfo("Gripper bias - Force: %s, Torque: %s" % (self.bias_force, self.bias_torque))
        except (tf.Exception) as e:
            rospy.logerr("Failed to capture initial pose: %s" % e)
            self.initial_rot = np.array([0.0, 0.0, 0.0, 1.0])
            self.bias_force = np.array([0.0, 0.0, 0.0])
            self.bias_torque = np.array([0.0, 0.0, 0.0])

    def send(self, command, count=0):
        header = 0x1234
        message = struct.pack('!HHI', header, command, count)
        self.sock.send(message)

    def receive(self):
        rawdata = self.sock.recv(1024)
        data = struct.unpack('!IIIiiiiii', rawdata)[3:]
        self.data = [data[i] for i in range(6)]
        return 

    def receiveHandler(self):
        while self.stream:
            self.receive()
            self.publish_to_ros()

    def startStreaming(self):
        self.getMeasurements(0)
        rospy.loginfo("NetFT Stream Started")
        self.stream = True
        self.receiveThread = Thread(target=self.receiveHandler)
        self.receiveThread.daemon = True
        self.receiveThread.start()

    def getMeasurements(self, n):
        self.send(2, count=n)

    def stopStreaming(self):
        self.stream = False
        time.sleep(0.1)
        self.send(0)
        
    def publish_to_ros(self):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "ati_link"
        msg.wrench.force.x = self.data[2] / self.cpf
        msg.wrench.force.y = -1.0 * self.data[0] / self.cpf
        msg.wrench.force.z = -1.0 * self.data[1] / self.cpf
        msg.wrench.torque.x = self.data[5] / self.cpt
        msg.wrench.torque.y = -1.0 * self.data[3] / self.cpt
        msg.wrench.torque.z = -1.0 * self.data[4] / self.cpt
        self.pub_raw.publish(msg)

    def update_imu_data(self, imu_msg):
        self.latest_imu = imu_msg

    def publish_absolute(self, raw_msg):
        absolute_msg = WrenchStamped()
        absolute_msg.header = raw_msg.header
        absolute_msg.header.frame_id = "ati_link"
        absolute_msg.wrench.force.x = raw_msg.wrench.force.x + self.bias_force[0]
        absolute_msg.wrench.force.y = raw_msg.wrench.force.y + self.bias_force[1]
        absolute_msg.wrench.force.z = raw_msg.wrench.force.z + self.bias_force[2]
        absolute_msg.wrench.torque.x = raw_msg.wrench.torque.x + self.bias_torque[0]
        absolute_msg.wrench.torque.y = raw_msg.wrench.torque.y + self.bias_torque[1]
        absolute_msg.wrench.torque.z = raw_msg.wrench.torque.z + self.bias_torque[2]
        self.pub_absolute.publish(absolute_msg)

    def publish_external_imu(self, msg):
        external_msg = WrenchStamped()
        external_msg.header = msg.header
        external_msg.header.frame_id = "ati_link"
        
        if self.latest_imu is None:
            return
        
        accel = np.array([
            self.latest_imu.linear_acceleration.y,
            self.latest_imu.linear_acceleration.x,
            self.latest_imu.linear_acceleration.z
        ])
        
        dynamic_force = self.gripper_mass * accel
        dynamic_torque = np.cross(self.gripper_cog, dynamic_force)
        external_msg.wrench.force.x = msg.wrench.force.x - dynamic_force[0]
        external_msg.wrench.force.y = msg.wrench.force.y - dynamic_force[1]
        external_msg.wrench.force.z = msg.wrench.force.z - dynamic_force[2]
        external_msg.wrench.torque.x = msg.wrench.torque.x - dynamic_torque[0]
        external_msg.wrench.torque.y = msg.wrench.torque.y - dynamic_torque[1]
        external_msg.wrench.torque.z = msg.wrench.torque.z - dynamic_torque[2]
    
        self.pub_external_imu.publish(external_msg)
    
    def publish_external(self, raw_msg):
        try:
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "ati_link", rospy.Time(0))
            R = quaternion_matrix(rot)[:3, :3]
            gravity_base = np.array([0.0, 0.0, -self.gravity])
            F_gravity = self.gripper_mass * R.T.dot(gravity_base)
            T_gravity = np.cross(self.gripper_cog, F_gravity)

            external_msg = WrenchStamped()
            external_msg.header = raw_msg.header
            external_msg.header.frame_id = "ati_link"
            external_msg.wrench.force.x = raw_msg.wrench.force.x - F_gravity[0] + self.bias_force[0]
            external_msg.wrench.force.y = raw_msg.wrench.force.y - F_gravity[1] + self.bias_force[1]
            external_msg.wrench.force.z = raw_msg.wrench.force.z - F_gravity[2] + self.bias_force[2]
            external_msg.wrench.torque.x = raw_msg.wrench.torque.x - T_gravity[0] + self.bias_torque[0]
            external_msg.wrench.torque.y = raw_msg.wrench.torque.y - T_gravity[1] + self.bias_torque[1]
            external_msg.wrench.torque.z = raw_msg.wrench.torque.z - T_gravity[2] + self.bias_torque[2]

            self.pub_external.publish(external_msg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed: %s" % e)

if __name__ == "__main__":
    sensor = Sensor()
    try:
        rospy.loginfo("Starting Net FT Streaming..")
        sensor.startStreaming()
        rospy.spin()
    finally:
        sensor.stopStreaming()
        rospy.loginfo("Net FT Streaming stopped")