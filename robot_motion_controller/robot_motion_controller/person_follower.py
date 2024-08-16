# The following imports are necessary
import rclpy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rclpy.node import Node
import math 
# Replace the following import with the interface this node is using
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan



# Class 
class PersonFollower(Node, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__("person_follower")
        self.following_distance = 0.2
        self.following_angle = 0
        self.angle_control_gain = 1.0
        self.following_distance_control_gain = 0.5
        self.get_logger().info(f'Following Distance {self.following_distance}')
        self.get_logger().info(f'following_angle {self.following_angle}')
        self.get_logger().info(f'angle_control_gain {self.angle_control_gain}')
        self.get_logger().info(f'distance_control_gain {self.following_distance_control_gain}')


        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)

        self.action_space = spaces.Discrete(100)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 1, 1), dtype=np.uint8)
    
    def listener_callback(self, msg):
        min_distance = min(msg.ranges)
        self.get_logger().info(f"The number of them are{len(msg.ranges)}")



# The code below should be left as is
def main(args=None):
    rclpy.init(args=args)

    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
