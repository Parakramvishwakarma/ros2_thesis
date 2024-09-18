# The following imports are necessary
import rclpy
import numpy as np
from rclpy.node import Node
import pandas as pd
import math 
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseStamped
import os

class GraphPath(Node):
    def __init__(self):
        super().__init__("graph_path")
        self.dic = {"type": [], "x": [], "y": []}
        
        # Create two distinct subscriptions
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        
        # self.timer = self.create_timer(1, self.timer_callback)
    
    def goal_callback(self, msg: PoseStamped):
        position = msg.pose.position
        self.dic["type"].append("goal")
        self.dic["x"].append(position.x)
        self.dic["y"].append(position.y)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        position = msg.pose.pose.position
        self.dic["type"].append("position")
        self.dic["x"].append(position.x)
        self.dic["y"].append(position.y)
        if self.dic["type"] and len(self.dic["type"]) % 1000 == 0:  # Check if there's any data to write
            df = pd.DataFrame(self.dic)
            df.to_csv("/home/parakram/ros2_ws/src/robot_motion_controller/robot_motion_controller/csv/test.csv", index=False)
            self.get_logger().info("CSV WRITTEN")


# The code below should be left as is
def main(args=None):
    rclpy.init(args=args)

    node = GraphPath()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()