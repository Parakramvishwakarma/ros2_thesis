# The following imports are necessary
import rclpy
import numpy as np
from rclpy.node import Node
import pandas as pd
import math 
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseStamped
import os

class PoseListener(Node):
    def __init__(self):
        super().__init__("pose_listener")
        self.pose_sub = self.create_subscription(PoseStamped, '/test', self.currentPoseCallback, 10)
        self.costMapSub = self.create_subscription(PoseStamped, '/amcl_pose', self.costmapCallback, 10)
        self.dic = {"type": [], "x": [], "y": []}

    
    def currentPoseCallback(self, msg: PoseStamped):
        position = msg.pose.position
        self.dic["type"].append("position")
        self.dic["x"].append(position.x)
        self.dic["y"].append(position.y)
        if self.dic["type"]:  # Check if there's any data to write
            df = pd.DataFrame(self.dic)
            df.to_csv("/home/parakram/tut_ws/src/robot_motion_controller/robot_motion_controller/csv/posetest.csv", index=False)
            self.get_logger().info("CSV WRITTEN")


# The code below should be left as is
def main(args=None):
    rclpy.init(args=args)

    node = PoseListener()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()