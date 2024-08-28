import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import math as m
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float64MultiArray
import time

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription_pose = self.create_subscription(
            Float64MultiArray,
            '/observations',
            self.observation_callback,
            10)

        self.scan_data = None
        self.observationData= None

    def scan_callback(self, msg):
        self.scan_data = msg

    def observation_callback(self, msg):
        self.observationData = msg.data
   
    def amcl_callback(self, msg):
        self.amcl_data = msg
    
class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.publish_initial_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.publish_goal_pose = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.publishAction = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def sendAction(self, linearVel, angularVel):
        msg = Twist()
        msg.linear.x = float(linearVel)
        msg.angular.z = float(angularVel)
        self.publishAction.publish(msg)

    def send_initial_pose(self):
        # self.get_logger().info("SENDING THE INTIIAL POSE NOW MAKE SURE THE PLAY BUTTON HAS BEEN PRESSED")
        initialPose_pose = PoseWithCovarianceStamped()
        initialPose_pose.header.stamp.sec = 0
        initialPose_pose.header.stamp.nanosec = 0
        initialPose_pose.header.frame_id = "map"
        initialPose_pose.pose.pose.position.x = 0.0 
        initialPose_pose.pose.pose.position.y = 0.0 
        initialPose_pose.pose.pose.position.z = 0.0 
        initialPose_pose.pose.pose.orientation.w = 1.0 
        self.publish_initial_pose.publish(initialPose_pose)
        
    def send_goal_pose(self, pose):
        # self.get_logger().info("SENDING THE GOAL POSE NOW MAKE SURE THE PLAY BUTTON HAS BEEN PRESSED")
        goalPose_pose = PoseStamped()
        goalPose_pose.header.stamp.sec = 0
        goalPose_pose.header.stamp.nanosec = 0
        goalPose_pose.header.frame_id = "map"
        goalPose_pose.pose = pose
        self.publish_goal_pose.publish(goalPose_pose)

class CustomGymnasiumEnvNav2(gym.Env):
    def __init__(self):
        super(CustomGymnasiumEnvNav2, self).__init__()
        rclpy.init()
        self.counter = 0
        #inititalise variables

        self.pathAngle = None
        self.goalAngle = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = None

        #set reward parameters
        self.beta = 3
        self.gamma  = -0.5
        self.alpha = 0.2

        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=12, shape=(640,), dtype=np.float32),
            # 'linear_velocity': spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32),
            'goal_angle': spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
            'path_angle': spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
            # 'angular_velocity': spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    
    def step(self, action):
        #the function includes a step counter to keep track of terminal condition
        self.counter += 1
        linear_vel = action[0] * 5.0  
        angular_vel = action[1] * 3.14 

        #send action from the model
        self.publishNode.sendAction(linear_vel, angular_vel)
        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
        
        # Wait for new scan and pose data
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        scan_data = self.subscribeNode.scan_data
        observation_data = self.subscribeNode.observationData
     

        #get udpated observations from odometry
        if (scan_data and observation_data):
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")
            self.closestObstacle = min(scan_data.ranges)  #find the closest obstacle
            lidar_observation = self._roundLidar(scan_data.ranges)
            self.newDistanceToTarget = observation_data[2]
            self.goalAngle = observation_data[0]
            self.pathAngle = observation_data[1]
            self.reward = observation_data[3]
            observation = {
                'lidar': lidar_observation,
                'goal_angle': np.array([self.goalAngle], dtype=np.float32),
                'path_angle': np.array([self.pathAngle], dtype=np.float32),
                'distance_to_target': np.array([self.newDistanceToTarget], dtype=np.float32)
            }
            self._calculateReward()
            reward = self.reward
            terminated = self.newDistanceToTarget < 0.5
            truncated = self.counter >= 2000
        else:
            self.subscribeNode.get_logger().info("Scan or odometry data missing")
            observation = self.observation_space.sample()
            reward = 0
            terminated = False
            truncated = self.counter >= 2000
        if terminated:
            self.subscribeNode.get_logger().info("WE REACHED THE GOAL")
        return observation, reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        #reset variables
        self.changeInDistanceToTarget = 0
        self.lastDistanceToTarget = None
        self.counter = 0
        #stop the robot
        self.publishNode.sendAction(0.0, 0.0)
        time.sleep(2)
        #set target
        if self.target_pose == None:
            self.target_pose = Pose()
            self.target_pose.position.x = 10.0
            self.target_pose.position.y = 0.0
            self.target_pose.position.z = 0.0
            self.target_pose.orientation.w = 1.0
        else:
            self._setNewTarget()

        # if (self.currentPose == None): 
        #     self.publishNode.send_initial_pose()
        self.publishNode.send_goal_pose(self.target_pose)
        # self.subscribeNode.get_logger().info(f"New Target is at [{self.target_pose.position.x}, {self.target_pose.position.y}]")

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
        #get inintial obs
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)
        scan_data = self.subscribeNode.scan_data
        observation_data = self.subscribeNode.observationData


        if scan_data and observation_data:
            lidar_observation = self._roundLidar(scan_data.ranges)
            self.newDistanceToTarget = observation_data[2]
            self.goalAngle = observation_data[0]
            self.pathAngle = observation_data[1]
            observation = {
                'lidar': lidar_observation,
                'goal_angle': np.array([self.goalAngle], dtype=np.float32),
                'path_angle': np.array([self.pathAngle], dtype=np.float32),
                'distance_to_target': np.array([self.newDistanceToTarget], dtype=np.float32)
            }
        else:
            observation = self.observation_space.sample()  # Return a random observation within space
        return observation, {}
    
    def render(self, mode='human'):
        pass

    def close(self):
        self.subscribeNode.destroy_node()
        self.publishNode.destroy_node()
        rclpy.shutdown()

    def _setNewTarget(self):
        self.target_pose.position.x = float(np.random.randint(-4,14))
        self.target_pose.position.y = float(np.random.randint(-9, 9))

    def _calculateReward(self):
        obstacleReward = 0
        if self.closestObstacle < 0.5:  
            obstacleReward = -3
        elif self.closestObstacle < 6:
            obstacleReward = (self.gamma)* (1 / self.closestObstacle) # this means a max pentaly possible is around -0.75

        self.reward += obstacleReward
        # self.subscribeNode.get_logger().info(f"Distance: {distanceReward}, Obstacle {obstacleReward} Speed: {speedReward}")


    def _roundLidar(self, original_ranges):
        original_ranges = np.where(np.isinf(original_ranges), 12, original_ranges)
        return np.array(original_ranges, dtype=np.float32)


    def _debugOutput(self):
        self.subscribeNode.get_logger().info(f"The current position is {self.currentPose.position.x} {self.currentPose.position.y}")
        self.subscribeNode.get_logger().info(f"The current velocity is: { self.currentLinearVelocity}")
        self.subscribeNode.get_logger().info(f"The closest obstacle is at: {self.closestObstacle}")
        self.subscribeNode.get_logger().info(f"The current distace to the target is: {self.newDistanceToTarget} m")
        # self.subscribeNode.get_logger().info(f"The the change in distance to target is {self.changeInDistanceToTarget}m")




