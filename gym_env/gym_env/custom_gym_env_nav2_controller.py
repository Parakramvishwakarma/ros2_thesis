import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import pandas as pd
import math as m
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from example_interfaces.msg import Float64MultiArray
from custom_interfaces.msg import Observations
from map_msgs.msg import OccupancyGridUpdate
import time
import matplotlib.pyplot as plt
import os

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription_speed = self.create_subscription(
            Odometry,
            '/diffdrive_controller/odom',
            self.speed_callback,
            qos_profile_sensor_data)
        self.subcription_path = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10)
        self.subcription_costMap = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.map_callback,
            10)
        self.subcription_costMap_update = self.create_subscription(
            OccupancyGridUpdate,
            '/global_costmap/costmap_updates',
            self.map_update_callback,
            10)
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            qos_profile_sensor_data)
        
        self.scan_data = None
        self.speed_data = None
        self.path_data = None
        self.og_cost_map_grid = None
        self.map_update = None
        self.pose_data = None


    def scan_callback(self, msg):
        self.scan_data = msg
   
    def speed_callback(self, msg):
        #this takes it straight to the linear and angular velocities
        self.speed_data = msg.twist.twist
       
    def path_callback(self, msg):
        #this is the array of poses to follow in the global plan
        self.path_data = msg.poses
        
    def map_callback(self, msg):
        #This is the array of occupancy grid we are not currently sure waht the obejctive of the origin is        
        self.og_cost_map_grid = msg.data

    def map_update_callback(self, msg):
        self.map_update = msg
        ##this needs code to how we update the orignal cost map
        ##msg type is map_msgs/msg/OccupancyGridUpdate
    
    def pose_callback(self, msg):
        self.pose_data = [msg.pose.pose.position.x, msg.pose.pose.position.y]
   

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

    def send_initial_pose(self, pose):
        # self.get_logger().info("SENDING THE INTIIAL POSE NOW MAKE SURE THE PLAY BUTTON HAS BEEN PRESSED")
        initialPose_pose = PoseWithCovarianceStamped()
        initialPose_pose.header.stamp.sec = 0
        initialPose_pose.header.stamp.nanosec = 0
        initialPose_pose.header.frame_id = "map"
        initialPose_pose.pose.pose = pose
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
        self.episode_length = 2000
        #inititalise variables

        self.data = {
            'timesteps': [],
            'path_angle': [],
            'change_distance': [],
            'distance_to_target': [],
            'reward': [],
            'speed' : [],
            'angular_speed' : []
        }
        self.plot_interval = 1000  # Interval for plotting


        self.angularVelocityCounter = 0
        self.lastAngVelocity = None
        self.pathAngle = None
        self.goalAngle = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.currentPose = []
        self.goalPose = []
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.array([[], [], []])

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = None
        self.intial_pose = Pose()

        #set reward parameters
        #this is the one for the closest obstacle 

        self.alpha = -0.2 #this one is for the goal angle
        
        self.beta = 3
        self.gamma  = -0.2
        


        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291

        #data variables
        self.scan_data = None
        self.speed_twist = None
        self.poseArray  = None
        self.pathArray = None
        self.mapArray = None
        self.mapUpdateData = None


        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=12, shape=(3,640), dtype=np.float32),
            'linear_velocity': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            'angular_velocity': spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            'goal_pose': spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
            'current_pose': spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
        })

    def _initialise(self):
        self.angularVelocityCounter = 0
        self.lastAngVelocity = None
        self.pathAngle = None
        self.goalAngle = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.currentPose = []
        self.goalPose = []
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.counter = 0

    
    def step(self, action):
        #the function includes a step counter to keep track of terminal //..condition
        self.counter += 1
        self.subscribeNode.get_logger().info(f"Count {self.counter}")
        linear_vel = action[0] * 5.0  
        angular_vel = action[1] * 3.14 
        self.publishNode.sendAction(linear_vel, angular_vel) #send action from the model
        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
    
        if self.lastAngVelocity == None:
            self.lastAngVelocity = round(angular_vel, 3)
        else:   
            if self.lastAngVelocity == round(angular_vel, 3):
                self.angularVelocityCounter += 1
                self.subscribeNode.get_logger().info(f"repetitive omega #:{self.angularVelocityCounter}")
            else:
                self.lastAngVelocity = round(angular_vel, 3)        

        # Wait for new scan and pose data
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        # self.mapArray = self.subscribeNode.og_cost_map_grid
        # This doesn't neccesarily need to be stored here, what we wil do is 
        # that if there is an update we will just simply update the global map variable
        # self.mapUpdateData = self.subscribeNode.map_update
        self.currentPose  = self.subscribeNode.pose_data


        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")

            self.closestObstacle = min(self.scan_data.ranges)  #find the closest obstacle
            self._roundLidar()
            self.updateLidar()

            self.goalPose = [self.target_pose.position.x, self.target_pose.position.y]
            self.lastDistanceToTarget = self.newDistanceToTarget
            self.newDistanceToTarget = self._getDistance()
            self.goalAngle = self._getGoalAngle()
            self.pathAngle = self._getPathAngle()
            self.changeInDistanceToTarget = self.newDistanceToTarget - self.lastDistanceToTarget
            
            self.linearVelocity= round(self.speed_twist.linear.x, 2) 
            self.angularVelocity = round(self.speed_twist.angular.z,2)

            #this is us updating the reward class variable with 
            self._calculateReward()
            #this will check the terminal conditions and if its terminated update self.reward accordingly
            terminated = self._checkTerminalConditions()

            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'angular_velocity': self.angularVelocity,
                'goal_pose': self.goalPose,
                'current_pose': self.currentPose ,
            }
        else:
            self.subscribeNode.get_logger().info("Scan or observation data missing")
            observation = self.observation_space.sample()
            self.reward = 0
            terminated = False

        #lastly if the episode is over and we have not terminated it then we end it and give a small negative reward for not reaching
        if terminated == False and self.counter > self.episode_length:
            self.subscribeNode.get_logger().info("Episode Finished")
            truncated = True
            self.reward = -1
        else:
            truncated = False

        # Store values for plotting
        self.data['timesteps'].append(self.counter)
        self.data['path_angle'].append(self.pathAngle)
        self.data['change_distance'].append(self.changeInDistanceToTarget)
        self.data['distance_to_target'].append(self.newDistanceToTarget)
        self.data['reward'].append(self.reward)
        self.data['speed'].append(self.linearVelocity)
        self.data['angular_speed'].append(self.angularVelocity)

        # Check if it's time to plot
        if len(self.data['reward']) % self.plot_interval == 0:
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv("data.csv")
        return observation, self.reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        #reset variables
        self._initialise()
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
            self._setNewTargetAndInitial()
            # self.publishNode.send_initial_pose(self.intial_pose)

        self.publishNode.send_goal_pose(self.target_pose)

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
        #get inintial obs
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        scan_data = self.subscribeNode.scan_data
        observation_data = self.subscribeNode.observationData

        if scan_data and observation_data:
            self.reward = observation_data.reward
            self.closestObstacle = min(scan_data.ranges)  #find the closest obstacle
            lidar_observation = self._roundLidar(scan_data.ranges)
            self.updateLidar(lidar_observation)
            self.currentPose = [observation_data.current_x, observation_data.current_y]
            self.goalPose = [self.target_pose.position.x, self.target_pose.position.y]
            self.newDistanceToTarget = observation_data.distance_target
            self.goalAngle = observation_data.goal_angle
            self.pathAngle = observation_data.path_angle
            self.linearVelocity= round(observation_data.speed, 2) 
            self.angularVelocity = round(observation_data.angular_speed,2)

            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'angular_velocity': self.angularVelocity,
                'goal_pose': self.goalPose,
                'current_pose': self.currentPose,
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

    def _setNewTargetAndInitial(self):
        self.target_pose.position.x = float(np.random.randint(-4,14))
        self.target_pose.position.y = float(np.random.randint(-9, 9))
        # self.intial_pose.position.x = float(np.random.randint(-4,14))
        # self.intial_pose.position.y = float(np.random.randint(-9, 9))

    def _calculateReward(self):
        obstacleReward = 0
        if self.closestObstacle < 6:
            obstacleReward = (self.gamma)* (1 / self.closestObstacle) # this means a max pentaly possible is around -0.75
        self.reward += obstacleReward
        # self.subscribeNode.get_logger().info(f"Distance: {distanceReward}, Obstacle {obstacleReward} Speed: {speedReward}")

    def _checkTerminalConditions(self): 
        if self.newDistanceToTarget < 0.5:
            self.reward = 1
            return True
        elif self.closestObstacle < 0.5:
            self.subscribeNode.get_logger().info("Terminated WE HIT AN OBSTACLE")
            linear_vel = -5.0  
            angular_vel = 0 
            self.publishNode.sendAction(linear_vel, angular_vel)
            self.reward = -1
            return True
        elif self.angularVelocityCounter >= 30:
            self.subscribeNode.get_logger().info("Terminated We are in a circular loop")
            self.reward = -1
            return True
        else:
            return False

    def _roundLidar(self):
        self.scan_data.ranges = np.where(np.isinf(self.scan_data.ranges), 12, self.scan_data.ranges)
        self.scan_data.ranges = np.array(self.scan_data.ranges, dtype=np.float32)

    def _getDistance(self):
        delta_x = self.target_pose.position.x - self.currentPose[0]
        delta_y = self.target_pose.position.y - self.poseArray[1]
        distance = float((delta_x**2 + delta_y**2) ** 0.5)
        return distance

    def updateLidar(self):
        # self.subscribeNode.get_logger().info(f"Length of rounded lidar {len(lidarObservation)}")
        self.lidarTracking[2] = self.lidarTracking[1]
        self.lidarTracking[1] = self.lidarTracking[0]
        self.lidarTracking[0] = self.scan_data.ranges







