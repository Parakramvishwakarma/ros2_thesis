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
from map_msgs.msg import OccupancyGridUpdate
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


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
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10  # This can be adjusted as needed
        )
        # Create the subscriber
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            qos_profile)
        
        self.subscription_pose 


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
        self.pose_data = msg.pose.pose
   

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

        #these are all the intermediary variables used to create the state and the reward for the agent
        self.angularVelocityCounter = 0
        self.lastAngVelocity = None
        self.pathAngle = None
        self.goalAngle = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.collision = False


        #this is the param for how many poses ahead the path we look to find the path angle
        self.lookAheadDist = 40 

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = None

        #set reward parameters
        self.alpha = -0.2 #this one is for the path angle
        self.beta = 1.5 # this one is for the distance from the target
        self.gamma  = -0.2 #this is for closest obstacle
        self.roh = 0.2 #this is for linear.x speed
        self.mu  = -0.2 #this is penalty for spinnging
    
        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291

        #observation data variables from the environment
        self.scan_data = None
        self.speed_twist = None
        self.currentPose = None
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
        self.collision = False
        self.lastAngVelocity = None
        self.pathAngle = None
        self.goalAngle = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.counter = 0
        #these are data hodling variables
        self.scan_data = None
        self.speed_twist = None
        self.currentPose = None
        self.pathArray = None
        self.mapArray = None
        self.mapUpdateData = None
    
    def step(self, action):
        #the function includes a step counter to keep track of terminal //..condition
        start_time = time.perf_counter()  # Start the timer

        self.counter += 1
        linear_vel = action[0] * 5.0  
        angular_vel = action[1] * 3.14 
        self.publishNode.sendAction(linear_vel, angular_vel) #send action from the model
        self.subscribeNode.get_logger().info(f"This is the new action {linear_vel}, {angular_vel}")

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
    
        # if self.lastAngVelocity == None:
        #     self.lastAngVelocity = round(angular_vel, 3)
        # else:   
        #     if self.lastAngVelocity == round(angular_vel, 3):
        #         self.angularVelocityCounter += 1
        #         self.subscribeNode.get_logger().info(f"repetitive omega #:{self.angularVelocityCounter}")
        #     else:
        #         self.lastAngVelocity = round(angular_vel, 3)        

        # Wait for new scan and pose data
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        self.currentPose  = self.subscribeNode.pose_data

        # self.mapArray = self.subscribeNode.og_cost_map_grid
        # This doesn't neccesarily need to be stored here, what we wil do is 
        # that if there is an update we will just simply update the global map variable
        # self.mapUpdateData = self.subscribeNode.map_update
     
        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            self.subscribeNode.get_logger().info(f"Count {len(self.pathArray)}")
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")

            self.closestObstacle = min(self.scan_data.ranges)  #find the closest obstacle
            self._roundLidar()
            self.updateLidar()
            self.lastDistanceToTarget = self.newDistanceToTarget
            self.newDistanceToTarget = self._getDistance()
            # self.goalAngle = self._getGoalAngle()
            if len(self.pathArray) > self.lookAheadDist:
                self.pathAngle = self._calculate_heading_angle(self.currentPose, self.pathArray[self.lookAheadDist].pose)
            else:
                self.pathAngle = self._calculate_heading_angle(self.currentPose, self.pathArray[-1].pose)
            if self.lastDistanceToTarget:
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
                'goal_pose': [self.target_pose.position.x, self.target_pose.position.y],
                'current_pose': [self.currentPose.position.x, self.currentPose.position.y] ,
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

        end_time = time.perf_counter()  # End the timer
        execution_time = end_time - start_time  # Calculate the elapsed time
        self.subscribeNode.get_logger().info(f"Step function execution time: {execution_time:.6f} seconds")
        return observation, self.reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        #check if we have collided with an obstacle if so then run the backup and spin
        if self.collision:
            self._backup_and_spin()
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
        
        #get inintial observations
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        self.currentPose  = self.subscribeNode.pose_data

        # self.mapArray = self.subscribeNode.og_cost_map_grid
        # This doesn't neccesarily need to be stored here, what we wil do is 
        # that if there is an update we will just simply update the global map variable
        # self.mapUpdateData = self.subscribeNode.map_update

        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            self._roundLidar()
            self.updateLidar()
            self.newDistanceToTarget = self._getDistance()
            self.linearVelocity= round(self.speed_twist.linear.x, 2) 
            self.angularVelocity = round(self.speed_twist.angular.z,2)
            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'angular_velocity': self.angularVelocity,
                'goal_pose': [self.target_pose.position.x, self.target_pose.position.y],
                'current_pose': [self.currentPose.position.x, self.currentPose.position.y] ,
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
      

    def _calculateReward(self):
        obstacleReward = 0
        self.reward = self.pathAngle * self.alpha + (self.beta/self.newDistanceToTarget) + self.roh * self.linearVelocity + self.mu * self.angularVelocity
        if self.closestObstacle < 2:
            obstacleReward = (self.gamma)* (1 / self.closestObstacle) # this means a max pentaly possible is around -0.75
        self.reward += obstacleReward

    def _checkTerminalConditions(self): 
        if self.newDistanceToTarget < 0.5:
            self.reward = 1
            return True
        elif self.closestObstacle < 0.5:
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")
            self.reward = -1
            return True
        elif self.angularVelocityCounter >= 30:
            self.subscribeNode.get_logger().info("TERMINATED - CIRCULAR LOOP")
            self.reward = -1
            return True
        else:
            return False

    def _roundLidar(self):
        # Ensure self.scan_data.ranges is a numpy array first
        ranges_array = np.array(self.scan_data.ranges, dtype=np.float32)
        # Replace infinite values with 12 and ensure the result is a numpy array of floats
        processed_ranges = np.where(np.isinf(ranges_array), 12.0, ranges_array).astype(float)
        # Convert back to list if required by the LaserScan message
        self.scan_data.ranges = processed_ranges.tolist()

    def _getDistance(self):
        delta_x = self.target_pose.position.x - self.currentPose.position.x
        delta_y = self.target_pose.position.y - self.currentPose.position.y
        distance = float((delta_x**2 + delta_y**2) ** 0.5)
        return distance

    def _calculate_heading_angle(self,current_pose, goal_pose):
        # Extract x and y positions
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # Calculate the desired heading using atan2
        desired_heading = m.atan2(goal_y - current_y, goal_x - current_x)

        # Extract the quaternion from the current pose
        current_orientation = current_pose.orientation
        current_yaw = self._quaternion_to_yaw(
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        )

        # Calculate the heading angle difference
        heading_angle = desired_heading - current_yaw

        # Normalize the heading angle to the range [-pi, pi]
        heading_angle = (heading_angle + m.pi) % (2 * m.pi) - m.pi

        return heading_angle
    
    def _quaternion_to_yaw(self,x, y, z, w):
        """Convert a quaternion into yaw (rotation around Z-axis)"""
        # Yaw (Z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return m.atan2(siny_cosp, cosy_cosp)


    def updateLidar(self):
        # self.subscribeNode.get_logger().info(f"Length of rounded lidar {len(lidarObservation)}")
        self.lidarTracking[2] = self.lidarTracking[1]
        self.lidarTracking[1] = self.lidarTracking[0]
        self.lidarTracking[0] = self.scan_data.ranges

    def _backup_and_spin(self):
        """
        Backs up and spins the robot 180 degrees if a collision is detected.
        """
        # Back up straight
        self.publishNode.sendAction(-1.0, 0.0)  # Backward linear velocity
        time.sleep(1.5)  # Duration for backing up

        # Spin in place
        self.publishNode.sendAction(0.0, m.pi)  # Angular velocity for 180 degree turn
        time.sleep(2.0)  # Duration to spin around

        # Stop the robot
        self.publishNode.sendAction(0.0, 0.0)
        time.sleep(1.0)  # Ensure the robot stops completely

        self.subscribeNode.get_logger().info("Executed backup and spin recovery maneuver")
    







