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
            Odometry,
            '/diffdrive_controller/odom',
            self.pose_callback,
            qos_profile_sensor_data)
        self.subscription_amcl = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            qos_profile_sensor_data)
        self.scan_data = None
        self.odom_data = None
        self.amcl_data = None

    def scan_callback(self, msg):
        self.scan_data = msg

    def pose_callback(self, msg):
        self.odom_data = msg
        
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
        self.get_logger().info("SENDING THE INTIIAL POSE NOW MAKE SURE THE PLAY BUTTON HAS BEEN PRESSED")
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
        self.get_logger().info("SENDING THE GOAL POSE NOW MAKE SURE THE PLAY BUTTON HAS BEEN PRESSED")
        goalPose_pose = PoseStamped()
        goalPose_pose.header.stamp.sec = 0
        goalPose_pose.header.stamp.nanosec = 0
        goalPose_pose.header.frame_id = "map"
        goalPose_pose.pose = pose
        self.publish_goal_pose.publish(goalPose_pose)

class CustomGymnasiumEnv(gym.Env):
    def __init__(self):
        super(CustomGymnasiumEnv, self).__init__()
        rclpy.init()
        self.counter = 0
        #inititalise variables
        self.currentPose = None
        self.currentLinearVelocity = None
        self.currentAngularVelocity = None
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.lastTime = None
        self.closestObstacle = None

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = Pose()
        self.target_pose.orientation.w = 1.0
    

        #set reward parameters
        self.beta = 2
        self.gamma  = -0.25
        self.alpha = 0.5

        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291


        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=12, shape=(640,), dtype=np.float32),
            'linear_velocity': spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32),
            'angular_velocity': spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def _get_distance(self, pose ):
        #pythagorous to find distance between (x,y) coordinates of target and robot
        delta_x = self.target_pose.position.x - pose.position.x
        delta_y = self.target_pose.position.y - pose.position.y
        distance = float((delta_x**2 + delta_y**2) ** 0.5)
        return distance

    def _calculateReward(self):
        #the reward is: Reach target (<0.25m) = +20 hit obstacle(<0.25m) = -10
        # alpha * current linear velocity 
        # -gamma * 1/distance to closest obstacle
        # beta * delta of distance to target
        reward = 0
        distanceReward = 0
        obstacleReward = 0
        speedReward = 0
        terminalReward = 0
        
        #this first case is for reaching the target
        if (self.newDistanceToTarget < 0.3):
            # the reward for reaching the target needs to outweight the penalty for being close to obstacles
            #this is becase this will ensure that we can reach targets close to obstacles
            terminalReward = 3
        else:
            #there is a reward for positive change in distance to target
            distanceReward = self.beta * self.changeInDistanceToTarget 
        #this case is for collision
        if self.closestObstacle < 0.3:  
            obstacleReward = -1
        elif self.closestObstacle < 2:
            obstacleReward = self.gamma * (1 / self.closestObstacle) # this means a max pentaly possible is around -0.75
        elif self.closestObstacle < 4:
            obstacleReward = (self.gamma/2) * (1 / self.closestObstacle) # this means a max pentaly possible is around -0.75
        #now reward for velocity
        speedReward = self.alpha * abs(self.currentLinearVelocity)

        reward = round(distanceReward,3) + round(obstacleReward, 3) + round(speedReward, 3) + round(terminalReward,3)
        self.subscribeNode.get_logger().info(f"Distance: {distanceReward}, Obstacle {obstacleReward} Speed: {speedReward}")
        return reward


    def _roundLidar(self, original_ranges):
        #round any element that is inf to 12 which is the max range of the LiDAR sensor
        original_ranges = np.where(np.isinf(original_ranges), 12, original_ranges)
        return np.array(original_ranges, dtype=np.float32)

    def step(self, action):
        #increase counter to count the time step of episode
        self.counter += 1
        #scale the policy's action given in between [-1, 1] into linear and angular velocity
        linear_vel = action[0] * 5.0  
        angular_vel = action[1] * 3.14 
        #publish action of the policy to the environment
        self.publishNode.sendAction(linear_vel, angular_vel)
        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
        
        # Scan for observations
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        scan_data = self.subscribeNode.scan_data
        odom_data = self.subscribeNode.odom_data
        amcl_data = self.subscribeNode.amcl_data

        #Check if observations are recieved from environment for odometry, lidar and amcl pose data
        if (odom_data and scan_data and amcl_data):
            #store the current position of the robot
            self.currentPose = amcl_data.pose.pose
            #store the current velocity
            self.currentLinearVelocity= round(odom_data.twist.twist.linear.x, 2) 
            self.currentAngularVelocity = round(odom_data.twist.twist.angular.z, 2 )
            #calcualte the new in distance to target pose
            self.newDistanceToTarget = self._get_distance(self.currentPose)
            #calculate the change in distance to the target pose
            if (self.lastDistanceToTarget):
                self.changeInDistanceToTarget = self.lastDistanceToTarget - self.newDistanceToTarget
            #update the last distance to target
            self.lastDistanceToTarget = self.newDistanceToTarget

            #find the distance from the LiDAR data to the closest obstacle
            self.closestObstacle = min(scan_data.ranges)  #find the closest obstacle
            #round the inf values to 12 in the LiDAR input
            lidar_observation = self._roundLidar(scan_data.ranges)
            #set the observations to be send to the agent
            observation = {
                'lidar': lidar_observation,
                'linear_velocity': np.array([self.currentLinearVelocity], dtype=np.float32),
                'angular_velocity': np.array([self.currentAngularVelocity], dtype=np.float32),
                'distance_to_target': np.array([self.newDistanceToTarget], dtype=np.float32)
            }
            #calculate the reward for the agent
            reward = self._calculateReward()
            #check if the robot has reached its target
            terminated = self.newDistanceToTarget < 0.3
            #check if the number of time steps of the episode have reached maximum
            truncated = self.counter >= 20000
        else:
            #create random observation
            observation = self.observation_space.sample()
            reward = 0
            terminated = False
            truncated = self.counter >= 20
        #return to the RL algorithm
        return observation, reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        #reset variables
        self.changeInDistanceToTarget = 0
        self.lastDistanceToTarget = None
        self.counter = 0
        #stop the robot by sending a velocity of 0,0
        self.publishNode.sendAction(0.0, 0.0)
        time.sleep(2)
        #set a new randomised target
        self._setNewTarget()
        #if the robot has not been given an intial pose send an initial pose to the AMCL localisation node
        if (self.currentPose == None): 
            self.publishNode.send_initial_pose()
        #publish new goal to the map
        self.publishNode.send_goal_pose(self.target_pose)
        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
  
        #get initial observations
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)
        scan_data = self.subscribeNode.scan_data
        odom_data = self.subscribeNode.odom_data
        amcl_data = self.subscribeNode.amcl_data
        #if the observations are received send them to the RL algorithm
        if scan_data and odom_data and amcl_data:
            lidar_observation = self._roundLidar(scan_data.ranges)
            self.currentPose = amcl_data.pose.pose
            self.currentLinearVelocity= round(odom_data.twist.twist.linear.x, 2) 
            self.currentAngularVelocity = round(odom_data.twist.twist.angular.z, 2 )
            observation = {
                'lidar': lidar_observation,
                'linear_velocity': np.array([self.currentLinearVelocity], dtype=np.float32),
                'angular_velocity': np.array([self.currentAngularVelocity], dtype=np.float32),
                'distance_to_target': np.array([self._get_distance(self.currentPose)], dtype=np.float32)
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
        #randomly choose x and y coordinates for new target location in the bounds of the map
        self.target_pose.position.x = float(np.random.randint(-5,15))
        self.target_pose.position.y = float(np.random.randint(-10, 10))



