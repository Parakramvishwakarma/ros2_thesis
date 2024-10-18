import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import pandas as pd
import csv
import math as m
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from rclpy.action import ActionClient
import time
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from builtin_interfaces.msg import Duration


class Subscriber(Node):
    def __init__(self, name):
        super().__init__(name)  # Use a unique node name

        # Define QoS profiles
        qos_profile_static = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_pose = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Create subscriptions
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data)

        self.subscription_speed = self.create_subscription(
            Odometry,
            '/diffdrive_controller/odom',
            self.speed_callback,
            qos_profile_sensor_data)

        self.subscription_path = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            qos_profile_static)

        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            qos_profile_pose)

        # Initialize variables to store the data
        self.scan_data = None
        self.speed_data = None
        self.path_data = None
        self.pose_data = Pose()
        self.pose_data.position.x = 0.0
        self.pose_data.position.y = 0.0
        self.pose_data.position.z = 0.0
        self.pose_data.orientation.w = 1.0

    # Callback methods to update data
    def scan_callback(self, msg):
        self.scan_data = msg

    def speed_callback(self, msg):
        self.speed_data = msg.twist.twist

    def path_callback(self, msg):
        self.path_data = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

    def pose_callback(self, msg):
        self.pose_data = msg.pose.pose

class Publisher(Node):
    def __init__(self, name):
        super().__init__(name)  # Use a unique node name
        self.publish_initial_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.publish_goal_pose = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.publishAction = self.create_publisher(Twist, '/cmd_vel', 10)

        # Services to clear the global and local costmaps
        self.local_costmap_clear_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.global_costmap_clear_client = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')

        # Ensure service servers are available
        self.local_costmap_clear_client.wait_for_service(timeout_sec=5.0)
        self.global_costmap_clear_client.wait_for_service(timeout_sec=5.0)

    def sendAction(self, linearVel, angularVel):
        msg = Twist()
        msg.linear.x = float(linearVel)
        msg.angular.z = float(angularVel)
        self.publishAction.publish(msg)

    def send_goal_pose(self, pose):
        goalPose_pose = PoseStamped()
        goalPose_pose.header.stamp.sec = 0
        goalPose_pose.header.stamp.nanosec = 0
        goalPose_pose.header.frame_id = "map"
        goalPose_pose.pose = pose
        self.publish_goal_pose.publish(goalPose_pose)

    # Asynchronous costmap clearing without blocking
    def clear_local_costmap(self):
        self.get_logger().info("Clearing local costmap asynchronously...")
        future = self.local_costmap_clear_client.call_async(ClearEntireCostmap.Request())
        future.add_done_callback(self.local_costmap_clear_callback)

    def clear_global_costmap(self):
        self.get_logger().info("Clearing global costmap asynchronously...")
        future = self.global_costmap_clear_client.call_async(ClearEntireCostmap.Request())
        future.add_done_callback(self.global_costmap_clear_callback)

    # Callbacks to handle the result of the asynchronous service call
    def local_costmap_clear_callback(self, future):
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info("Successfully cleared local costmap.")
            else:
                self.get_logger().info("Failed to clear local costmap.")
        except Exception as e:
            self.get_logger().error(f"Exception in clearing local costmap: {str(e)}")

    def global_costmap_clear_callback(self, future):
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info("Successfully cleared global costmap.")
            else:
                self.get_logger().info("Failed to clear global costmap.")
        except Exception as e:
            self.get_logger().error(f"Exception in clearing global costmap: {str(e)}")


class CustomGymnasiumEnvNav2(gym.Env):
    def __init__(self, pubName = "Publisher", subName = "Subscriber"):
        super(CustomGymnasiumEnvNav2, self).__init__()

        if not rclpy.ok():  # Check if rclpy has already been initialized
            rclpy.init()   

              #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber(subName)
        self.publishNode = Publisher(pubName)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.subscribeNode)
        self.executor.add_node(self.publishNode)

        # Start spinning the executor in a separate thread
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        #episode counter and lenght variables
        self.counter = 0
        self.episode_length = 4000

        self.csv_file = "./reward_log.csv"
        with open(self.csv_file, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(['TimeStep', 'HeadingError', 'ChangeInDistance', 'ClosestObstacle', 'LinearSpeed', 'AngularPenalty', 'PathDeviation', 'DistanceToGoal', 'TotalReward'])
        #reward components
        self.heading_penalty = 0
        self.distance_reward = 0
        self.obstacle_penalty = 0
        self.linear_speed_reward = 0
        self.angular_penalty = 0
        self.path_deviation_penalty = 0
        self.goal_reached_bonus = 0
        self.collision_penalty = 0
        self.overall_progress_reward = 0

        #these are all the intermediary variables used to create the state and the reward for the agent
        self.pathAngle = -1
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.collision = False
        self.obstacleAngle = None
        self.closestPathPointIndex  = 0
        self.closestPathDistance = 0
        self.lastPose = None

        #this is the param for how many poses ahead the path we look to find the path angle
        self.lookAheadDist = 40 
        self.lookAheadPointIndex = self.lookAheadDist
        self.lookAheadPoint = None

        #define the target pose for training
        self.target_pose = None
        self.relativeGoal = None
    
        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291

        #observation data variables from the environment
        self.scan_data = None
        self.speed_twist = None
        self.pathArray = None
        self.currentPose = Pose()
        self.currentPose.position.x = 0.0
        self.currentPose.position.y = 0.0
        self.currentPose.position.z = 0.0
        self.currentPose.orientation.w = 1.0

        # Define action and observation space
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3*640 + 10,), dtype=np.float32)
    
    def _spin(self):
        # Spin using the MultiThreadedExecutor
        self.executor.spin()
       
    def _initialise(self):
        self.lookAheadPointIndex = 0
        self.lookAheadPoint = None
        self.collision = False
        self.pathAngle = -1
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.counter = 0
        self.obstacleAngle = None
        self.closestPathPointIndex  = 0
        self.closestPathDistance = 0
        #reward components
        self.heading_penalty = 0
        self.distance_reward = 0
        self.obstacle_penalty = 0
        self.linear_speed_reward = 0
        self.angular_penalty = 0
        self.path_deviation_penalty = 0
        self.goal_reached_bonus = 0
        self.collision_penalty = 0
        self.overall_progress_reward = 0

    def _initialiseDataVars(self):
        self.currentPose = None
        self.scan_data = None
        self.speed_twist = None
        self.subscribeNode.scan_data = None
        self.subscribeNode.speed_data = None
        self.subscribeNode.pose_data = None


    def step(self, action):
        #the function includes a step counter to keep track of terminal //..condition
        self.counter += 1
        linear_vel = round(action[0],3)
        angular_vel = round(action[1], 3)
        
        #we need to make sure we initialise the data vars before the action is sent
        self._initialiseDataVars()
        self.publishNode.sendAction(linear_vel, angular_vel) #send action from the model
        #wait for action to happen
        time.sleep(0.75)
        #get data
        self._getObservations()
        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")
            self.lastPose = self.currentPose
            #find the pose of the target in the global frame
            self._findRelativeGoal()
            
            #process all the Lidar observations and update lidar array of historical observations
            self.closestObstacle = min(self.scan_data.ranges)  #find the closest obstacle
            self.obstacleAngle = round(self.scan_data.ranges.index(self.closestObstacle) * 0.5625,3)
            self.closestObstacle = round(self.closestObstacle, 3)
            self._roundLidar()
            self._updateLidar()

            self.lastDistanceToTarget = self.newDistanceToTarget
            self.newDistanceToTarget = self._getDistance()
            if self.lastDistanceToTarget:
                self.changeInDistanceToTarget = self.newDistanceToTarget - self.lastDistanceToTarget

            self._find_closest_path_point_and_distance() #this will update the class variables for closets point index and distance
            self.pathAngle = self._findPathAngle()

            self.linearVelocity= round(self.speed_twist.linear.x, 2) 
            self.angularVelocity = round(self.speed_twist.angular.z,2)
            #this is us updating the reward class variable with 
            self._calculateReward()
            #this will check te terminal conditions and if its terminated update self.reward accordingly
            terminated = self._checkTerminalConditions()
            other_obs = np.concatenate([
                np.array([self.linearVelocity, self.angularVelocity, self.pathAngle, self.closestPathDistance]),  # Single valued observations
                np.array(self.relativeGoal),  # 2D array
                np.array([self.currentPose.position.x, self.currentPose.position.y]),  # 2D array
                np.array([self.target_pose.position.x, self.target_pose.position.y])  # 2D array
            ])
            obs = np.concatenate([self.lidarTracking.flatten(), other_obs])
            observation = obs
        else:
            self.subscribeNode.get_logger().info("Scan or observation data missing")
            observation = self.observation_space.sample()
            self.reward = 0
            terminated = False

        #lastly if the episode is over and we have not terminated it then we end it and give a small negative reward for not reaching
        if terminated == False and self.counter > self.episode_length:
            self.subscribeNode.get_logger().info("Episode Finished")
            truncated = True
        else:
            truncated = False

        return observation, self.reward, terminated, truncated
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options) 
        #check if we have collided with an obstacle if so then run the backup and spin
        if self.collision:
            self._handleCollision()
        #reset the costmaps
 
        self.publishNode.sendAction(0.0, 0.0)
        self.publishNode.clear_local_costmap()
        self.publishNode.clear_global_costmap()

        if self.target_pose == None:
            #this case is for first ever episode
            self.target_pose = Pose()
            self.target_pose.position.x = 4.0
            self.target_pose.position.y = 2.5
            self.target_pose.position.z = 0.0
            self.target_pose.orientation.w = 1.0
        else:
            #reset variables
            self._initialise()
            self._setNewTargetAndInitial()

        self.publishNode.send_goal_pose(self.target_pose)
        rclpy.spin_once(self.publishNode, timeout_sec=0.1)

        time.sleep(2)
        #get inintial observations
        self._getObservations()

        #get udpated observations from odometry
        if (self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            self._findRelativeGoal()
            self._roundLidar()
            self._updateLidar()
            self.newDistanceToTarget = self._getDistance()
            self.linearVelocity= round(self.speed_twist.linear.x, 2) 
            self.angularVelocity = round(self.speed_twist.angular.z,2)
            self.lookAheadPointIndex = min(len(self.pathArray) -1, self.lookAheadDist)
            self.pathAngle = self._calculate_heading_angle(self.currentPose, self.pathArray[self.lookAheadPointIndex])
            other_obs = np.concatenate([
                np.array([self.linearVelocity, self.angularVelocity, self.pathAngle, self.closestPathDistance]),  # Single valued observations
                np.array(self.relativeGoal),  # 2D array
                np.array([self.currentPose.position.x, self.currentPose.position.y]),  # 2D array
                np.array([self.target_pose.position.x, self.target_pose.position.y])  # 2D array
            ])
            obs = np.concatenate([self.lidarTracking.flatten(), other_obs])
            observation = obs
        else:
            observation = self.observation_space.sample()  # Return a random observation within space
        return observation
    
    def render(self, mode='human'):
        pass

    def close(self):
        self.executor.shutdown()
        self.subscribeNode.destroy_node()
        self.publishNode.destroy_node()
        rclpy.shutdown()

    def _setNewTargetAndInitial(self):
        self.target_pose.position.x = float(np.random.randint(-4,14))
        self.target_pose.position.y = float(np.random.randint(-9, 9))

    def _handleCollision(self):
        while self.collision:
            if self.obstacleAngle >=135 and self.obstacleAngle <= 225:
                self.subscribeNode.get_logger().info(f"Running Backup Manouvre obstacle at front {self.obstacleAngle}")
                self.publishNode.sendAction(-5.0, 0.0)
            elif self.obstacleAngle > 225 and self.obstacleAngle <= 285:
                self.subscribeNode.get_logger().info(f"Obstacle on the left {self.obstacleAngle}")
                self.publishNode.sendAction(0.0, -3.0)
            elif self.obstacleAngle < 75 or self.obstacleAngle > 285:
                self.subscribeNode.get_logger().info(f"Obstacle at the back running front Manouvre {self.obstacleAngle}")
                self.publishNode.sendAction(5.0, 0.0)
            elif self.obstacleAngle >=75 or self.obstacleAngle < 135 :
                self.subscribeNode.get_logger().info(f"Obstacle on the right {self.obstacleAngle}")
                self.publishNode.sendAction(0.0, 3.0)
            time.sleep(2)
            self.scan_data = self.subscribeNode.scan_data
            if (self.scan_data):
                self._roundLidar()
                self.closestObstacle = min(self.scan_data.ranges)
                if self.closestObstacle > 1:
                    self.collision = False
                    self.subscribeNode.get_logger().info("Obstacle Not in Range anymore")
                else:
                    self.obstacleAngle = self.scan_data.ranges.index(self.closestObstacle) * 0.5625
                    self.subscribeNode.get_logger().info(f"Still in collision zone!!!!!! New closest: {min(self.scan_data.ranges)}")


    def _calculateReward(self):
        # Coefficients for each reward component
        alpha = -0.5  # Penalty for heading error
        beta = 5.0    # Reward for reducing distance to the goal
        gamma = -3.0  # Penalty for proximity to obstacles
        roh = 0.2     # Reward for maintaining linear speed
        delta = -0.8  # Path deviation penalty
        mu = -0.3     # Penalty for high angular velocity
        goal_reached_bonus = 2000  # Large bonus for reaching the goal
        collision_penalty = -1500  # High penalty for collisions

        # Calculate each reward component
        self.heading_penalty = alpha * self.pathAngle

        self.distance_reward = 0
        if self.lastDistanceToTarget is not None:
            self.distance_reward = beta * self.changeInDistanceToTarget

        self.overall_progress_reward = 10 / (self.newDistanceToTarget + 0.1)

        self.obstacle_penalty = 0
        if self.newDistanceToTarget > 1:
            if self.closestObstacle < 1.25:
                self.obstacle_penalty = gamma * (1 / self.closestObstacle)
    
        self.linear_speed_reward = roh * self.linearVelocity

        self.angular_penalty = mu * abs(self.angularVelocity)
        
        self.path_deviation_penalty = delta * self.closestPathDistance

        total_reward = (self.heading_penalty + self.distance_reward + self.obstacle_penalty +  self.overall_progress_reward +
                        self.linear_speed_reward + self.angular_penalty + self.path_deviation_penalty)

        if self.newDistanceToTarget < 0.5:  # Goal reached
            total_reward += goal_reached_bonus
            self.subscribeNode.get_logger().info("Goal reached!")
        elif self.closestObstacle < 0.5 and self.obstacleAngle >= 90 and self.obstacleAngle <= 270:  # Collision with obstacle
            total_reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")
        elif self.closestObstacle < 0.65 and (self.obstacleAngle < 90 or self.obstacleAngle > 270) :  # Collision with obstacle
            total_reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")

        self.reward = round(total_reward,3)
        # self.subscribeNode.get_logger().info(f"obs: {self.closestObstacle}, heading: {self.pathAngle}, dist: {self.newDistanceToTarget}, path_dev: {self.closestPathDistance} v: {self.linearVelocity} w: {self.angularVelocity}")
        # self.subscribeNode.get_logger().info(f"The total reward is {self.reward}")
        self._log_rewards_to_csv()
        return total_reward
    

    def _getObservations(self, reset = False):
        count = 0
        while (self.scan_data is None or self.speed_twist is None or self.currentPose is None or self.pathArray is None):
            # Get the latest data from the Subscriber class
            self.speed_twist = self.subscribeNode.speed_data
            self.pathArray = self.subscribeNode.path_data
            self.currentPose = self.subscribeNode.pose_data
            self.scan_data = self.subscribeNode.scan_data
            if self.scan_data is None:
                self.subscribeNode.get_logger().info("Scan is missing")
            if self.speed_twist is None:
                self.subscribeNode.get_logger().info("speeed is missing")
            if self.currentPose is None:
                count += 1
                if count == 3: 
                        if self.lastPose: self.currentPose = self.lastPose
                self.subscribeNode.get_logger().info("pose is missing")
            if self.pathArray is None:
                self.subscribeNode.get_logger().info(f"path is missing")
            time.sleep(0.2)  # Small sleep to avoid busy-waiting


    def _checkTerminalConditions(self): 
        if self.newDistanceToTarget < 0.5:  # Goal reached
            return True
        elif self.closestObstacle < 0.5 and self.obstacleAngle >= 90 and self.obstacleAngle <= 270 :  # Collision with obstacle
            return True
        elif self.closestObstacle < 0.65 and (self.obstacleAngle < 90 or self.obstacleAngle > 270) :  # Collision with obstacle
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
        return round(self.relativeGoal[0],3)

    def _calculate_heading_angle(self,current_pose, goal_pose):
        # Extract x and y positions
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        goal_x = goal_pose[0]
        goal_y = goal_pose[0]

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
        heading_angle = round(abs((heading_angle + m.pi) % (2 * m.pi) - m.pi),2)

        return heading_angle
    
    def _quaternion_to_yaw(self,x, y, z, w):
        """Convert a quaternion into yaw (rotation around Z-axis)"""
        # Yaw (Z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return m.atan2(siny_cosp, cosy_cosp)


    def _updateLidar(self):
        # self.subscribeNode.get_logger().info(f"Length of rounded lidar {len(lidarObservation)}")
        self.lidarTracking[2] = self.lidarTracking[1]
        self.lidarTracking[1] = self.lidarTracking[0]
        self.lidarTracking[0] = self.scan_data.ranges

    
    def _findRelativeGoal(self):
      
        # Extract robot's current position and orientation (yaw) in the global frame
        x_robot = self.currentPose.position.x
        y_robot = self.currentPose.position.y
        yaw_robot = self._quaternion_to_yaw(
            self.currentPose.orientation.x,
            self.currentPose.orientation.y,
            self.currentPose.orientation.z,
            self.currentPose.orientation.w
        )

        x_goal = self.target_pose.position.x
        y_goal = self.target_pose.position.y
        
        dx = x_goal - x_robot
        dy = y_goal - y_robot
        goal_yaw_global = m.atan2(dy, dx)

        # Calculate the relative yaw (difference between the goal's yaw and the robot's current yaw)
        relative_yaw = goal_yaw_global - yaw_robot

        relative_yaw = (relative_yaw + m.pi) % (2 * m.pi) - m.pi
        
        # Transform the displacement vector into the robot's local frame
        closestDis = m.sqrt(dx**2 + dy**2)
        
        self.relativeGoal =  np.array([closestDis, relative_yaw]).astype(float)

    def _find_closest_path_point_and_distance(self):
        """
        Find the closest point on the path to the robot's current position.
        """
        if not self.pathArray:
            return None  # Return None if no path is available

        min_distance = float('inf')
        closest_point_index = 0
        robot_x = self.currentPose.position.x
        robot_y = self.currentPose.position.y

        # Iterate through each waypoint in the path
        for i, waypoint in enumerate(self.pathArray):
            path_x = waypoint[0]
            path_y = waypoint[0]

            # Calculate Euclidean distance to the waypoint
            distance = m.sqrt((robot_x - path_x) ** 2 + (robot_y - path_y) ** 2)

            # Update minimum distance and index
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i

        self.closestPathDistance = round(min_distance,2)
        self.closestPathPointIndex =  closest_point_index

    def _findPathAngle(self):
        self.lookAheadPointIndex = min(self.closestPathPointIndex + self.lookAheadDist, len(self.pathArray) - 1)
        self.lookAheadPoint = self.pathArray[self.lookAheadPointIndex]
        return self._calculate_heading_angle(self.currentPose, self.lookAheadPoint)


    def _log_rewards_to_csv(self):
        # Append the reward components to the CSV file at each time step
        with open(self.csv_file, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([self.counter, self.pathAngle, self.changeInDistanceToTarget, self.closestObstacle,
                             self.linearVelocity, self.angularVelocity, self.closestPathDistance, self.newDistanceToTarget, self.reward])





