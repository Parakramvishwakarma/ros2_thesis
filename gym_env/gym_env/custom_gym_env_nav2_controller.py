import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import pandas as pd
import math as m
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from nav2_msgs.action import BackUp, Spin
from rclpy.action import ActionClient
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from builtin_interfaces.msg import Duration



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

        self.scan_data = None
        self.speed_data = None
        self.path_data = None
        self.pose_data = None

    def scan_callback(self, msg):
        self.scan_data = msg

    def speed_callback(self, msg):
        #this takes it straight to the linear and angular velocities
        self.speed_data = msg.twist.twist

    def path_callback(self, msg):
        #this is the array of poses to follow in the global plan
        self.path_data = msg.poses

    def pose_callback(self, msg):
        self.pose_data = msg.pose.pose


class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.publish_initial_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.publish_goal_pose = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.publishAction = self.create_publisher(Twist, '/cmd_vel', 10)

        #these are action clients to get my robot out of trouble after collision
        self.backup_client = ActionClient(self, BackUp, '/backup')
        self.spin_client = ActionClient(self, Spin, '/spin')

        #services to clear the global and the local costmap:
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

    def clear_local_costmap(self):
        self.get_logger().info("Clearing local costmap...")
        future = self.local_costmap_clear_client.call_async(ClearEntireCostmap.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Successfully cleared local costmap.")
        else:
            self.get_logger().error(f"Failed to clear local costmap: {future.exception()}")

    def clear_global_costmap(self):
        self.get_logger().info("Clearing global costmap...")
        future = self.global_costmap_clear_client.call_async(ClearEntireCostmap.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Successfully cleared global costmap.")
        else:
            self.get_logger().error(f"Failed to clear global costmap: {future.exception()}")

class CustomGymnasiumEnvNav2(gym.Env):
    def __init__(self):
        super(CustomGymnasiumEnvNav2, self).__init__()
        rclpy.init()
        self.counter = 0
        self.episode_length = 4000

        #these are all the intermediary variables used to create the state and the reward for the agent
        self.lastDistanceToTarget = None
        self.newDistanceToTarget = None
        self.changeInDistanceToTarget = 0
        self.closestObstacle = None
        self.reward = 0
        self.linearVelocity = None
        self.angularVelocity = None
        self.lidarTracking = np.zeros((3, 640), dtype=np.float32)
        self.collision = False
        self.prunedPath = []
        self.obstacleAngle = None
        self.headingAngle = None
        self.lastHeadingAngle = None
        self.changeInHeadingAngle = 0
        self.closestPathPointIndex  = 0
        self.closestPathDistance = None
        self.lookAheadPointIndex = 0
        self.lookAheadPoint = None

        #this is the param for how many poses ahead the path we look to find the path angle
        self.lookAheadDist = 50

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = None

        #scanner parameters
        self.scannerRange = [0.164000004529953, 12.0]
        self.scannerIncrementRads = 0.009817477315664291

        #observation data variables from the environment
        self.scan_data = None
        self.speed_twist = None
        self.currentPose = None
        self.pathArray = None
        self.distanceToGoal = None

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=12, shape=(3,640), dtype=np.float32),
            'linear_velocity': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            'relative_goal': spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
            'heading_error': spaces.Box(low=0.0, high=4.0, shape=(1,), dtype=np.float32),
            'current_pose': spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
            'target_pose' : spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
            'global_path': spaces.Box(low=-50.0, high=50.0, shape=(self.lookAheadDist,2), dtype=np.float32),
        })


    def _initialise(self):
        self.headingAngle = None
        self.lastHeadingAngle = None
        self.changeInHeadingAngle = 0
        self.closestPathPointIndex  = 0
        self.closestPathDistance = None
        self.lookAheadPointIndex = 0
        self.lookAheadPoint = None
        self.prunedPath = []
        self.collision = False
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
        #these are data hodling variables
        self.scan_data = None
        self.speed_twist = None
        self.currentPose = None
        self.pathArray = None
        self.distanceToGoal = None

    def step(self, action):
        #the function includes a step counter to keep track of terminal //..condition
        self.counter += 1
        linear_vel = action[0] * 5.0
        angular_vel = action[1] * 3.14
        self.publishNode.sendAction(linear_vel, angular_vel) #send action from the model

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
        # Wait for new scan and pose data
        time.sleep(0.5)
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        self.currentPose  = self.subscribeNode.pose_data

        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")
            self.lastDistanceToTarget = self.newDistanceToTarget
            self.lastHeadingAngle = self.headingAngle
            if self.lastDistanceToTarget:
                self.newDistanceToTarget = self._getDistance()
                self.changeInDistanceToTarget = round(self.newDistanceToTarget - self.lastDistanceToTarget,3)
            if self.lastHeadingAngle:
                self.headingAngle = self._getDistance()
                self.changeInHeadingAngle = round(self.lastHeadingAngle - self.newDistanceToTarget,3)


            self.distanceToGoal = self._getDistanceToGoal()
            
            #we will go an update the points now for the new iteration
            self._find_closest_path_point_and_distance() #this will update the attributes for closest point index and distance

            self._findLookAheadPoint() #this gets the new lookahead point

            #process all the Lidar observations and update lidar array of historical observation
            self._roundLidar()
            self._updateLidar()
            self.closestObstacle = min(self.scan_data.ranges)  #find the closest obstacle
            self.obstacleAngle = round(self.scan_data.ranges.index(self.closestObstacle) * 0.5625,3)
            self.closestObstacle = round(self.closestObstacle, 3)
           
            self.headingAngle = self._calculate_heading_angle()
            self.newDistanceToTarget = self._getDistance() # the last one up there is used to find new distance to the old lookahead this used to find to new
 
            #this will take the path array that is given and convert the poses in the array into an array of x,y coordinates
            self._prunePath()

            self.linearVelocity= round(self.speed_twist.linear.x, 2)
            self.angularVelocity = round(self.speed_twist.angular.z,2)


            #this is us updating the reward class variable with
            self._calculateReward()

            #this will check the terminal conditions and if its terminated update self.reward accordingly
            terminated = self._checkTerminalConditions()

            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'relative_goal': self.newDistanceToTarget,
                'heading_error ': self.headingAngle,
                'current_pose': [self.currentPose.position.x, self.currentPose.position.y],
                'target_pose': [self.lookAheadPoint.position.x, self.lookAheadPoint.position.y] ,
                'global_path': self.prunedPath,
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
        else:
            truncated = False

        return observation, self.reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        #reset the costmaps
        self.publishNode.clear_local_costmap()
        self.publishNode.clear_global_costmap()

        #check if we have collided with an obstacle if so then run the backup and spin
        while self.collision:
            # self._backup_and_spin()
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
            rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)
            self.scan_data = self.subscribeNode.scan_data
            if (self.scan_data):
                self._roundLidar()
                self.closestObstacle = min(self.scan_data.ranges)
                if self.closestObstacle > 0.75:
                    self.collision = False
                    self.subscribeNode.get_logger().info("Obstacle Not in Range anymore")
                else:
                    self.obstacleAngle = self.scan_data.ranges.index(self.closestObstacle) * 0.5625
                    self.subscribeNode.get_logger().info(f"Still in collision zone!!!!!! New closest: {min(self.scan_data.ranges)}")

        #reset variables
        self._initialise()
        self.publishNode.sendAction(0.0, 0.0)
        time.sleep(2)
        if self.target_pose == None:
            self.target_pose = Pose()
            self.target_pose.position.x = 4.0
            self.target_pose.position.y = 2.5
            self.target_pose.position.z = 0.0
            self.target_pose.orientation.w = 1.0
        else:
            self._setNewTargetAndInitial()

        self.publishNode.send_goal_pose(self.target_pose)

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)

        #get inintial observations
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        self.currentPose  = self.subscribeNode.pose_data

        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray and len(self.pathArray) > 0):
            #set the lookahead and closest vars
            self.closestPathPointIndex = 0
            self.lookAheadPointIndex = min(self.lookAheadDist, len(self.pathArray) -1)
            self.lookAheadPoint = self.pathArray[self.lookAheadPointIndex].pose
            self.subscribeNode.get_logger().info(f"{self.lookAheadPoint.position.x}")
            #process the lidar
            self._roundLidar()
            self._updateLidar()
            self.newDistanceToTarget = self._getDistance()
            self.headingAngle = self._calculate_heading_angle()
            self.linearVelocity= round(self.speed_twist.linear.x, 2)
            self.angularVelocity = round(self.speed_twist.angular.z,2)
            self._prunePath()
            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'relative_goal': self.newDistanceToTarget,
                'heading_error ': self.headingAngle,
                'current_pose': [self.currentPose.position.x, self.currentPose.position.y],
                'target_pose': [self.lookAheadPoint.position.x, self.lookAheadPoint.position.y] ,
                'global_path': self.prunedPath,
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

    def _prunePath(self):
        self.prunedPath = np.zeros((self.lookAheadDist, 2), dtype=np.float32)
        for i in range(self.closestPathPointIndex, self.lookAheadPointIndex):
            self.prunedPath[i - self.closestPathPointIndex] = [self.pathArray[i].pose.position.x, self.pathArray[i].pose.position.y]

    def _calculateReward(self):
        # Coefficients for each reward component
        beta = 5.0    # Reward for reducing distance to the goal
        alpha = 2
        gamma = -4 # Penalty for proximity to obstacles
        roh = 0.7    # Reward for maintaining linear speed
        goal_reached_bonus = 2000  # Large bonus for reaching the goal
        collision_penalty = -1500  # High penalty for collisions
        # Base reward
        reward = 0

        reward += beta * self.changeInDistanceToTarget

        if self.closestObstacle < 1.5:
            obstacle_penalty = (1 / self.closestObstacle)  # Higher penalty the closer the obstacle
            reward += gamma * obstacle_penalty

        reward += roh * self.linearVelocity

        reward += alpha * self.changeInHeadingAngle #heading angle reward

        if self.distanceToGoal < 0.5:  # Goal reached
            reward += goal_reached_bonus
            self.subscribeNode.get_logger().info("Goal reached!")
        elif self.closestObstacle < 0.5 and self.obstacleAngle >= 90 and self.obstacleAngle <= 270 :  # Collision with obstacle
            reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")
        elif self.closestObstacle < 0.65 and (self.obstacleAngle < 90 or self.obstacleAngle > 270) :  # Collision with obstacle
            reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")

        self.reward = reward
        self.subscribeNode.get_logger().info(f"obs: {self.closestObstacle}, dist: {self.changeInDistanceToTarget}, vel: {self.linearVelocity}")
        self.subscribeNode.get_logger().info(f"The reward is {self.reward}")
        return reward

    def _checkTerminalConditions(self):
        if self.distanceToGoal < 0.5:  # Goal reached
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
        robot_x = self.currentPose.position.x
        robot_y = self.currentPose.position.y
        look_x = self.lookAheadPoint.position.x
        look_y = self.lookAheadPoint.position.y
        return round(m.sqrt((robot_x - look_x) ** 2 + (robot_y - look_y) ** 2),2)
    
    def _getDistanceToGoal(self):
        robot_x = self.currentPose.position.x
        robot_y = self.currentPose.position.y
        goal_x = self.target_pose.position.x
        goal_y = self.target_pose.position.y
        return round(m.sqrt((robot_x - goal_x) ** 2 + (robot_y - goal_y) ** 2),2)

    def _updateLidar(self):
        # self.subscribeNode.get_logger().info(f"Length of rounded lidar {len(lidarObservation)}")
        self.lidarTracking[2] = self.lidarTracking[1]
        self.lidarTracking[1] = self.lidarTracking[0]
        self.lidarTracking[0] = self.scan_data.ranges


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
            path_x = waypoint.pose.position.x
            path_y = waypoint.pose.position.y

            # Calculate Euclidean distance to the waypoint
            distance = m.sqrt((robot_x - path_x) ** 2 + (robot_y - path_y) ** 2)

            # Update minimum distance and index
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i

        self.closestPathDistance = min_distance
        self.closestPathPointIndex =  closest_point_index

    
        
    def _quaternion_to_yaw(self,x, y, z, w):
        """Convert a quaternion into yaw (rotation around Z-axis)"""
        # Yaw (Z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return m.atan2(siny_cosp, cosy_cosp)


    def _calculate_heading_angle(self):
        # Extract x and y positions
        current_x = self.currentPose.position.x
        current_y = self.currentPose.position.y
        goal_x = self.lookAheadPoint.position.x
        goal_y = self.lookAheadPoint.position.y

        # Calculate the desired heading using atan2
        desired_heading = m.atan2(goal_y - current_y, goal_x - current_x)

        # Extract the quaternion from the current pose
        current_orientation = self.currentPose.orientation
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

    def _findLookAheadPoint(self):
        self.lookAheadPointIndex = min(self.closestPathPointIndex + self.lookAheadDist, len(self.pathArray) - 1)
        self.lookAheadPoint = self.pathArray[self.lookAheadPointIndex].pose







