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

        #these are action clients to get my robot out of trouble after collision
        self.backup_client = ActionClient(self, BackUp, '/backup')
        self.spin_client = ActionClient(self, Spin, '/spin')

        #services to clear the global and the local costmap:
        self.local_costmap_clear_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.global_costmap_clear_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        
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
    
    def send_backup_goal(self):
        goal_msg = BackUp.Goal()
        goal_msg.target = Point(x=2.0, y=0.0, z=0.0)
        goal_msg.speed = float(0.1)
        goal_msg.time_allowance = Duration(sec=5, nanosec=0)  # Adjust as needed
        # self.get_logger().info("Sending backup goal")
        self.backup_client.wait_for_server()
        future = self.backup_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        # if future.result().status == 4:
        #     self.get_logger().info("Backup completed successfully")
        # else:
        #     self.get_logger().error("Backup action failed")


    def send_spin_goal(self):
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = float(1.57)
        # self.get_logger().info("Sending Spin goal")
        self.spin_client.wait_for_server()
        future = self.spin_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        # self.spin_client.wait_for_result()
    
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
        self.pathArrayConverted = []
        self.obstacleAngle = None

        #this is the param for how many poses ahead the path we look to find the path angle
        self.lookAheadDist = 40 

        #define the subcriber and publisher nodes
        self.subscribeNode = Subscriber()
        self.publishNode = Publisher()

        #define the target pose for training
        self.target_pose = None
        self.relativeGoal = None

        #set reward parameters
        self.alpha = -0.25 #this one is for the path angle
        self.beta = 1 # this one is for the distance from the target
        self.gamma  = -0.45 #this is for closest obstacle
        self.roh = 0.3 #this is for linear.x speed
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
            'angular_velocity': spaces.Box(low=0, high=3.14, shape=(1,), dtype=np.float32),
            'heading_error': spaces.Box(low=0, high=3.14, shape=(1,), dtype=np.float32),
            'relative_goal': spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
            'global_path': spaces.Box(low=-50.0, high=50.0, shape=(400,2), dtype=np.float32),
        })

    def _initialise(self):
        self.relativeGoal = None
        self.angularVelocityCounter = 0
        self.pathArrayConverted = []
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
        self.obstacleAngle = None
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

        rclpy.spin_once(self.publishNode, timeout_sec=1.0)
    
        # Wait for new scan and pose data
        rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)

        self.scan_data = self.subscribeNode.scan_data
        self.speed_twist = self.subscribeNode.speed_data
        self.pathArray = self.subscribeNode.path_data
        self.currentPose  = self.subscribeNode.pose_data
     
        #get udpated observations from odometry
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            if self.counter ==1:
                self.subscribeNode.get_logger().info("Running New Episode")
            
            #find the pose of the target in the global frame
            self._findRelativeGoal()

            #this will take the path array that is given and convert the poses in the array into an array of x,y coordinates
            self._convertPathArray()

            #process all the Lidar observations and update lidar array of historical observations
            self.closestObstacle = min(self.scan_data.ranges)  #find the closest obstacle
            self.obstacleAngle = self.scan_data.ranges.index(self.closestObstacle) * 0.5625

            self._roundLidar()
            self._updateLidar()

            self.lastDistanceToTarget = self.newDistanceToTarget
            self.newDistanceToTarget = self._getDistance()

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
                'heading_error': self.pathAngle,
                'relative_goal': self.relativeGoal,
                'global_path': self.pathArrayConverted,
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
        # self.subscribeNode.get_logger().info(f"Step function execution time: {execution_time:.6f} seconds")
        return observation, self.reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        #reset the costmaps
        self.publishNode.clear_local_costmap()
        self.publishNode.clear_global_costmap()

        #check if we have collided with an obstacle if so then run the backup and spin
        while self.collision:
            # self._backup_and_spin()
            if self.obstacleAngle >=90 and self.obstacleAngle <= 270:
                self.publishNode.sendAction(-5.0, 0.0)
            else:
                self.publishNode.sendAction(5.0, 0.0)
            time.sleep(2)
            #reset variables
            rclpy.spin_once(self.subscribeNode, timeout_sec=1.0)
            self.scan_data = self.subscribeNode.scan_data
            #get udpated observations from odometry
            if (self.scan_data):
                if min(self.scan_data.ranges) > 1:
                    self.collision = False
                    self.subscribeNode.get_logger().info("Obstacle Not in Range anymore")
                else:
                    self.subscribeNode.get_logger().info(f"Still in collision zone!!!!!! New closest: {min(self.scan_data.ranges)}")
                
        #reset variables
        self._initialise()
        self.publishNode.sendAction(0.0, 0.0)

     
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
        if ( self.scan_data and self.speed_twist and self.currentPose and self.pathArray):
            self._findRelativeGoal()
            self._roundLidar()
            self._updateLidar()
            self.newDistanceToTarget = self._getDistance()
            self.linearVelocity= round(self.speed_twist.linear.x, 2) 
            self.angularVelocity = round(self.speed_twist.angular.z,2)
            if len(self.pathArray) > self.lookAheadDist:
                self.pathAngle = self._calculate_heading_angle(self.currentPose, self.pathArray[self.lookAheadDist].pose)
            else:
                self.pathAngle = self._calculate_heading_angle(self.currentPose, self.pathArray[-1].pose)
            self._convertPathArray()
            self.subscribeNode.get_logger().info(f"{self.linearVelocity} ,{self.angularVelocity} {self.pathAngle}, {self.relativeGoal}")
            observation = {
                'lidar': self.lidarTracking,
                'linear_velocity': self.linearVelocity,
                'angular_velocity': self.angularVelocity,
                'heading_error': self.pathAngle,
                'relative_goal': self.relativeGoal,
                'global_path': self.pathArrayConverted,
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

    def _convertPathArray(self):
        self.pathArrayConverted = np.zeros((400, 2), dtype=np.float32)  
        path_length = min(len(self.pathArray), 400) 
        for i in range(path_length):
            self.pathArrayConverted[i] = [self.pathArray[i].pose.position.x, self.pathArray[i].pose.position.y]
        

    def _calculateReward(self):
        # Coefficients for each reward component
        alpha = -0.5  # Penalty for deviation from the path (heading angle)
        beta = 3.0    # Reward for reducing distance to the goal
        gamma = -0.3  # Penalty for proximity to obstacles
        roh = 0.5     # Reward for maintaining linear speed
        mu = -0.1     # Penalty for high angular velocity
        time_penalty = -0.005  # Small penalty per time step
        goal_reached_bonus = 100  # Large bonus for reaching the goal
        collision_penalty = -50  # High penalty for collisions

        # Base reward
        reward = 0

        # Distance to the goal reward
        if self.lastDistanceToTarget is not None:
            distance_reward = (self.lastDistanceToTarget - self.newDistanceToTarget)
            reward += beta * distance_reward

        # Heading alignment reward (0 when aligned, pi when opposite)
        heading_reward = self.pathAngle
        reward += alpha * heading_reward

        # Penalty for being too close to obstacles
        if self.closestObstacle < 1.0:
            obstacle_penalty = (1 / self.closestObstacle)  # Higher penalty the closer the obstacle
            reward += gamma * obstacle_penalty

        # Reward for maintaining a reasonable linear velocity
        reward += roh * self.linearVelocity

        # Penalty for excessive angular velocity
        reward += mu * abs(self.angularVelocity)

        # Add a small time penalty to encourage quicker task completion
        # reward += self.counter * time_penalty

        # Check for terminal conditions and apply appropriate rewards/penalties
        if self.newDistanceToTarget < 0.5:  # Goal reached
            reward += goal_reached_bonus
            self.subscribeNode.get_logger().info("Goal reached!")
        elif self.closestObstacle < 0.5 and self.obstacleAngle >= 90 and self.obstacleAngle <= 270 :  # Collision with obstacle
            reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")
        elif self.closestObstacle < 0.65 and self.obstacleAngle < 90 and self.obstacleAngle > 270 :  # Collision with obstacle
            reward += collision_penalty
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")

        self.reward = reward
        self.subscribeNode.get_logger().info(f"obs: {self.closestObstacle}, heading: {self.pathAngle}, dist: {self.newDistanceToTarget}, vel: {self.linearVelocity}")
        self.subscribeNode.get_logger().info(f"The reward is {self.reward}")
        return reward

    def _checkTerminalConditions(self): 
        if self.newDistanceToTarget < 0.5:
            self.reward = 1
            return True
        elif self.closestObstacle < 0.5:
            self.collision = True
            self.subscribeNode.get_logger().info("TERMINATED - COLLISION WITH OBSTACLE")
            # self.reward = -1
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
        return (self.relativeGoal[0]**2 + self.relativeGoal[1]**2)**0.5

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
        heading_angle = abs((heading_angle + m.pi) % (2 * m.pi) - m.pi)

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

    def _backup_and_spin(self):
        """
        Backs up and spins the robot 180 degrees if a collision is detected.
        """
        # Send backup action
        self.publishNode.send_backup_goal() 

        # Send spin action
        self.publishNode.send_spin_goal()  # Spin 180 degrees

        self.subscribeNode.get_logger().info("Executed backup and spin recovery maneuver")
    
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
        
        # Transform the displacement vector into the robot's local frame
        x_relative = dx * m.cos(yaw_robot) + dy * m.sin(yaw_robot)
        y_relative = -dx * m.sin(yaw_robot) + dy * m.cos(yaw_robot)
        
        self.relativeGoal =  np.array([x_relative, y_relative]).astype(float)






