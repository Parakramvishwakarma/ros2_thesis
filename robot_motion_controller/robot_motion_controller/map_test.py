import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped

class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')
        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.map_callback,
            10)
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10)
        self.subscription_plan = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10)

        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.path_poses = []

    def map_callback(self, msg):
        # Get map information
        self.map_info = msg.info
        width = msg.info.width
        height = msg.info.height

        # Convert the map data to a numpy array
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Replace -1 (unknown) values with a different value for better visualization
        self.map_data = np.where(self.map_data == -1, 50, self.map_data)

        # Plot the map, robot pose, and path if available
        self.plot_map()

    def pose_callback(self, msg):
        # Get the robot's position in the map frame
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Plot the map, robot pose, and path if available
        self.plot_map()

    def plan_callback(self, msg):
        # Clear the existing path poses
        self.path_poses = []

        # Extract the sequence of poses from the /plan topic
        for pose_stamped in msg.poses:
            self.path_poses.append((pose_stamped.pose.position.x, pose_stamped.pose.position.y))

        # Plot the map, robot pose, and path if available
        self.plot_map()

    def plot_map(self):
        if self.map_data is None or self.map_info is None:
            # Wait until map data is available
            return

        # Calculate the robot's position in grid coordinates
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        # Make a copy of the map data to modify for visualization
        display_map = self.map_data.copy()

        # Convert robot's position (meters) to grid cells
        if self.robot_pose:
            grid_x = int((self.robot_pose[0] - origin_x) / resolution)
            grid_y = int((self.robot_pose[1] - origin_y) / resolution)
            # Change the map value at the robot's position to a specific value (e.g., 100 for visualization)
            padding_size = 3  # Number of cells to pad around the robot
            for dx in range(-padding_size, padding_size + 1):
                for dy in range(-padding_size, padding_size + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < display_map.shape[1] and 0 <= ny < display_map.shape[0]:
                        display_map[ny, nx] = 100  # Mark the robot's position area on the map

        # Mark the path on the map
        for (pose_x, pose_y) in self.path_poses:
            path_grid_x = int((pose_x - origin_x) / resolution)
            path_grid_y = int((pose_y - origin_y) / resolution)

            # Update the map value for the path
            if 0 <= path_grid_x < display_map.shape[1] and 0 <= path_grid_y < display_map.shape[0]:
                display_map[path_grid_y, path_grid_x] = 75  # Mark the path on the map with a different value

        # Visualize the map using matplotlib
        plt.imshow(display_map, origin='lower')

        # Display the plot
        plt.title('Occupancy Grid Map with Robot Position and Path')
        plt.xlabel('Width (cells)')
        plt.ylabel('Height (cells)')
        plt.colorbar(label='Occupancy Value')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    map_visualizer = MapVisualizer()

    try:
        rclpy.spin(map_visualizer)
    except KeyboardInterrupt:
        pass

    map_visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
