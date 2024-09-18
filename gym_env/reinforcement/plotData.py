import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv("../data/rewards_data.csv")

# Ensure all data is 1-dimensional numpy arrays and handle potential NaNs or None values
timesteps = data['timesteps'].values
path_angle = data['heading_error'].values
change_distance = data['change_distance'].values
speed = data['linear_velocity'].values
angular_speed = data['angular_speed'].values
closest_obstacle = data['closest_obstacle'].values
reward = data['reward'].values

# Handling NaNs (if any), you can replace NaNs with zeros or another value:
path_angle = np.nan_to_num(path_angle, nan=0.0)
change_distance = np.nan_to_num(change_distance, nan=0.0)
closest_obstacle = np.nan_to_num(closest_obstacle, nan=0.0)
speed = np.nan_to_num(speed, nan=0.0)
angular_speed = np.nan_to_num(angular_speed, nan=0.0)
reward = np.nan_to_num(reward, nan=0.0)

# Plotting
plt.figure(figsize=(20, 8))

# Plot each variable against the timestep number
plt.plot(timesteps, path_angle, label='Path Angle', color='blue')
plt.plot(timesteps, closest_obstacle, label='closest_obstacle', color='red')
plt.plot(timesteps, speed, label='Lin Vel', color='yellow')
plt.plot(timesteps, angular_speed, label='Angular Speed', color='black')
plt.plot(timesteps, reward, label='Reward', color='orange')

# Set labels and title
plt.xlabel('Timestep Number')
plt.ylabel('Values')
plt.title('Path Angle, Goal Angle, Distance to Target, and Reward vs Timestep')
plt.grid(True)
plt.legend()

# Optimize layout and display
plt.tight_layout()
plt.show()
