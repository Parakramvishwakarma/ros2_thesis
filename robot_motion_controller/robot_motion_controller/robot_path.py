import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

def plot_robot_paths(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize plot
    plt.figure(figsize=(12, 8))

    # Get a colormap
    colormap = cm.get_cmap('tab20')
    
    # Initialize variables
    current_goal = None
    path_x = []
    path_y = []
    iteration_count = 0
    start_x = None
    start_y = None

    for index, row in df.iterrows():
        if row['type'] == 'goal':
            # If we have a current goal, plot the previous path
            if current_goal is not None:
                color = colormap(iteration_count % 20)  # Use modulo to cycle colors if more than 20 iterations
                plt.plot(path_x, path_y, label=f"Iteration {iteration_count}", color=color, linewidth=2)  # Thicker line
                plt.scatter(current_goal[1], current_goal[2], color=color, marker='X', s=100)  # Goal marker
                plt.scatter(start_x, start_y, color=color, s=100)  # Start point marker
                path_x.clear()
                path_y.clear()
                iteration_count += 1
            
            # Update current goal
            current_goal = (index, row['x'], row['y'])
            start_x = None
            start_y = None
        elif row['type'] == 'position':
            # Append the position to the current path
            path_x.append(row['x'])
            path_y.append(row['y'])
            # Set the start point
            if start_x is None and start_y is None:
                start_x = row['x']
                start_y = row['y']

    # Plot the last path if exists
    if current_goal is not None:
        color = colormap(iteration_count % 20)
        plt.plot(path_x, path_y, label=f"Iteration {iteration_count}", color=color, linewidth=2)  # Thicker line
        plt.scatter(current_goal[1], current_goal[2], color=color, marker='X', s=100)  # Goal marker
        plt.scatter(start_x, start_y, color=color, s=100)  # Start point marker
    
    # Plot obstacles
    obstacles = [(5, 0), (0, 5), (0, -5), (10, -5), (10, 5)]
    for obs in obstacles:
        square = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
        plt.gca().add_patch(square)

    # Set axis limits
    plt.xlim(-5, 15)
    plt.ylim(-10, 10)

    # Labeling the plot
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Paths to Goals')
    
    # Place legend outside the plot area
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.subplots_adjust(right=0.8)  # Adjust the right boundary of the plot
    plt.grid(True)
    plt.savefig('./path/DDPG_path.png')
    plt.show()
    plt.close()
# Use the function to plot the data
csv_file = './csv/test.csv'  # Replace with your actual CSV file path
plot_robot_paths(csv_file)
