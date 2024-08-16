import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

def plot_robot_paths(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize the figure
    plt.figure(figsize=(50, 30))

    # Get a colormap
    colormap = cm.get_cmap('tab20')

    # Find the total number of iterations
    iterations = df[df['type'] == 'goal'].shape[0]
    print(iterations)
    # num_plots = (iterations) //   # Calculate number of subplots needed
    num_plots = 3

    # Create subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 12))

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    # Initialize variables
    current_goal = None
    path_x = []
    path_y = []
    iteration_count = 0
    start_x = None
    start_y = None
    plot_index = 0

    for index, row in df.iterrows():
        if row['type'] == 'goal':
            # If we have a current goal, plot the previous path
            if current_goal is not None and iteration_count % 20 == 0 or iteration_count == 2:
                color = colormap(iteration_count % 20)  # Use modulo to cycle colors if more than 20 iterations
                axes[plot_index].plot(path_x, path_y, label=f"Iteration {iteration_count}", color=color, linewidth=2)  # Thicker line
                axes[plot_index].scatter(current_goal[1], current_goal[2], color=color, marker='X', s=100, label=f"Goal {iteration_count}")  # Goal marker
                axes[plot_index].scatter(start_x, start_y, color=color, s=100, label=f"Start {iteration_count}")  # Start point marker
                path_x.clear()
                path_y.clear()
                plot_index += 1
            
            # Update current goal
            current_goal = (index, row['x'], row['y'])
            start_x = None
            start_y = None
            iteration_count += 1
        elif row['type'] == 'position':
            # Append the position to the current path
            path_x.append(row['x'])
            path_y.append(row['y'])
            # Set the start point
            if start_x is None and start_y is None:
                start_x = row['x']
                start_y = row['y']

    # Plot the last path if exists
    if current_goal is not None and iteration_count % 20 == 0 or iteration_count == 2:
        color = colormap(iteration_count % 30)
        axes[plot_index].plot(path_x, path_y, label=f"Iteration {iteration_count}", color=color, linewidth=2)  # Thicker line
        axes[plot_index].scatter(current_goal[1], current_goal[2], color=color, marker='X', s=100, label=f"Goal {iteration_count}")  # Goal marker
        axes[plot_index].scatter(start_x, start_y, color=color, s=100, label=f"Start {iteration_count}")  # Start point marker

    # Plot obstacles
    obstacles = [(5, 0), (0, 5), (0, -5), (10, -5), (10, 5)]
    for ax in axes:
        for obs in obstacles:
            square = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(square)
        
        # Set axis limits
        ax.set_xlim(-5, 15)
        ax.set_ylim(-10, 10)
        
        # Ensure square aspect ratio
        ax.set_aspect('equal')

        # Labeling the plot
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Robot Paths to Goals')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
    plt.savefig('./path/DDPG_path_2.png')
    plt.show()
    plt.close()

# Use the function to plot the data
csv_file = './csv/test.csv'  # Replace with your actual CSV file path
plot_robot_paths(csv_file)