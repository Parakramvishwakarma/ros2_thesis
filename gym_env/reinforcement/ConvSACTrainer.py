import numpy as np
import os
import torch
import gym
from gym_env.custom_gym_env_nav2_controller import CustomGymnasiumEnvNav2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for plotting
import matplotlib.pyplot as plt
import time
from gym_env.networks import MLPActorCritic
from gym_env.sac import sac
from spinup.utils.run_utils import setup_logger_kwargs

def plot_results(log_folder, title="Learning Curve"):
    """
    Plot the results.
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    filePath = os.path.join(log_folder, 'progress.txt')
    print("infunction filePath", filePath)
    
    try:
        data = np.loadtxt(filePath, delimiter='\t', skiprows=1, usecols=(0, 1), unpack=True)
        x, y = data[0], data[1]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(title)
        plt.savefig(f'./graphs/{title}.png')
        print("Graph Saved")
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    # Parameters
    lr = 0.0001

    # Create the custom environment
    env = CustomGymnasiumEnvNav2()

    # Create directories for logging and saving
    path = os.getcwd()
    parent = os.path.dirname(path)
    log_dir = os.path.join(parent, "/tmp/gym/")
    os.makedirs(log_dir, exist_ok=True)

    # Set up logger
    logger_kwargs = setup_logger_kwargs("SAC_Nav2_2", log_dir, 0)
    #this is where the results are
    results_dir = logger_kwargs["output_dir"]

    # # Train the model using Spinning Up's SAC
    sac(env_fn=lambda: CustomGymnasiumEnvNav2(),logger_kwargs=logger_kwargs)
    
    # Plot the results    
    plot_results(results_dir, title=f"SAC Training Curve, lr={lr}")


if __name__ == "__main__":
    main()