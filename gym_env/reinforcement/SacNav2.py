from gym_env.custom_gym_env_nav2_controller import CustomGymnasiumEnvNav2
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

def plot_results(log_folder, title="Learning Curve"):
    """
    Plot the results.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    
    # print(x, y)
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title)
    plt.savefig('./graphs/SAC_results_PLEASE.png')
    print("Graph Saved")
    plt.show()


path = os.getcwd()
parent = os.path.dirname(path)
log_dir = "/tmp/gym/"
path = os.path.join(parent, log_dir)
# print("path", path)
os.makedirs(path, exist_ok=True)

lr = 0.00005

env = CustomGymnasiumEnvNav2()
print(env.observation_space.shape)
print(env.action_space.shape)

env = Monitor(env, log_dir)

model = SAC("MultiInputPolicy", env, learning_rate=lr, batch_size=512, ent_coef='auto_0.1', verbose=1)
print(model.policy)
#learn the model
model.learn(total_timesteps=400000, log_interval=10)
#save learnt model
model.save(f"./models/SAC_trained_nav2_{lr}_Please")

# model.save_replay_buffer(f"./replay/sac_replay_buffer_{lr}")

policy = model.policy
policy.save(f"./policy/sac_{lr}")

# #get training results and save to csv
df = load_results(log_dir)
# print(f"There are {len(df)} results")
df.to_csv(f"./results/SAC_training_results_06_10_lr_{lr}_epLen_{4000}_please.csv", index=False)
print("Training Results Written")

#plot training results
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "SAC Results")
plot_results(log_dir, f"Learning Curve, lr={lr} and episode length = {4000}")





