from gym_env.custom_gym_env_nav2_controller import CustomGymnasiumEnvNav2
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

path = os.getcwd()
parent = os.path.dirname(path)
log_dir = "/tmp/gym/"
path = os.path.join(parent, log_dir)
# print("path", path)
os.makedirs(path, exist_ok=True)
env = CustomGymnasiumEnvNav2()
env = Monitor(env, log_dir)
model = SAC("MultiInputPolicy", env, verbose=1)
#learn the model
model.learn(total_timesteps=200000, log_interval=10)
#save learnt model
model.save("./models/SAC_trained_nav")



