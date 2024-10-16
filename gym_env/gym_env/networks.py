import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gym_env.core as core
from torch.distributions.normal import Normal

#These are to be change at some point
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        obs_dim_non_conv =  obs_dim - 3 * 640
         # 1D Convolutional layers for laser scan measurements these should have a dimension of 3x640
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        #second conv layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(32 * 158, 256) 

        #this is where the flattened output of the convolution is passed into after the first FC layer adn otehr obs are added
        self.fc2 = nn.Linear(256 + obs_dim_non_conv, 128)  
        self.fc3 = nn.Linear(128, 128) 
        # Output layers for mean velocities
        self.mu_layer = nn.Linear(128, act_dim)
        self.log_std_layer = nn.Linear(128, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        laser_scan, obs = core.process_observation(obs)
        laser_scan = laser_scan.to(next(self.parameters()).device)
        obs = obs.to(next(self.parameters()).device)

        x1 = F.relu(self.conv1(laser_scan))  # First conv layer
        x2 = F.relu(self.conv2(x1))

        #convolution parses are done now we will run through fully conected layers and then add the other observations  
        x_flat = x2.view(x2.size(0),-1)  # Flatten the output 
        x3 = F.relu(self.fc1(x_flat))
        x4 = torch.cat([x3, obs], dim=1)
        x5 = F.relu(self.fc2(x4))
        x6 = F.relu(self.fc3(x5))
        mu = self.mu_layer(x6)
        log_std = self.log_std_layer(x6)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        obs_dim_non_conv =  obs_dim - 3 * 640
        # 1D Convolutional layers for laser scan measurements these should have a dimension of 3x640
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        #second conv layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(32 * 158, 256) #not sure if this is right yet

        #this is where the flattened output of the convolution is passed into after the first FC layer adn otehr obs are added
        self.fc2 = nn.Linear(256 + obs_dim_non_conv + act_dim, 128)  
        self.fc3 = nn.Linear(128, 128) 
        # Output layers for mean velocities
        self.q_val = nn.Linear(128, 1)

    def forward(self, obs, act):
        laser_scan, obs = core.process_observation(obs)
        laser_scan = laser_scan.to(next(self.parameters()).device)
        obs = obs.to(next(self.parameters()).device)
        act = act.to(next(self.parameters()).device)
        x1 = F.relu(self.conv1(laser_scan))  # First conv layer
        x2 = F.relu(self.conv2(x1))
        # print(f"Q Network before flatten: {x2.shape}")
        x_flat = x2.view(x2.size(0),-1)  # Flatten the output 
        # print(f"Q Network after flatten: {x_flat.shape}")
        x3 = F.relu(self.fc1(x_flat))
        # print("VALUE FUINCTION BEOFRE APPENDING o, a", x3.shape)
        x4 = torch.cat([x3, obs, act], dim=1)
        # print(f"Q Network after concat: {x4.shape}")
        # print("VALUE FUINCTION AFTER APPENDING o, a", x4.shape)
        x5 = F.relu(self.fc2(x4))
        x6 = F.relu(self.fc3(x5))
        q = self.q_val(x6)
        # print(f"output shape: {q.shape}")
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) 
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            a = torch.squeeze(a)
            return a.cpu().numpy()
