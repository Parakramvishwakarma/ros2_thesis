from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import gym_env.core as core
from gym_env.networks import MLPActorCritic
from gym_env.buffer import ReplayBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs


def sac(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=2000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)
    """
    logger = EpochLogger(**logger_kwargs)
    serializable_locals = {key: val for key, val in locals().items() if isinstance(val, (int, float, str, dict, list, tuple, bool, type(None)))}
    logger.save_config(serializable_locals)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Choose the device to run the computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = [torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in ['obs', 'act', 'rew', 'obs2', 'done']]

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = torch.as_tensor(data['obs'], dtype=torch.float32, device=device)
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        logger.store(LossQ=loss_q.item(), **q_info)

        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        o = torch.as_tensor(o, dtype=torch.float32, device=device)
        return ac.act(o, deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o = torch.as_tensor(o, dtype=torch.float32, device=device)
                o, r, d, _ = test_env.step(get_action(o, True).cpu().numpy())
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    logger.log(f"\nStarting Training Loop with {steps_per_epoch} and {epochs} = {total_steps}")

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
