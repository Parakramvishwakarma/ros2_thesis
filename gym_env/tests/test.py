import gymnasium as gym
from gym_env.custom_gym_env import CustomGymnasiumEnv
from stable_baselines3.common.env_checker import check_env


def main():
    env = CustomGymnasiumEnv()
    print("The environment is good for SB3", check_env(env))
    obs, _ = env.reset()

    while True:  # Run for 100 steps
        action = env.action_space.sample()  # Take random action
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {terminated}")
        # print(obs)
        # print()
        # print(f'MINUMUM range is: {min(obs)}')
        if terminated:
            break

    env.close()

if __name__ == '__main__':
    main()
