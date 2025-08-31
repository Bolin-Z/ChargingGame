# from src.env import EVCSGameEnv
# import logging

# logging.basicConfig(level=logging.INFO)

# env = EVCSGameEnv("./siouxfalls", "siouxfalls")

import gymnasium as gym

env = gym.make("CartPole-v1")
env.action_space.n