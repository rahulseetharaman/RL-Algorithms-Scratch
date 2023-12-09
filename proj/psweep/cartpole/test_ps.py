import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import CartPoleAgent
from constants import *
from tqdm import tqdm
import gym
import copy
import pickle
import matplotlib.pyplot as plt
import random

READ_BINARY_MODE = "rb"

agent = pickle.load(open("agent.pickle", READ_BINARY_MODE))
env = gym.make("CartPole-v1")

rewards_over_time = pickle.load(open("reward_history.pickle", READ_BINARY_MODE))

state, _ = env.reset()
env.state = env.unwrapped.state = [0,0,0,0]
total_reward = 0
initial_state = copy.deepcopy(state)

done = False

while not done:
    action = agent.sample_action(state.tolist(), method='greedy')
    next_state, reward, truncated, terminated, info = env.step(action)
    done = truncated or terminated
    total_reward += reward
    state = next_state

