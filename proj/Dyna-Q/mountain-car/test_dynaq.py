import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import MountainCarAgent
from constants import *
from tqdm import tqdm
import gym
import copy
import pickle
import matplotlib.pyplot as plt
import random

READ_BINARY_MODE = "rb"

agent = pickle.load(open("agent_600.pickle", READ_BINARY_MODE))
env = gym.make("MountainCar-v0", render_mode='human')

rewards_over_time = pickle.load(open("rewards.pkl", READ_BINARY_MODE))

state, _ = env.reset()
total_reward = 0
initial_state = copy.deepcopy(state)
while True:
    action = agent.sample_action(state.tolist(), method='greedy')
    next_state, reward, truncated, terminated, info = env.step(action)
    if truncated or terminated: 
        if truncated:
            print("Truncated")
        break
    total_reward += reward
    state = next_state

filtered_total_rewards_over_time = []
for i, r in enumerate(rewards_over_time):
    if i % 50  == 0:
        filtered_total_rewards_over_time.append(r)


