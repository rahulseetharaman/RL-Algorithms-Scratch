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

READ_BINARY_MODE = "rb"

agent = pickle.load(open("agent.pickle", READ_BINARY_MODE))
env = gym.make("CartPole-v1")

rewards_over_time = pickle.load(open("reward_history.pickle", READ_BINARY_MODE))

state, _ = env.reset()
total_reward = 0
initial_state = copy.deepcopy(state)


for i in range(500):
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


print(f"Total reward is {total_reward}")
plt.plot(range(len(filtered_total_rewards_over_time)), np.mean(filtered_total_rewards_over_time, axis=0))
plt.show()