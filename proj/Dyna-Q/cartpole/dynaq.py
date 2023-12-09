import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import CartPoleAgent
from constants import *
from tqdm import tqdm
import gym
import copy
import pickle

agent = CartPoleAgent()
env = gym.make("CartPole-v1")

rewards_over_time = []
actions_completed = []
a=0

eps = 0.4

for i in tqdm(range(500)):

    state, _ = env.reset()
    
    j = 0
    r = 0
    while True:
        j+=1
        action = agent.sample_action(state.tolist(), hyperparams={'eps': eps})
        next_state, reward, truncated, terminated, info = env.step(action)
        if truncated or terminated: 
            break
        agent.learn(reward, state.tolist(), action, next_state.tolist())
        agent.learn_offline()
        a+=1
        r+=1
        state=next_state

    if i % 100 == 0:
        eps = eps/2
        

    actions_completed.append(a)
    rewards_over_time.append(r)

    if i%100 == 0:
        print(f"Reward obtained :{r}")
    
state, _ = env.reset()
total_reward = 0
initial_state = copy.deepcopy(state)

for i in range(500):
    action = agent.sample_action(state.tolist(), method='greedy')
    print(f"Action is {action}")
    next_state, reward, truncated, terminated, info = env.step(action)
    if truncated or terminated: 
        if truncated:
            print("Truncated")
        break
    total_reward += reward

# print(agent._get_tile_aggregate(initial_state))
print(f"Total reward is {total_reward}")

# Open a file in binary write mode
with open('agent_2.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(agent, file)

# Open a file in binary write mode
with open('reward_history_2.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(rewards_over_time, file)
