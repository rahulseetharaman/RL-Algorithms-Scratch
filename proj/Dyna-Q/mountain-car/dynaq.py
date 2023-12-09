import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import MountainCarAgent
from constants import *
from tqdm import tqdm
import gym
import copy
import pickle

agent = MountainCarAgent()
env = gym.make("MountainCar-v0")

rewards_over_time = []
eps = 0.4

for i in tqdm(range(600)):

    state, _ = env.reset()
    
    j = 0
    r = 0
    while True:
        j+=1
        action = agent.sample_action(state.tolist(), hyperparams={'eps': eps})
        # print(action)
        next_state, reward, truncated, terminated, info = env.step(action)
        if truncated or terminated: 
            break
        agent.learn(reward, state.tolist(), action, next_state.tolist())
        agent.learn_offline()
        # print(reward)
        state=next_state
        r+=reward
        

    if i % 100 == 0:
        eps = eps/2
        
    # print(r)
    rewards_over_time.append(r)

    if i%100 == 0:
        print(f"Reward obtained :{r}")
        with open(f'agent_2{i}.pickle', 'wb') as file:
            # Serialize the object and save it to the file
            pickle.dump(agent, file)

pickle.dump(rewards_over_time, open("rewards.pkl", "wb"))