import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import GridWorldAgent
from environments import GridWorldEnv
from constants import *
from tqdm import tqdm
import pickle

agent = GridWorldAgent()
env = GridWorldEnv()

logfile=open("logs.txt", "w")

def log(text):
    logfile.write(text)

eps = 0.8

ideal_V = [
    4.0187, 4.5548, 5.1575, 5.8336, 6.4553,
    4.3716, 5.0324, 5.8013, 6.6473, 7.3907,
    3.8672, 4.3900, 0.0, 7.5769, 8.4637,
    3.4182, 3.8319, 0.0, 8.5738, 9.6946,
    2.9977, 2.9309, 6.0733, 9.6946, 0.0000
]

QVals = np.zeros((20, 25, 4))
error_vals = np.zeros((20, 10000))

actions_completed = np.zeros((20, 10000))


def compute_V(Q, eps):
    optimal_actions = np.argmax(Q, axis=1)
    optimal_actions= np.eye(4)[optimal_actions]
    policy = np.where(optimal_actions == 1, 1-eps+eps/4, eps/4)
    V = np.sum(Q * policy, axis=1)
    return V

rewards_over_time = []

for t in range(20):
    for i in tqdm(range(10000)):

        env.reset()
        actions_completed_list = []
        error_in_episodes = []
        a=0
        r=0
        while True:
            current_state = env.current_state
            current_action = agent.sample_action(current_state, hyperparams={'eps': eps})
            log(f"Timestep {i}, Current state is {current_state} Action : {current_action}\n")
            reward, next_state, done = env.step(current_action)
            if done: 
                break
            a+=1
            r+=reward
            agent.learn(reward, current_state, current_action, next_state)
            agent.learn_offline()

        V = compute_V(agent.Q, 0.1)
        mse = np.sum((V - ideal_V)**2)/25
        # print(mse)
        error_in_episodes.append(mse)
        actions_completed_list.append(a)
        rewards_over_time.append(r)

    error_vals[t] = np.array(error_in_episodes)
    actions_completed[t] = np.array(actions_completed_list)
    QVals[t] = np.array(agent.Q)

    if i % 1000 == 0:
        eps = eps/2


with open('agent.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(agent, file)

# Open a file in binary write mode
with open('error_mse.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(error_vals, file)

# Open a file in binary write mode
with open('qvals.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(QVals, file)

# Open a file in binary write mode
with open('actions_completed.pickle', 'wb') as file:
    # Serialize the object and save it to the file
    pickle.dump(actions_completed, file)

for i in range(5):
    for j in range(5):
        state = i*5 + j
        if state == 24:
            print("T", end=' ')
        elif state in OBSTACLE_STATES:
            print("O", end=' ')
        else:
            action = agent.sample_action(state, method='greedy')
            print(action_img[action], end=' ')
    print('\n')


