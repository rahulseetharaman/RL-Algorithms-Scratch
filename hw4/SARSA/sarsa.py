import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import GridWorldAgent
from environments import GridWorldEnv
from constants import *
from tqdm import tqdm
import sys

agent = GridWorldAgent()
env = GridWorldEnv()

def print_V(V):
    for i in range(5):
        for j in range(5):
            state = i*5 + j
            if state == 24:
                print("T", end=' ')
            elif state in OBSTACLE_STATES:
                print("O", end=' ')
            else:
                print('{:.3f}'.format(V[state]), end=' ')
        print('\n')


def compute_V(Q, eps):
    optimal_actions = np.argmax(Q, axis=1)
    optimal_actions= np.eye(4)[optimal_actions]
    policy = np.where(optimal_actions == 1, 1-eps+eps/4, eps/4)
    V = np.sum(Q * policy, axis=1)
    return V


num_episodes = 10000

QVals = np.zeros((20, 25, 4))
error_vals = np.zeros((20, num_episodes))

ideal_V = [
    4.0187, 4.5548, 5.1575, 5.8336, 6.4553,
    4.3716, 5.0324, 5.8013, 6.6473, 7.3907,
    3.8672, 4.3900, 0.0, 7.5769, 8.4637,
    3.4182, 3.8319, 0.0, 8.5738, 9.6946,
    2.9977, 2.9309, 6.0733, 9.6946, 0.0000
]

actions_completed = np.zeros((20, num_episodes))

for trial in range(20):

    error_in_episodes = []
    eps = 0.8
    timesteps = 0

    agent = GridWorldAgent()

    for i in tqdm(range(num_episodes)):

        # print(i)
        env.reset()
        step_size = 0.1

        current_state = env.current_state
        current_action = agent.sample_action(current_state, hyperparams={'eps': eps})

        j = 0

        while True:
            reward, next_state, done = env.step(current_action)
            if done: 
                break
            next_action = agent.sample_action(next_state, hyperparams={'eps': eps})
            agent.learn(reward, current_state, current_action, next_state, next_action, step_size = step_size)
            current_action = next_action
            current_state = next_state
            j+=1
            timesteps += 1
        
        # print("Q value")
        # print(agent.Q)
        actions_completed[trial][i] = timesteps

        if i % 1000 == 0:
            eps=eps/2
        
        # V = compute_V(agent.Q, eps)
        # V[[12,17,24]] = 0
        V = compute_V(agent.Q, eps)
        # V = np.max(agent.Q, axis=1)
        mse = np.sum((V - ideal_V)**2)/25
        error_in_episodes.append(mse)
    
    error_vals[trial] = np.array(error_in_episodes)

    QVals[trial] = agent.Q



agent.Q = np.mean(QVals, axis=0)
agent.Q[[12,17,24]] = 0.0

# print("Q values are:")
# print(agent.Q)

V = np.max(agent.Q, axis=1)

for i in range(5):
    for j in range(5):
        state = i*5 + j
        if state in OBSTACLE_STATES:
            print("O", end=' ')
        else:
            print(V[state], end=' ')
    print('\n')

print(agent.Q)

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


plt.plot(range(num_episodes), np.mean(error_vals, axis=0))
plt.xlabel("Episodes")
plt.ylabel("Mean squared error (20 indep. trials)")
plt.savefig("sarsa_output.png")

plt.clf()

print(actions_completed[10][200:300])
plt.plot(np.mean(actions_completed, axis=0), range(num_episodes))
plt.xlabel("Timesteps")
plt.ylabel("Number of episodes completed")
plt.savefig("actions_completed.png")

