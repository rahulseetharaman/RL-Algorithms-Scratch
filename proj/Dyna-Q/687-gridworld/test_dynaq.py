import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import GridWorldAgent
from environments import GridWorldEnv
from constants import *
from tqdm import tqdm
import pickle

agent=pickle.load(open('agent.pickle', 'rb'))
QVals = pickle.load(open('qvals.pickle', 'rb'))
actions_completed = pickle.load(open('actions_completed.pickle', 'rb'))
errors = pickle.load(open('error_mse.pickle', 'rb'))

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

learned_V = compute_V(agent.Q, 0.1)
print_V(learned_V)

ideal_V = [
    4.0187, 4.5548, 5.1575, 5.8336, 6.4553,
    4.3716, 5.0324, 5.8013, 6.6473, 7.3907,
    3.8672, 4.3900, 0.0, 7.5769, 8.4637,
    3.4182, 3.8319, 0.0, 8.5738, 9.6946,
    2.9977, 2.9309, 6.0733, 9.6946, 0.0000
]


for i in range(5):
    for j in range(5):
        state = i*5 + j
        if state == 24:
            print("T", end=' ')
        elif state in OBSTACLE_STATES:
            print("O", end=' ')
        else:
            print(learned_V[state], end=' ')
    print('\n')


plt.plot(np.mean(actions_completed, axis=0), list(range(500)))
plt.xlabel("Actions completed")
plt.ylabel("Number of episodes")
plt.show()
plt.clf()

print(errors)

plt.plot(list(range(500)), np.mean(errors, axis=0))
plt.xlabel("Episodes")
plt.ylabel("MSE error")
plt.show()
