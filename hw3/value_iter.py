import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities import *

p = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
v = np.zeros((NUM_STATES, ))
policy = np.zeros((NUM_STATES, ))
rewards = np.zeros((NUM_STATES, ))
for i in range(NUM_STATES):
     rewards[i] = get_reward(i)

# Initialize state transition for 687-GridWorld
for s in range(NUM_STATES):
    for a in range(NUM_ACTIONS):
          if s in TERMINAL_STATES:
               p[s][a] = np.zeros(NUM_STATES, )
          else:
            state_dist, next_coords = get_next_steps(s,a)
            p[s][a] = np.array(state_dist)


# Run Value Iteration

iterations = 0

while True:
     old_v = v.copy()
     temp = gamma * old_v + rewards
     prod = p.dot(temp)
     v = np.max(prod, axis=1)
     max_norm_val = np.max(np.abs((v-old_v))) 
     policy = np.argmax(prod, axis=1)
     iterations += 1
     if max_norm_val < DELTA:
          print(f"Value iteration converged after {iterations} iterations. Max norm value: {max_norm_val}")
          break
     
     
policy = policy.reshape(5,5)

for i in range(5):
     for j in range(5):
        # if i*5+j in WATER_STATES:
        #     print('W', end=' ')
        if i*5+j in REWARD_STATES:
            print("G", end= ' ')
        elif i*5+j in OBSTACLE_STATES:
            print("B", end = ' ')
        elif i*5+j in GOLD_STATES:
            print("C", end=' ')
        else: 
            print(action_dir[policy[i][j]], end=' ')

     print('\n')

for i in range(5):
    for j in range(5):
        print("{:.4f}".format(v[i*5+j]), end=' ')
    print('\n')


