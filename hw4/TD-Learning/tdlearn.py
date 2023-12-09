import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import GridWorldAgent
from environments import GridWorldEnv
from constants import *
from tqdm import tqdm
import pickle
import copy

f=open("log.txt", "w")

def log(f, text):
    f.write(text+'\n')


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


V_sum = np.zeros((NUM_STATES, ))
n_episodes_needed = []

ideal_V = np.array([
    4.0187, 4.5548, 5.1575, 5.8336, 6.4553,
    4.3716, 5.0324, 5.8013, 6.6473, 7.3907,
    3.8672, 4.3900, 0.0, 7.5769, 8.4637,
    3.4182, 3.8319, 0.0, 8.5738, 9.6946,
    2.9977, 2.9309, 6.0733, 9.6946, 0.0000
])


num_trials = 50

for trial in tqdm(range(num_trials)):

    agent = GridWorldAgent()
    env = GridWorldEnv()

    s_counter = np.zeros((25,))
    i = 0
    while True:
        env.reset()
        s_counter[env.current_state]+=1
        prev_V = copy.deepcopy(agent.V)
        start_state = env.current_state
        # start episode
        step_size = 0.1
        while True:
            current_state = env.current_state
            current_action = agent.get_action(current_state)
            reward, next_state, done = env.step(current_action)
            agent.learn(reward, current_state, current_action, next_state, step_size=step_size)
            if done: 
                break
        if np.max(np.abs(prev_V-agent.V)) < DELTA:
            # print(f"Converged in {i+1} episodes, max norm is {np.max(np.abs(agent.V-ideal_V))}")
            break
        i+=1

    # print("d0:")
    # print(s_counter/np.sum(s_counter))
    
    n_episodes_needed.append(i)
    
    V_sum += agent.V

V_obtained = (V_sum/num_trials)
V_obtained[[12, 17, 24]] = 0.0

print("Max Norm: {}".format(np.max(np.abs(V_obtained-ideal_V))))
print(f"Mean episodes needed {np.mean(n_episodes_needed)}")
print(f"Std dev. of episodes needed {np.std(n_episodes_needed)}")

print(V_obtained)
print(ideal_V)

pickle.dump(V_obtained, open("V.pkl", "wb"))
pickle.dump(agent.P, open("transitions.pkl", "wb"))