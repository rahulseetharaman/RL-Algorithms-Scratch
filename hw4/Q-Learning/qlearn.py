import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import GridWorldAgent
from environments import GridWorldEnv
from constants import *
from tqdm import tqdm


agent = GridWorldAgent()
env = GridWorldEnv()

# logfile=open("logs.txt", "w")

# def log(text):
#     logfile.write(text)

num_episodes = 10000
actions_taken_arr = np.zeros((20, num_episodes))


ideal_V = [
    4.0187, 4.5548, 5.1575, 5.8336, 6.4553,
    4.3716, 5.0324, 5.8013, 6.6473, 7.3907,
    3.8672, 4.3900, 0.0, 7.5769, 8.4637,
    3.4182, 3.8319, 0.0, 8.5738, 9.6946,
    2.9977, 2.9309, 6.0733, 9.6946, 0.0000
]

errors_all_trials = np.zeros((20, 10000))
all_Q = np.zeros((20, 25, 4))

for trial in range(20):

    error_vals = []
    agent = GridWorldAgent()
    actions_taken = 0
    eps = 0.8

    for i in tqdm(range(num_episodes)):
        env.reset()
        
        while True:
            current_state = env.current_state
            current_action = agent.sample_action(current_state, hyperparams={'eps': eps})
            actions_taken += 1
            # log(f"Timestep {i}, Current state is {current_state} Action : {current_action}\n")
            reward, next_state, done = env.step(current_action)
            if done: 
                break
            agent.learn(reward, current_state, current_action, next_state, step_size=0.9)

        if i % 1000 == 0:
            eps = eps/2

        actions_taken_arr[trial][i] = actions_taken
        qvals = np.max(agent.Q, axis=1)
        error=np.sum((qvals-ideal_V)**2)/25
        error_vals.append(error)

    errors_all_trials[trial] = np.array(error_vals)
    all_Q[trial]=agent.Q


agent.Q = np.mean(all_Q, axis=0)

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


for i in range(5):
    for j in range(5):
        state = i*5 + j
        if state in OBSTACLE_STATES:
            print("O", end=' ')
        else:
            qval = np.max(agent.Q[state])
            print(qval, end=' ')
    print('\n')


plt.plot(range(errors_all_trials.shape[1]), np.mean(errors_all_trials, axis=0))
plt.ylabel("Mean squared error (MSE)")
plt.xlabel("Episodes")
plt.savefig("qlearn_output_normal_dist_q.png")


plt.clf()

plt.plot(np.mean(actions_taken_arr, axis=0), range(10000))
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.savefig("qlearn_actions.png")