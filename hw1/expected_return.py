import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

def init_agent():
    # Define states, actions and transition probabilities.
    num_states = 7
    num_actions = 2
    d0 = [0.0, 0.6, 0.3, 0.1, 0.0 ,0.0 ,0.0, 0.0]
    s_inf = [6,7]

    p = np.zeros((num_states+1, num_actions+1, num_states+1))
    gamma = 0.9

    p[1, 1, 4] = 1.0
    p[2, 1, 4] = 0.8 
    p[1, 2, 4] = 1.0
    p[2, 1, 5] = 0.2
    p[2, 2, 4] = 0.6
    p[2, 2, 5] = 0.4
    p[3, 1, 4] = 0.9 
    p[3, 1, 5] = 0.1
    p[3, 2, 5] = 1.0
    p[4, 1, 6] = 1.0
    p[4, 2, 6] = 0.3 
    p[4, 2, 7] = 0.7
    p[5, 1, 6] = 0.3 
    p[5, 1, 7] = 0.7
    p[5, 2, 7] = 1.0
    

    R = np.zeros((num_states+1, num_actions+1))

    R[1, 1] = 7 
    R[1, 2] = 10
    R[2, 1] = -3 
    R[2, 2] = 5
    R[3, 1] = 4 
    R[3, 2] = -6
    R[4, 1] = 9 
    R[4, 2] = -1
    R[5, 1] = -8 
    R[5, 2] = 2

    agent=Agent(num_actions=num_actions, num_states=num_states, p=p, s_inf=s_inf, gamma=gamma, R=R, d0=d0)

    return agent

pi = np.array([[1.0,0], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.5, 0.5]])


agent = init_agent()

n_episodes = 150000

mean_reward, variance_reward, expected_rewards_so_far = agent.run_simulation(n_episodes=n_episodes, policy=pi)

print(mean_reward)
print(variance_reward)

# plt.title("How E[R] varies with number of simulations")
# plt.xlabel("Training simulations")
# plt.ylabel("Expected reward estimate")
# plt.plot(list(range(1, n_episodes+1)), expected_rewards_so_far)
# plt.show()