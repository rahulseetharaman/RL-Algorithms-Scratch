import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from tqdm import tqdm
import sys

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

    agent=Agent(num_actions=num_actions, num_states=num_states, d0=d0, p=p, s_inf=s_inf, gamma=gamma, R=R)

    return agent


agent = init_agent()
num_tries = 250
num_episodes = 100

max_reward = -np.inf
best_policy = None
rewards_obtained = []


def get_random_policy(num_policies):
    '''
        1. To generate a stochastic policy, convert randomly generated matrices using softmax.
        
        policy = np.random.rand(num_policies * (agent.num_states+1), agent.num_actions)
        policy = policy.reshape((num_policies, agent.num_states+1, agent.num_actions))
        policy_probs = np.exp(policy)/np.sum(np.exp(policy), axis=2, keepdims=True)
        
        2. The below snippet generates random deterministic policies - creates identity matrix, 
        and populates policy matrix from that by randomly sampling one hot vectors.
    '''
    eye = np.eye(agent.num_actions)
    rands = np.random.randint(0,2,size=num_policies * (agent.num_states+1))
    policy_probs = eye[rands]
    policy_probs = policy_probs.reshape(num_policies, agent.num_states+1, agent.num_actions)
    return policy_probs

plot_y = []
plot_x = []

plot_x_2 = []
plot_y_2 = []


for i in tqdm(range(1,num_tries+1)):
    policies = get_random_policy(i)
    max_expected_reward = -1
    for j in range(i):
        expected_reward, _, _ = agent.run_simulation(n_episodes=num_episodes, policy=policies[j, :, :])
        if expected_reward > max_expected_reward:
            max_expected_reward = expected_reward
            best_policy = policies[j, :, :]
    plot_x.append(i)
    plot_y.append(max_expected_reward)
    rewards_obtained.append(max_expected_reward)

    if i==1 or i%10 == 0:
        plot_x_2.append(i)
        plot_y_2.append(max_expected_reward)

plt.xlabel("Iterations")
plt.ylabel("Expected Reward")
plt.title("Random deterministic policy generation")
plt.plot(plot_x, plot_y)
plt.show()


plt.xlabel("Iterations")
plt.ylabel("Expected Reward")
plt.title("Random deterministic policy generation")
plt.plot(plot_x_2, plot_y_2, marker='o')
plt.show()