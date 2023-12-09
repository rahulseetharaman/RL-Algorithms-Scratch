import numpy as np

class Agent:
    def __init__(self, num_states, num_actions, p, R, s_inf, gamma, d0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.p=p
        self.R=R
        self.s_inf=s_inf
        self.gamma=gamma
        self.d0=d0


    def run_episode(self, policy):
        cur_state = np.random.choice(np.arange(0,8), p=self.d0)
        reward = 0
        cur_discount = 1
        while True:
            action = np.random.choice(np.arange(1,3), p=policy[cur_state])
            next_state = np.random.choice(np.arange(0,8), p=self.p[cur_state][action])            
            reward += (cur_discount * self.R[cur_state][action])
            cur_state = next_state
            if cur_state in self.s_inf:
                break
            cur_discount = cur_discount * self.gamma
        return reward, policy

    def run_simulation(self, policy, n_episodes=150000):

        mean_reward = 0
        all_rewards = []
        expected_rewards_so_far = []

        for i in range(1, n_episodes+1):
            cur_reward, _ = self.run_episode(policy=policy)
            mean_reward = mean_reward * (i-1)/i + cur_reward/i
            all_rewards.append(cur_reward)
            expected_rewards_so_far.append(mean_reward)

        variance_reward = np.std(all_rewards)**2
        return mean_reward, variance_reward, expected_rewards_so_far

