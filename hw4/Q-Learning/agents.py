from constants import *
import numpy as np
import random
import copy

class GridWorldAgent:
    def __init__(self):
        # q value estimate
        # self.Q = np.random.z(0, 1, (NUM_STATES, NUM_ACTIONS))
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    def learn(self, reward, current_state, current_action, next_state, step_size=0.1):

        self._update_q_estimate(reward, current_state, current_action, next_state, step_size=step_size)
        
        return self.Q[current_state, current_action]

    def sample_action(self, state, method='eps_greedy', hyperparams={}):
        if method == 'eps_greedy':
            eps = hyperparams.get('eps', 0.4)
            action = self._eps_greedy_action(self.Q[state], eps=eps)
            return action
        elif method == 'greedy':
            action = self._greedy_action(self.Q[state])
            return action
        else:
            print(f"Method {method} not defined")
            return None

    def _update_q_estimate(self, reward, current_state, current_action, next_state, step_size):
        # update rule
        cur_q = self.Q[current_state, current_action]
        self.Q[current_state, current_action] = cur_q + step_size * (reward + GAMMA * np.max(self.Q[next_state]) - cur_q)

    def _get_coords(self, state):
        i=state//5
        j=state - i*5
        return (i,j)

    def _get_reward(self, state):
        if state in WATER_STATES:
            return -10
        if self._get_coords(state) == (4,4):
            return +10
        return 0

    def _greedy_action(self, qvals):
        return np.argmax(qvals)

    def _eps_greedy_action(self, qvals, eps):
        if np.random.uniform(0,1,1) >= eps:
            return np.argmax(qvals)
        return random.choice(range(len(qvals)))