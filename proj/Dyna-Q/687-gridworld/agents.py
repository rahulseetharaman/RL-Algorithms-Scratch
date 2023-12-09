from constants import *
import numpy as np
import random
import copy

class GridWorldAgent:
    def __init__(self):
        # model is analogous to p (transition prob.)
        self.P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
        self.R = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

        # q value estimate
        # self.Q = np.random.normal(0, 5, (NUM_STATES, NUM_ACTIONS))
        self.Q = np.random.rand(NUM_STATES, NUM_ACTIONS)

        self.history = []

        self.rewards = np.zeros((NUM_STATES, ))
        for i in range(NUM_STATES):
            self.rewards[i] = self._get_reward(i)


    def learn(self, reward, current_state, current_action, next_state):

        self._update_q_estimate(reward, current_state, current_action, next_state)
        
        # update count
        self.P[current_state, current_action, next_state] += 1

        # update reward
        self.R[current_state, current_action, next_state] = reward

        # add to history for learning offline later
        self.history.append((current_state, current_action))
        return self.Q[current_state, current_action]

    def learn_offline(self):

        random.shuffle(self.history)
        length = min(NUM_OFFLINE_ITERS, len(self.history))

        for i in range(length):
            state,action = self.history[i]
            probs = copy.deepcopy(self.P[state][action])
            probs/=np.sum(probs)
            next_state = np.random.choice(range(NUM_STATES), p=probs)
            reward = self.R[state][action][next_state]
            self._update_q_estimate(reward, state, action, next_state)

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

    def _update_q_estimate(self, reward, current_state, current_action, next_state):
        # update rule
        cur_q = self.Q[current_state, current_action]
        self.Q[current_state, current_action] = cur_q + STEP_SIZE * (reward + GAMMA * np.max(self.Q[next_state]) - cur_q)

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