from constants import *
import numpy as np
import random
import copy
from tiles3 import IHT, tiles


class MountainCarAgent:
    def __init__(self):

        self.num_tilings = int(16 * 16 * 64 * 8)
        self.num_tiles = 8

        # q value estimate
        self.hashtable = IHT(self.num_tilings)
        self.Q = np.zeros((self.num_tilings, 1))

        self.history = []
        self.scaling = [16/1.8, 16/0.14]

    def _scale(self, state_repr):
        new_state_repr = []
        # print(state_repr)
        for s, f in zip(state_repr, self.scaling):
            new_state_repr.append(s*f)
        return new_state_repr

    def learn(self, reward, current_state, current_action, next_state):
        self._update_Q_estimate(reward, current_state, current_action, next_state)
        self.history.append((current_state, current_action, next_state, reward))

    def learn_offline(self):

        random.shuffle(self.history)
        length = min(NUM_OFFLINE_ITERS, len(self.history))

        for i in range(length):
            state, action, next_state, reward = self.history[i]            
            self._update_Q_estimate(reward, state, action, next_state)

    def sample_action(self, state, method='eps_greedy', hyperparams={}):
        if method == 'eps_greedy':
            eps = hyperparams.get('eps', 0.4)
            action = self._eps_greedy_action(state, eps=eps)
            return action
        elif method == 'greedy':
            action = self._greedy_action(state)
            return action
        else:
            print(f"Method {method} not defined")
            return None

    def _within_bounds(self, state):
        [x,v] = state
        if not -1.2 < x < 0.6:
            return False
        elif not -0.07 < v < 0.07:
            return False
        return True
    
    def _tiles(self, state, action=None):
        scaled_state = self._scale(state)
        if action:
            return tiles(self.hashtable, self.num_tiles, scaled_state, ints=[action])
        else:
            return tiles(self.hashtable, self.num_tiles, scaled_state)

    def _get_tile_aggregate(self, state, action=None):
        tiles = self._tiles(state, action)
        estimate = 0
        for tile in tiles:
            estimate += self.Q[tile]
        # print(estimate)
        return estimate

    def _update_Q_table(self, state, action, error):
        tiles = self._tiles(state, action)
        for tile in tiles:
            self.Q[tile] = self.Q[tile] + STEP_SIZE * error

    def _update_Q_estimate(self, reward, current_state, current_action, next_state):
        # update rule
        cur_q = self._get_tile_aggregate(current_state, current_action)
        next_q_left = self._get_tile_aggregate(next_state, 0)
        next_q_still = self._get_tile_aggregate(next_state, 1)
        next_q_right = self._get_tile_aggregate(next_state, 2)
        next_q = max(next_q_left, next_q_still, next_q_right)

        error = reward + GAMMA * next_q - cur_q
        self._update_Q_table(current_state, current_action, error)

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

    def _greedy_action(self, state):
        next_q_left = self._get_tile_aggregate(state, 0)
        next_q_still = self._get_tile_aggregate(state, 1)
        next_q_right = self._get_tile_aggregate(state, 2)
        return np.argmax([next_q_left, next_q_still, next_q_right])

    def _eps_greedy_action(self, state, eps):
        next_q_left = self._get_tile_aggregate(state, 0)
        next_q_still = self._get_tile_aggregate(state, 1)
        next_q_right = self._get_tile_aggregate(state, 2)
        if np.random.uniform(0,1,1) >= eps:
            return np.argmax([next_q_left, next_q_still, next_q_right])
        return random.choice(range(3))