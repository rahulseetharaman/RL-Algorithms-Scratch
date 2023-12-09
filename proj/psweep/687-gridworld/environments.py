from constants import *
import numpy as np
import random

class GridWorldEnv():
    def __init__(self):
        self.num_states = 25
        self.num_actions = 4
        self.current_state = 0
        self.possible_states = [i for i in range(25) if i not in [12,17,24]]

    def step(self, action):
        if self.current_state == 24:
            return 0, self.current_state, True
        
        state_distribution = self._get_next_states(self.current_state, action)
        # sample from state distribution
        next_state = np.random.choice(range(25), p=state_distribution)
        self.current_state = next_state
        reward = self._get_reward(self.current_state)
        return reward, self.current_state, False
    
    def reset(self):
        self.current_state = random.choice(self.possible_states)

    def _get_reward(self, state):
        if state in WATER_STATES:
            return -10
        if self._get_coords(state) == (4,4):
            return +10
        return 0
    
    def _get_coords(self, state):
        i=state//5
        j=state - i*5
        return (i,j)

    def _get_next_states_list(self, state):
        i = state//5
        j = state - 5*i
        
        up = (i-1)*5 + j if i > 0 else -1
        down = (i+1)*5 + j if i < 4 else -1
        left = i*5 + j-1 if j > 0 else -1
        right = i*5 + j+1 if j < 4 else -1
        next_s = [up, down, left, right]
        next_s = [-1 if n in OBSTACLE_STATES else n for n in next_s]
        return next_s        

    def _get_next_states(self, state, action):
        next_s = self._get_next_states_list(state)
        if action == 0:
            probs = [0.8, 0.0, 0.05, 0.05]
        elif action == 1:
            probs = [0.0, 0.8, 0.05, 0.05]
        elif action == 2:
            probs = [0.05, 0.05, 0.8, 0.0]
        else:
            probs = [0.05, 0.05, 0.0, 0.8]
        state_dist = [0 for _ in range(NUM_STATES)]
        state_dist[state] = 0.1
        for (s,p) in zip(next_s, probs):
            if s == -1:
                # if it is an obstacle, it will hit it and come back to same state
                state_dist[state] += p
            else:
                state_dist[s] += p
        return state_dist