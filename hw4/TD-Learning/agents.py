from constants import *
import numpy as np
import random
import copy


class GridWorldAgent:
    def __init__(self):

        # q value estimate
        self.V = np.random.rand(NUM_STATES,) * 4
        # self.V = np.zeros((NUM_STATES,))
        self.actions = [
            RIGHT, RIGHT, RIGHT, DOWN, DOWN,
            RIGHT, RIGHT, RIGHT, DOWN, DOWN,
            UP, UP, RIGHT, DOWN, DOWN,
            UP, UP, RIGHT, DOWN, DOWN,
            UP, UP, RIGHT, RIGHT, RIGHT
        ]

        self.P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))


    def learn(self, reward, current_state, current_action, next_state, step_size):
        # update v estimate
        self.V[current_state] = self.V[current_state] + step_size * (reward + GAMMA*self.V[next_state] - self.V[current_state])
        self.P[current_state][current_action][next_state] += 1

    def get_action(self, state):
        return self.actions[state]


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