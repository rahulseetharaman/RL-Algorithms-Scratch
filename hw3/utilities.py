from constants import *

def get_next_states(state):
    i = state//5
    j = state - 5*i
    
    up = (i-1)*5 + j if i > 0 else -1
    down = (i+1)*5 + j if i < 4 else -1
    left = i*5 + j-1 if j > 0 else -1
    right = i*5 + j+1 if j < 4 else -1
    next_s = [up, down, left, right]
    next_s = [-1 if n in OBSTACLE_STATES else n for n in next_s]
    return next_s

def get_reward(state):
    if state in WATER_STATES:
        return -10
    if get_coords(state) == (4,4):
        return +10
    if state in GOLD_STATES:
        return +5
    return 0

def get_coords(state):
    i=state//5
    j=state - i*5
    return (i,j)


def get_next_steps(state, action):
    next_s = get_next_states(state)
    if action == 0:
        probs = [0.8, 0.0, 0.05, 0.05]
    elif action == 1:
        probs = [0.0, 0.8, 0.05, 0.05]
    elif action == 2:
        probs = [0.05, 0.05, 0.8, 0.0]
    else:
        probs = [0.05, 0.05, 0.0, 0.8]
    state_dist = [0 for _ in range(NUM_STATES)]
    next_coords=dict()
    state_dist[state] = 0.1
    for (s,p) in zip(next_s, probs):
        if s == -1:
            # if it is an obstacle, it will hit it and come back to same state
            state_dist[state] += p
        else:
            state_dist[s] += p
    for i, p in enumerate(state_dist):
        if p > 0:
            next_coords[get_coords(i)] = p
    return state_dist, next_coords
