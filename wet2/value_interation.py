#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dealer_prob

DEALER_PROBS = {}

A_STICK = 0
A_HIT = 1

def calculate_dealer_probs():
    for i in range(2, 12):
        DEALER_PROBS[i] = dealer_prob.calc_prob(i)

def get_hit_prob(start_state, end_state):
    (start_x, start_y) = start_state
    (end_x, end_y) = end_state

    if start_y != end_y:
        return 0

    diff = end_x - start_x
    if diff in [2, 3, 4, 5, 6, 7, 8, 9, 11]:
        return 1.0/13
    if diff == 10:
        return 4.0/13
    return 0

def get_stick_prob(start_state, end_state):
    (start_x, start_y) = start_state
    (end_x, end_y) = end_state

    if start_x == end_x and start_y == end_y:
        return 1
    return 0

def reward(state):
    player_sum, dealer_initial = state
    if player_sum > 21:
        return -1
    if not DEALER_PROBS:
        calculate_dealer_probs()
    dealer_final_prob = DEALER_PROBS[dealer_initial]
    reward = 0
    for (dealer_sum, p) in dealer_final_prob.items():
        if dealer_sum < player_sum or dealer_sum > 21:
            reward += p
        elif dealer_sum > player_sum:
            reward -= p
    return reward

def get_initial_state():
    state_vec = {}
    for x in range(4, 33):
        for y in range(2, 12):
            state_vec[(x,y)] = 0
    return state_vec

def value_iteration(current_state_vec):
    policy = {}
    new_state_vec = {}
    for current_state, current_value in current_state_vec.items():
        stick_value = reward(current_state) + current_value
        hit_value = 0
        for state, value in current_state_vec.items():
            hit_value += get_hit_prob(current_state, state)*(reward(state) + value)
        if hit_value > stick_value and current_state[0] <= 21:
            new_state_vec[current_state] = hit_value
            policy[current_state] = A_HIT
        else:
            new_state_vec[current_state] = stick_value
            policy[current_state] = A_STICK

    return new_state_vec, policy

def find_best_policy():
    state_vec = get_initial_state()
    for n in range(100):
        state_vec, policy = value_iteration(state_vec)
    return state_vec, policy

def plot_value_vec(value_vec):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(4, 22), range(2, 12))
    value_array = np.ndarray((18, 10), dtype=float)
    for state, value in value_vec.items():
        x, y = state
        if x > 21:
            continue
        value_array[x-4, y-2] = value
    ax.plot_wireframe(X, Y, value_array.T)
    ax.set_ylabel('Dealer showing')
    ax.set_xlabel('Player sum')
    ax.set_zlabel('Value')
    ax.set_title('Optimal value for each state')
    plt.show()

def plot_policy(policy):
    policy_array = np.ndarray((18, 10), dtype=float)
    for state, action in policy.items():
        x, y = state
        if x > 21:
            continue
        policy_array[x-4, y-2] = action
    fig = plt.figure()
    plt.imshow(policy_array, aspect=0.5, extent=(2, 11, 21, 4), cmap='Pastel1')
    plt.ylabel('Player sum')
    plt.xlabel('Dealer showing')
    plt.title('Optimal Policy')
    plt.gca().invert_yaxis()
    plt.text(6, 17, 'Stick')
    plt.text(6, 8, 'Hit')
    plt.show()
    
if __name__ == '__main__':
    value_vec, policy = find_best_policy()
    plot_value_vec(value_vec)
    plot_policy(policy)
