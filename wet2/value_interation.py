#!/usr/bin/python
import dealer_prob

DEALER_PROBS = {}

def calculate_dealer_probs:
    for i in range(2, 12):
        DEALER_PROBS[i] = dealer_prob.calc_prob(i)

def reward(player_sum, dealer_initial):
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


