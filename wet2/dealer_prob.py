#!//usr/bin/python

import random

NUM_EXPERIMENTS = 100000

def get_card():
    card = random.randint(1, 13)
    if card >= 10:
        return 10
    elif card == 1:
        return 11
    else:
        return card

def calc_prob(begin_value):
    results = {}
    for i in range(NUM_EXPERIMENTS):
        sum1 = begin_value + get_card()
        hits = 0
        while sum1 < 17:
            if sum1 > 21:
                break;
            sum1 += get_card()
            hits += 1
        if sum1 > 22:
            sum1 = 22
        if sum1 in results:
            results[sum1] += 1.0/NUM_EXPERIMENTS
        else:
            results[sum1] = 0

    return results

if __name__ == '__main__':
    for i in range(2, 12):
        res = calc_prob(i)
        print(i, res, sum(res.values()))
