import numpy as np

PROBS = np.array([[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2 ,0.2, 0.4], [1, 0, 0, 0]])
LETTER = dict(b=0, k=1, o=2, eow=3)
K = 5

def cost(s, a):
    p = PROBS[LETTER[s], LETTER[a]]
    if p == 0:
        return np.inf
    return -np.log(p)

def new_state(letter, target, state_cost=0):
    if not state_cost:
        state_cost = cost(letter, target)
    return dict(letter=letter, target=target, cost=state_cost)

def find_state(state_list, letter):
    return list(filter(lambda s: s['letter'] == letter, state_list))[0]

def get_states():
    states = []
    states += ([new_state('eow', 'eow')], [new_state('b', 'eow'), new_state('o', 'eow'),
                                           new_state('k', 'eow')])
    for _ in range(1, K):
        state = []
        previous_state = states[-1]
        for letter in LETTER:
            if letter == 'eow':
                continue
            letter_cost = np.inf
            chosen_target = None
            for target in LETTER:
                if target == 'eow':
                    continue
                target_to_end_cost = find_state(previous_state, target)['cost']
                me_to_target_cost = cost(letter, target)
                target_cost = me_to_target_cost + target_to_end_cost
                if target_cost < letter_cost:
                    letter_cost = target_cost
                    chosen_target = target
            state.append(new_state(letter, chosen_target, letter_cost))
        states.append(state)

    return states

def get_word(states):
    first_state = find_state(states[-1], 'b')
    first_letter = first_state['letter']
    next_letter = first_state['target']
    word = [first_letter]
    for s in reversed(states[:-1]):
        state = find_state(s, next_letter)
        word.append(state['letter'])
        next_letter = state['target']

    return word
