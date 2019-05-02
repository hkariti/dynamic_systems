def get_action(state1, state2):
    actions = state1.get_actions()
    for a in actions:
        candidate_state = state1.apply_action(a)
        if candidate_state == state2:
            return a
    assert False, 'disconnected states' # This shouldn't happen

def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    current_state = goal_state
    while prev[current_state.to_string()]:
        prev_state = current_state
        current_state = prev[prev_state.to_string()]
        action = get_action(current_state, prev_state)
        result.append((current_state, action))
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
