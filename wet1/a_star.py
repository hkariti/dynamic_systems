from puzzle import *
from planning_utils import *
import heapq
import datetime
import numpy as np


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    alpha = 9999 # The weight of the heuristic function in the cost
    expanded_states = 0
    while len(fringe) > 0:
        expanded_states += 1
        cost, state = heapq.heappop(fringe)
        state_name = state.to_string()
        if state == goal:
            break
        actions = state.get_actions()
        for a in actions:
            new_state = state.apply_action(a)
            new_state_name = new_state.to_string()
            current_distance = distances.get(new_state_name, np.inf)
            # Each neighboring states have a distance of 1
            new_distance = distances[state_name] + 1
            if current_distance > new_distance:
                distances[new_state_name] = new_distance
                prev[new_state_name] = state
                heuristic_distance = new_state.get_manhattan_distance(goal)
                total_distance = new_distance + alpha*heuristic_distance
                heapq.heappush(fringe, (total_distance, new_state))
        concluded.add(state_name)
    print "Expanded {} states".format(expanded_states)
    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u',
    ]
    # Hard puzzle: solution is 25 steps long
    hard_actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u', 'l', 'd', 'd', 'r',
        'l', 'r', 'u', 'l', 'd', 'r', 'r',
        'u'
    ]
    goal_state = initial_state
    puzzle_actions = hard_actions
    for a in puzzle_actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(puzzle_actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
