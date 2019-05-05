import numpy as np
from cartpole_cont import CartPoleContEnv


def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    Abar = np.matrix([[0, 1, 0, 0],
                      [0, 0, pole_mass/cart_mass * g,  0],
                      [0, 0, 0, 1],
                      [0, 0, g/pole_length * (1 + pole_mass/cart_mass), 0]])
    A = (np.asmatrix(np.identity(4)) + Abar * dt)
    return A


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    Bbar = np.matrix([0, 1/cart_mass, 0, 1/(cart_mass * pole_length)]).T
    B = Bbar * dt
    return B

def calculate_Ps_Ks(A, B, Q, R, horizon):
    P = []
    K = []
    P.append(Q)
    for i in range(1, horizon + 1):
        prev_P = P[i-1]
        inv_part = np.linalg.inv(R + B.T * prev_P * B)
        cur_K = - inv_part * B.T * prev_P * A
        cur_P = Q + A.T * prev_P * A - A.T * prev_P * B * (-cur_K)
        P.append(cur_P)
        K.append(cur_K)
    P.reverse()
    K.reverse()
    return P, K


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    Q = np.matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = np.matrix([0.5])

    horizon = cart_pole_env.planning_steps
    Ps, Ks = calculate_Ps_Ks(A, B, Q, R, horizon)

    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    for i in range(horizon):
        X = xs[-1]
        K = Ks[i]
        U = K * X
        X_next = A*X + B*U

        us.append(U)
        xs.append(X_next)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    env = CartPoleContEnv(initial_theta=np.pi * 0.37)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

