from __future__ import division, print_function
import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv

import matplotlib.pyplot as plt
# %% LSPI
def lspi_data_sample(N=3000):
    env = MountainCarWithResetEnv()
    goal_pos = 0.5
    min_pos = -1.2
    max_pos = 0.6
    min_speed = -0.07
    max_speed = 0.07
    data = []
    rewards = np.zeros([N, 1])
    states = np.zeros([N, 2])
    actions = np.zeros(N)
    next_states = np.zeros([N, 2])
    for i in range(N):
    #for pos in np.linspace(min_pos, max_pos, num=N_pos):
        #for speed in np.linspace(min_speed, max_speed, num=N_speed):
            #for action in [0, 1, 2]:
        pos = (max_pos - min_pos) * np.random.sample() + min_pos
        speed = (max_speed - min_speed) * np.random.sample() + min_speed
        action = np.random.choice(3)
        #res = {'s' : np.array([pos, speed]), 'a' : action}
        states[i, :] = np.array([pos, speed])
        actions[i] = action
        if pos >= goal_pos:
            #res['r'] = 1
            rewards[i, 0] = 1
            #res['s_next'] = np.array([pos, speed])
            next_states[i, :] = np.array([pos, speed])
        else:
            env.reset_specific(pos, speed)
            s_next, reward, _, _ = env.step(action)
            #res['r'] = reward
            rewards[i, 0] = reward
            #res['s_next'] = s_next
            next_states[i, :] = s_next

    return data, states, actions, rewards, next_states

def data_stats(states):
    global pos_mu, pos_sigma, speed_mu, speed_sigma
    pos_mu = np.mean(states[:, 0])
    pos_sigma = np.std(states[:, 0])
    speed_mu = np.mean(states[:, 1])
    speed_sigma = np.std(states[:, 1])

    return pos_mu, pos_sigma, speed_mu, speed_sigma

def e(s):
    # Implementation of RBF features

    # pos, speed statistics should be global
    global pos_mu, pos_sigma, speed_mu, speed_sigma # Does this work? Hagai pls help
    n_s = np.zeros(s.shape)
    n_s[:,0] = ( s[:,0] - pos_mu ) / pos_sigma
    n_s[:,1] = ( s[:,1] - speed_mu ) / speed_sigma
    centers = [(-1.2, -0.07), (-1.2, 0.07), (0.5, -0.07), (0.5, 0.07), (0, 0)]
    n_centers = np.array([( (c[0] - pos_mu / pos_sigma), (c[1] - speed_mu / speed_sigma) ) for c in centers])
    scales = [1, 1, 1, 1, 1]
    feats = np.ones([n_s.shape[0], np.size(scales) + 1])
    for i, n_c in enumerate(n_centers):
        feats[:,i] = np.exp(-scales[i] * np.linalg.norm(n_s - n_c, axis = 1))
    #for b, c in zip(scales, n_centers):
    #    feats = np.append(feats, np.exp(-b * np.linalg.norm( n_s - np.array(c) ) ) )
    
    return feats

def feats(s, actions):
    N_a = 3
    e_s = e(s)
    N_f = np.shape(e_s)[1]
    feats = np.zeros([np.shape(e_s)[0], N_a * N_f])
    for i, a in enumerate(actions):
        np.put(feats[i,:], range(int(a)*N_f, int(a+1)*N_f), e_s[i,:])
    return feats
     
def next_a(next_s, theta):
    N = np.shape(next_s)[0]
    
    Q_est = np.zeros([N,3])
    Q_est[:,0] = theta.T.dot(feats(next_s, np.zeros(N)).T)
    Q_est[:,1] = theta.T.dot(feats(next_s, np.ones(N)).T)
    Q_est[:,2] = theta.T.dot(feats(next_s, 2*np.ones(N)).T)
    
    max_a = np.argmax(Q_est, axis = 1)

    
#    max_a = np.zeros(N)
#    for i in range(N):
#        max_val = -np.inf
#        for a in [0, 1, 2]:
#            curr_val = theta.T.dot(feats(next_s[i,:].reshape([1,2]), np.array([a])).T)
#            if curr_val > max_val:
#                max_val = curr_val
#                max_a[i] = a
    return max_a
               
def train_lspi(data, states, actions, rewards, next_states, gamma = 0.999):
    #data = lspi_data_sample()
    global pos_mu, pos_sigma, speed_mu, speed_sigma
    pos_mu, pos_sigma, speed_mu, speed_sigma = data_stats(states)
    #proj_data = np.array([ feats(d['s'], d['a']) for d in data])
    #rs = [d['r'] for d in data]
    proj_data = feats(states, actions)
    theta = np.zeros([np.size(proj_data[0]), 1 ])
    
    #d_est = (1 / np.size(data)) * sum([ r * d for d, r in zip(proj_data, rs) ] )
    d_est = 0
    #for d, r in zip(proj_data, rs):
    #    d_est += (1 / np.size(data)) * r * d
    d_est = (1 / np.shape(proj_data)[0]) * proj_data.T.dot(rewards)
    N = 100
    eps = 0.00001
    for i in range(N):
        proj_next = feats( next_states, next_a(next_states, theta) )
        # C_est = (1 / np.size(data)) * sum( [ np.matmul(f_st_at, f_st_at.T - gamma*f_st1_at1.T   ) for f_st_at, f_st1_at1 in zip(proj_data, proj_next) ] )
        #C_est = np.zeros(np.matmul(proj_data[0], proj_data[0].T).shape) # Init zeros in correct shape
        #for f_st_at, f_st1_at1 in zip(proj_data, proj_next):
        #    C_est += (1 / np.size(data)) * np.matmul( f_st_at, f_st_at.T - gamma*f_st1_at1.T )
        C_est = (1 / np.shape(proj_data)[0]) * proj_data.T.dot(proj_data - gamma * proj_next)
        theta_next = np.matmul( np.linalg.inv(C_est) , d_est ).reshape([np.size(theta), 1])
        
        if max( theta_next - theta ) < eps:
            yield theta_next
            return
        else:
            theta = theta_next
            yield theta

def test_lspi(N_list=[3000]):
    results = []
    fig, ax = plt.subplots()
    #plt.rcParams.update({'font.size': 10})
    for N in N_list:
        N = int(N)
        env = MountainCarWithResetEnv()
        high = -0.4
        low = -0.6
        init_states = [( (high - low)*np.random.sample() + low, 0) for i in range(10)]
        max_iter = 1000
        total_success = 5 * [[]]
        for i in range(5):
            #print("Starting iteration i=", i)
            np.random.seed(seed = i)
            data, states, actions, rewards, next_states = lspi_data_sample(N)
            theta_n = list(train_lspi(data, states, actions, rewards, next_states))
            success_theta = []
            for theta in theta_n:
                #print("New theta")
                success_rate = 0
                for init_s in init_states:
                    #print("New init state")
                    env.reset_specific(*init_s)
                    #env.render()
                    is_done = False
                    a = next_a(np.array(init_s).reshape([1,2]), theta) # First step
                    for j in range(max_iter):
                        #print("Game iteration:", j)
                        next_s, r, is_done, _ = env.step(int(a))
                        a = next_a(next_s.reshape([1,2]), theta)
                        if is_done:
                            success_rate += 1.0
                            break
                success_theta.append(success_rate/10.0)
            total_success[i] = success_theta
        max_len = max([len(total_success[i]) for i in range(len(total_success))])
        for i in range(len(total_success)):
            l = len(total_success[i])
            if l < max_len:
                for j in range(max_len - l):
                    total_success[i].append(total_success[i][-1])
        res = np.array(total_success)
        res_mean = np.mean(res, axis = 0)
        results.append(list(res_mean))
        
    max_len = max([len(results[i]) for i in range(len(results))]) 
    for i in range(len(results)):
        l = len(results[i])
        if l < max_len:
            for j in range(max_len - l):
                results[i].append(results[i][-1])
    for N, res_mean in zip(N_list, results):
        it = list(range(1, len(res_mean) + 1))

        ax.plot(it, res_mean, label = 'N = ' + str(int(N)))    
    ax.grid(True)
    if len(N_list) > 1:
        plt.title('Average success rate per iteration for different amounts of samples')
    else:
        plt.title('Average success rate per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Success rate')
    plt.legend(loc = 'lower left')
    plt.ylim(-0.2, 1.2)
    plt.show()

    
    return res_mean

def visualize_lspi(states, theta):
    N = states.shape[0]
    opt_a = next_a(states, theta)
    fig, ax = plt.subplots()
    fsize = 22
    #plt.rcParams.update({'font.size': fsize})
    ac = [0, 1, 2]
    for a, color, label in zip(ac, ['tab:blue', 'tab:orange', 'tab:green'], ['LEFT', 'STAY', 'RIGHT']):
        xy = states[a == opt_a, :]
        ax.scatter(xy[:, 0], xy[:, 1], c=color, label=label, edgecolors='none')
        
    ax.legend()
    ax.grid(True)
    plt.title('Sample size - ' +str(N))
    plt.xlabel('Position',fontsize=fsize)
    plt.ylabel('Velocity',fontsize=fsize)

    plt.show()



#if __name__ == '__main__':
#    import sys
#    if len(sys.argv) > 1:
#        N = int(sys.argv[1])
#        print(test_lspi(N))
#    else:
#        print(test_lspi())
 # %%  
if __name__ == '__main__':
    env = MountainCarWithResetEnv()
    # # run no force
    # env.reset()
    # env.render()
    # is_done = False
    # while not is_done:
    #     _, r, is_done, _ = env.step(1)
    #     env.render()
    #     print(r)
    # # run random forces
    # env.reset()
    # env.render()
    # is_done = False
    # while not is_done:
    #     _, r, is_done, _ = env.step(env.action_space.sample())  # take a random action
    #     env.render()
    #     print(r)
    
    
    
    # set specific
    N = 3000
    test_lspi([N])
    test_lspi([N/10, N/2, N, 2*N, 10*N])
    data, states, actions, rewards, next_states = lspi_data_sample(N)
    theta_n = list(train_lspi(data, states, actions, rewards, next_states))

    env.reset()
    s, _, _, _ = env.step(0)
    env.render()
    is_done = False
    i = 0
    while (not is_done) and (i < 500):
        s, r, is_done, _ = env.step(int(next_a(s.reshape([1, 2]), theta_n[-1])))  # Use optimal policy
        env.render()
        i = i+1
        #print(r)
    env.close()
    
    #visualize_lspi(states, theta_n[-1])
