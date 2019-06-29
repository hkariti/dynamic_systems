"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

from __future__ import division
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


pos_mu = pos_sigma = speed_mu = speed_sigma = 0

class MountainCarWithResetEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = float(done)
        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        random_position = self.np_random.uniform(low=-0.6, high=-0.4)
        initial_velocity = 0
        return self.reset_specific(random_position, initial_velocity)

    def reset_specific(self, current_position, current_velocity):
        assert self.min_position <= current_position <= self.max_position
        assert -self.max_speed <= current_velocity <= self.max_speed
        self.state = np.array([current_position, current_velocity])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# %% LSPI 
def lspi_data_sample(N = 100000):
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
    next_states = np.zeros([N,2])
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
        if pos >= goal_pos :
            #res['r'] = 1
            rewards[i, 0] = 1
            #res['s_next'] = np.array([pos, speed])
            next_states[i,:] = np.array([pos, speed])
        else:
            env.reset_specific(pos, speed)
            s_next, reward, _ , _ = env.step(action)
            #res['r'] = reward
            rewards[i, 0] = reward
            #res['s_next'] = s_next
            next_states[i,:] = s_next
        
    return data, states, actions, rewards, next_states

def data_stats(states):
    
    pos_mu = np.mean(states[:,0])
    pos_sigma = np.std(states[:,0]) 
    speed_mu = np.mean(states[:,1])
    speed_sigma = np.std(states[:,1])
    
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
    scales = [1, 1, 1, 1 ,1]
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
    eps = 0.001
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
    
def test_lspi(N = 100000):
    env = MountainCarWithResetEnv()
    high = -0.4
    low = -0.6
    init_states = [( (high - low)*np.random.sample() + low, 0) for i in range(10)]
    max_iter = 1000
    total_success = 5 * [[]]
    for i in range(5):
        print("Starting iteration i=", i)
        np.random.seed(seed = i)
        data, states, actions, rewards, next_states = lspi_data_sample(N)
        theta_n = list(train_lspi(data, states, actions, rewards, next_states))
        success_theta = []
        for theta in theta_n:
            print("New theta")
            success_rate = 0
            for init_s in init_states:
                print("New init state")
                env.reset_specific(*init_s)
                #env.render()
                is_done = False
                a = next_a(np.array(init_s).reshape([1,2]), theta) # First step
                for j in range(max_iter):
                    print("Game iteration:", j)
                    next_s, r, is_done, _ = env.step(int(a))
                    a = next_a(next_s.reshape([1,2]), theta)
                    if is_done:
                        success_rate += 1
                        break
            success_theta.append(success_rate/10)
        total_success[i] = success_theta
    return total_success

# %% main
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
    env.reset_specific(0.3, 0.0)
    env.render()
    is_done = False
    while not is_done:
        _, r, is_done, _ = env.step(2)  # go left
        env.render()
        print(r)
    env.close()
