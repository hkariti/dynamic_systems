"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

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
def lspi_data_sample():
    env = MountainCarWithResetEnv()
    goal_pos = 0.5
    min_pos = -1.2
    max_pos = 0.6
    min_speed = -0.07
    max_speed = 0.07
    N_speed = 100
    N_pos = 1000
    data = []
    for pos in np.linspace(min_pos, max_pos, num=N_pos):
        for speed in np.linspace(min_speed, max_speed, num=N_speed):
            for action in [0, 1, 2]:
                res = {'s' : np.array([pos, speed]), 'a' : action}
                if pos >= goal_pos :
                    res['r'] = 1
                    res['s_next'] = np.array([pos, speed])
                else:
                    env.reset_specific(pos, speed)
                    s_next, reward, _ , _ = env.step(action)
                    res['r'] = reward
                    res['s_next'] = s_next
                
                data.append(res)
    return data

def data_stats(data):
    pos_l = []
    speed_l = []
    for d in data:
        pos_l.append(d['s'][0])
        speed_l.append(d['s'][1])
    pos_mu = np.mean(pos_l)
    pos_sigma = np.std(pos_l)
    speed_mu = np.mean(speed_l)
    speed_sigma = np.std(speed_l)
    
    return pos_mu, pos_sigma, speed_mu, speed_sigma

def e(s):
    # Implementation of RBF features

    # pos, speed statistics should be global    
    global pos_mu, pos_sigma, speed_mu, speed_sigma # Does this work? Hagai pls help
    
    n_pos = ( s[0] - pos_mu ) / pos_sigma
    n_speed = ( s[1] - speed_mu ) / speed_sigma
    n_s = np.array([n_pos, n_speed])
    centers = [(-1.2, -0.07), (-1.2, 0.07), (0.5, -0.07), (0.5, 0.07), (0, 0)]
    scales = [1, 1, 1, 1 ,1]
    feats = np.array([])
    for b, c in zip(scales, centers):
        feats = np.append(feats, np.exp(-b * np.linalg.norm( n_s - np.array(c) ) ) )
    feats = np.append(feats, 1)
    
    return feats
        
    
def feat(s, a):
    N_a = 3
    e = e(s)
    N_f = np.size(e)
    feats = np.zeros([N_f * N_a]) 
    np.put(feats, range(a*N_f, (a+1)*N_f), e[:])
    return feats
                    
                
            
    

    
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
