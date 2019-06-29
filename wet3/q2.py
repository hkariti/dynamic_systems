from __future__ import division, print_function
import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv

class QLearningAgent:
    def __init__(self):
        self.game = MountainCarWithResetEnv()
        self.theta = np.zeros((1, 18))

        # Constants used for data standardization
        self.pos_mu = (self.game.min_position + self.game.max_position)/2
        self.pos_sigma = (self.game.max_position - self.game.min_position)/12
        self.speed_mu = 0
        self.speed_sigma = 2*self.game.max_speed/12

    def reset(self, state=None):
        if state:
            self.game.reset_specific(*state)
        else:
            self.game.reset()

    def next_a(self, state):
        N = np.shape(state)[0]

        Q_est = np.zeros([N, 3])
        Q_est[:, 0] = self.theta.dot(self.extract_features(state, np.zeros(N)).T)
        Q_est[:, 1] = self.theta.dot(self.extract_features(state, np.ones(N)).T)
        Q_est[:, 2] = self.theta.dot(self.extract_features(state, 2*np.ones(N)).T)

        return np.argmax(Q_est, axis=1)

    def extract_features(self, s, actions):
        N_a = 3
        e_s = self.rbf(s)
        N_f = np.shape(e_s)[1]
        feats = np.zeros([np.shape(e_s)[0], N_a * N_f])
        for i, a in enumerate(actions):
            np.put(feats[i, :], range(int(a)*N_f, int(a+1)*N_f), e_s[i, :])
        return feats
     
    def rbf(self, s):
        # Implementation of RBF features
        # pos, speed statistics should be global
        n_s = np.zeros(s.shape)
        n_s[:, 0] = (s[:, 0] - self.pos_mu) / self.pos_sigma
        n_s[:, 1] = (s[:, 1] - self.speed_mu) / self.speed_sigma
        centers = [(-1.2, -0.07), (-1.2, 0.07), (0.5, -0.07), (0.5, 0.07), (0, 0)]
        n_centers = np.array([(
            (c[0] - self.pos_mu/self.pos_sigma),
            (c[1] - self.speed_mu/self.speed_sigma)) for c in centers])
        scales = [1, 1, 1, 1, 1]

        feats = np.ones([n_s.shape[0], np.size(scales) + 1])
        for i, n_c in enumerate(n_centers):
            feats[:, i] = np.exp(-scales[i] * np.linalg.norm(n_s - n_c, axis=1))

        return feats

    def gather_data(self, epsilon, iterations=100):
        states = np.zeros((iterations, 2))
        actions = np.zeros((iterations, 1))
        next_states = np.zeros((iterations, 2))
        rewards = np.zeros((iterations, 1))
        data = (states, actions, next_states, rewards)

        state = self.game.state.reshape((1, 2))
        i = None
        for i in range(iterations):
            if np.random.uniform() > epsilon:
                action = self.next_a(state)[0]
            else:
                action = np.random.choice([0, 1, 2])
            next_state, reward, is_done, _ = self.game.step(action)
            states[i, :] = state
            actions[i, :] = action
            next_states[i, :] = next_state
            rewards[i, :] = reward
            if is_done:
                break
            state = next_state.reshape((1, 2))
        return data, i
