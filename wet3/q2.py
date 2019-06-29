from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from mountain_car_with_data_collection import MountainCarWithResetEnv

class QLearningAgent:
    def __init__(self):
        self.game = MountainCarWithResetEnv()
        self.theta = np.zeros((1, 18))

        # Constants used for data standardization
        self.pos_mu = (self.game.min_position + self.game.max_position)/2
        self.pos_sigma = (self.game.max_position - self.game.min_position)/12
        self.speed_mu = 0
        self.speed_sigma = 2*self.game.max_speed/np.sqrt(12)

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

    def q_max(self, state):
        N = np.shape(state)[0]

        Q_est = np.zeros([N, 3])
        Q_est[:, 0] = self.theta.dot(self.extract_features(state, np.zeros(N)).T)
        Q_est[:, 1] = self.theta.dot(self.extract_features(state, np.ones(N)).T)
        Q_est[:, 2] = self.theta.dot(self.extract_features(state, 2*np.ones(N)).T)

        return np.max(Q_est, axis=1)

    def q(self, state, action):
        Q_est = self.theta.dot(self.extract_features(state, action * np.ones(1)).T)

        return Q_est

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
        scales = [1 for c in centers]

        feats = np.ones([n_s.shape[0], np.size(scales) + 1])
        for i, n_c in enumerate(n_centers):
            feats[:, i] = np.exp(-scales[i] * np.linalg.norm(n_s - n_c, axis=1))

        return feats

    def visualize_lspi(self, states):
        N = states.shape[0]
        opt_a = self.next_a(states)

        fig, ax = plt.subplots()
        fsize = 22
        plt.rcParams.update({'font.size': fsize})
        ac = [0, 1, 2]
        for a, color, label in zip(ac, ['tab:blue', 'tab:orange', 'tab:green'], ['LEFT', 'STAY', 'RIGHT']):
            xy = states[a == opt_a, :]
            print(xy)
            ax.scatter(xy[:, 0], xy[:, 1], c=color, label=label, edgecolors='none')

        ax.legend()
        ax.grid(True)
        plt.title('Sample size - ' +str(N))
        plt.xlabel('Position',fontsize=fsize)
        plt.ylabel('Velocity',fontsize=fsize)

        plt.show()

    def gather_data(self, epsilon, iterations=10000):
        states = np.zeros((iterations, 2))
        actions = np.zeros((iterations, 1))
        next_states = np.zeros((iterations, 2))
        rewards = np.zeros((iterations, 1))
        data = (states, actions, next_states, rewards)

        state = np.array(self.game.state).reshape((1, 2))
        is_done = False
        i = None
        for i in range(iterations):
            if np.random.uniform() < epsilon:
                rand = True
                action = self.next_a(state)[0]
            else:
                rand = False
                action = np.random.choice([0, 1, 2])
            next_state, reward, is_done, _ = self.game.step(action)
            states[i, :] = state
            actions[i, :] = action
            next_states[i, :] = next_state
            rewards[i, :] = reward
            #print("i: {}, state: {}, action: {}, next: {}, r: {}, rand: {}".format(i, state, action, next_state, reward, rand))
            if is_done or (not is_done and i > 1000):
                self.reset_random()
            state = np.array(next_state).reshape((1, 2))
        return data, is_done, i

    def train_step(self, alpha, data, batch_size=500, gamma=0.99):
        data_length = data[0].shape[0]
        batch_indices = list(range(batch_size))
        np.random.shuffle(batch_indices)

        states = data[0][batch_indices]
        actions = data[1][batch_indices]
        next_states = data[2][batch_indices]
        rewards = data[3][batch_indices]

        update_step = 0
        for i in range(batch_size):
            coeff = rewards[i] + gamma * self.q_max(next_states[i].reshape((1, 2))) - self.q(states[i].reshape((1, 2)), actions[i])
            update_step += self.extract_features(states[i].reshape((1, 2)), actions[i]) * coeff
        return self.theta + alpha * update_step / batch_size

    def reset_random(self):
        init_state = (np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07))
        self.reset(init_state)

    def train(self, init_state=None, init_epsilon=0.1, init_alpha=0.1, max_iterations=100):
        for i in range(max_iterations):
            self.reset_random()
            data, is_done, max_ind = self.gather_data(init_epsilon)

            if not is_done:
                print("Didn't finish, num", i)

            data = (data[0][:max_ind + 1, :],
                    data[1][:max_ind + 1, :],
                    data[2][:max_ind + 1, :],
                    data[3][:max_ind + 1, :])
            new_theta = self.train_step(init_alpha, data)
            new_theta = self.train_step(init_alpha, data)
            new_theta = self.train_step(init_alpha, data)
            new_theta = self.train_step(init_alpha, data)
            theta_diff = new_theta - self.theta
            self.theta = new_theta

            diff_max = np.max(np.abs(theta_diff))
            print("theta_new - theta (max) =", diff_max)

            #if diff_max <= 0.001:
            #    print("Converged!")
            #    return

    def play(self):
        self.reset()

        for i in range(1000):
            state = np.array(self.game.state).reshape((1, 2))
            action = int(self.next_a(state))
            next_state, reward, is_done, _ = self.game.step(action)
            self.game.render()
            state = np.array(next_state).reshape((1,2))
            if is_done:
                return

