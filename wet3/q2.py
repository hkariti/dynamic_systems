from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from mountain_car_with_data_collection import MountainCarWithResetEnv

class QLearningAgent:
    def __init__(self):
        self.game = MountainCarWithResetEnv()
        #self.theta = np.zeros((1, 18))
        self.theta = np.zeros((1, 78))

        # Constants used for data standardization
        self.pos_mu = (self.game.min_position + self.game.max_position)/2
        self.pos_sigma = (self.game.max_position - self.game.min_position)/np.sqrt(12)
        self.speed_mu = 0
        self.speed_sigma = 2*self.game.max_speed/np.sqrt(12)

    def reset(self, state=None):
        if state:
            return self.game.reset_specific(*state)
        else:
            return self.game.reset()

    def next_a(self, state):
        N = np.shape(state)[0]

        Q_est = np.zeros([N, 3])
        Q_est[:, 0] = self.theta.dot(self.extract_features(state, np.zeros(N)).T)
        Q_est[:, 1] = self.theta.dot(self.extract_features(state, np.ones(N)).T)
        Q_est[:, 2] = self.theta.dot(self.extract_features(state, 2*np.ones(N)).T)

        action = np.argmax(Q_est, axis=1)
        action = -(action - 1) + 1

        return action

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
        #centers = [(-1.2, -0.07), (-1.2, 0.07), (0.5, -0.07), (0.5, 0.07), (0, 0)]
        centers = [(-1.2, -0.07), (-1.2, 0.07), (0.5, -0.07), (0.5, 0.07), (0, 0)]
        centers = []
        for i in -1.2, -0.6, 0, 0.6, 1.2:
            for j in -0.07, -0.03, 0, 0.03, 0.07:
                centers.append((i, j))
        n_centers = np.array([(
            (c[0] - self.pos_mu)/self.pos_sigma,
            (c[1] - self.speed_mu)/self.speed_sigma) for c in centers])
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
            ax.scatter(xy[:, 0], xy[:, 1], c=color, label=label, edgecolors='none')

        ax.legend()
        ax.grid(True)
        plt.title('Sample size - ' +str(N))
        plt.xlabel('Position',fontsize=fsize)
        plt.ylabel('Velocity',fontsize=fsize)

        plt.show()

    def train_online(self, epsilon=0.5, alpha=0.1, gamma=0.999, iterations=500):
        state = self.reset()
        for i in range(iterations):
            if np.random.uniform() > epsilon:
                rand = False
                action = self.next_a(state.reshape((1, 2)))[0]
            else:
                rand = True
                action = np.random.choice(3)
            print(action)
            next_state, reward, is_done, _ = self.game.step(action)
            coeff = reward + gamma * self.q_max(next_state.reshape((1, 2))) - self.q(state.reshape((1, 2)), np.array([action]))
            update_step = self.extract_features(state.reshape((1, 2)), np.array([action])) * coeff
            old_theta = self.theta.copy()
            self.theta += alpha * update_step
            diff = np.max(np.abs(self.theta - old_theta))
            state = next_state
            print("Iteration", i, "diff", diff)
            if is_done:
                print("Done on iteration", i)
                return 1
        return 0

    def gather_data(self, epsilon, iterations_per_game=1000, games=20):
        states = np.zeros((iterations_per_game*games, 2))
        actions = np.zeros((iterations_per_game*games, 1))
        next_states = np.zeros((iterations_per_game*games, 2))
        rewards = np.zeros((iterations_per_game*games, 1))
        data = (states, actions, next_states, rewards)

        was_done = 0
        data_index = 0
        for g in range(games):
            state = self.reset_random()
            state = state.reshape((1, 2))
            for i in range(iterations_per_game):
                if np.random.uniform() > epsilon:
                    rand = False
                    action = self.next_a(state)[0]
                else:
                    rand = True
                    action = np.random.choice(3)
                next_state, reward, is_done, _ = self.game.step(action)
                was_done += np.sum(reward)
                states[data_index, :] = state
                actions[data_index, :] = action
                next_states[data_index, :] = next_state
                rewards[data_index, :] = reward
                data_index += 1
                #print("i: {}, state: {}, action: {}, next: {}, r: {}, rand: {}".format(i, state, action, next_state, reward, rand))
                state = np.array(next_state).reshape((1, 2))
                if is_done:
                    break
        return data, was_done, data_index

    def train_step(self, alpha, data, batch_size=500, gamma=0.999):
        data_length = data[0].shape[0]
        reward_indices = (data[3] == 1).reshape(data_length)
        reward_count = reward_indices.sum()
        batch_indices = np.random.randint(0, data_length, batch_size - reward_count)
        batch_marker = np.zeros(data_length, dtype=bool)
        batch_marker[batch_indices] = True
        batch_marker[reward_indices] = True
        batch_size = batch_marker.sum()

        states = data[0][batch_marker]
        actions = data[1][batch_marker]
        next_states = data[2][batch_marker]
        rewards = data[3][batch_marker]

        update_step = 0
        for i in range(batch_size):
            coeff = rewards[i] + gamma * self.q_max(next_states[i].reshape((1, 2))) - self.q(states[i].reshape((1, 2)), actions[i])
            update_step += self.extract_features(states[i].reshape((1, 2)), actions[i]) * coeff
        return self.theta + alpha * update_step / batch_size

    def reset_random(self):
        init_state = (np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07))
        return self.reset(init_state)

    def train(self, init_state=None, init_epsilon=1, init_alpha=1, max_iterations=100):
        #_, states, actions, rewards, next_states = q1.lspi_data_sample(10000)
        #data = (states, actions.reshape((10000, 1)), next_states, rewards)
        alpha = init_alpha
        epsilon = init_epsilon
        import q1
        ret = q1.lspi_data_sample(10000)
        lspi_data = (ret[1], ret[2], ret[4], ret[3])
        vis_samples = ret[1]
        for i in range(max_iterations):
            data, is_done, max_ind = self.gather_data(epsilon)

            data = (data[0][:max_ind, :],
                    data[1][:max_ind, :],
                    data[2][:max_ind, :],
                    data[3][:max_ind, :])
            #data = lspi_data
            #max_ind = 1
            #is_done = lspi_data[3].sum()
            old_theta = self.theta
            for j in range(20):
                self.theta = self.train_step(alpha, data)
            theta_diff = self.theta - old_theta

            diff_max = np.max(np.abs(theta_diff))
            theta_max = np.max(np.abs(self.theta))
            print("Iter", i, "max_ind", max_ind, "rewards", is_done, "alpha", alpha, "ep", epsilon, "theta_new - theta (max) =", diff_max, "theta_max", theta_max)
            epsilon = 0.9 * epsilon
            alpha = 0.8 * alpha
            self.visualize_lspi(vis_samples)

            #if diff_max <= 0.001:
            #    print("Converged!")
            #    return

    def play(self):
        state = self.reset().reshape((1,2))
        for i in range(1000):
            action = int(self.next_a(state))
            next_state, reward, is_done, _ = self.game.step(action)
            self.game.render()
            state = np.array(next_state).reshape((1,2))
            if is_done:
                break
        self.game.close()

