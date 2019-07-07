from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from mountain_car_with_data_collection import MountainCarWithResetEnv
from q1 import lspi_data_sample

class QLearningAgent:
    def __init__(self):
        self.game = MountainCarWithResetEnv()
        self.reset_theta()

        # Constants used for data standardization
        self.pos_mu = (self.game.min_position + self.game.max_position)/2
        self.pos_sigma = (self.game.max_position - self.game.min_position)/np.sqrt(12)
        self.speed_mu = 0
        self.speed_sigma = 2*self.game.max_speed/np.sqrt(12)

        # Cache of samples used for visualizing the policy
        self.vis_samples = None

    def reset_theta(self):
        self.theta = np.random.normal(size=(1, 78))

    def reset(self, state=None):
        if state is None:
            return self.game.reset()
        return self.game.reset_specific(*state)

    def not_a(self, action):
        return -(action -1) + 1

    def next_a(self, state):
        N = np.shape(state)[0]

        Q_est = np.zeros([N, 3])
        Q_est[:, 0] = self.theta.dot(self.extract_features(state, np.zeros(N)).T)
        Q_est[:, 1] = self.theta.dot(self.extract_features(state, np.ones(N)).T)
        Q_est[:, 2] = self.theta.dot(self.extract_features(state, 2*np.ones(N)).T)

        action = np.argmax(Q_est, axis=1)

        return self.not_a(action)

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

    def visualize(self):
        if self.vis_samples is None:
            ret = lspi_data_sample(10000)
            self.vis_samples = ret[1]

        N = self.vis_samples.shape[0]
        opt_a = self.next_a(self.vis_samples)

        fsize = 22
        plt.rcParams.update({'font.size': fsize})
        plt.clf()
        ac = [0, 1, 2]
        for a, color, label in zip(ac, ['tab:blue', 'tab:orange', 'tab:green'], ['LEFT', 'STAY', 'RIGHT']):
            xy = self.vis_samples[a == opt_a, :]
            plt.scatter(xy[:, 0], xy[:, 1], c=color, label=label, edgecolors='none')

        plt.legend()
        plt.grid(True)
        plt.title('Sample size - ' +str(N))
        plt.xlabel('Position',fontsize=fsize)
        plt.ylabel('Velocity',fontsize=fsize)

        plt.pause(0.1)

    def train_online(self, epsilon=1, alpha=1, gamma=0.999, iterations=30):
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

    def gather_data(self, epsilon, iterations_per_game=1000, games=5):
        states = np.zeros((iterations_per_game*games, 2))
        actions = np.zeros((iterations_per_game*games, 1))
        next_states = np.zeros((iterations_per_game*games, 2))
        rewards = np.zeros((iterations_per_game*games, 1))
        data = (states, actions, next_states, rewards)

        success_count = 0
        data_index = 0
        for g in range(games):
            state = self.reset()
            state = state.reshape((1, 2))
            for i in range(iterations_per_game):
                if np.random.uniform() > epsilon:
                    rand = False
                    action = self.next_a(state)[0]
                else:
                    rand = True
                    action = np.random.choice(3)
                next_state, reward, is_done, _ = self.game.step(action)
                success_count += np.sum(reward)
                states[data_index, :] = state
                actions[data_index, :] = action
                next_states[data_index, :] = next_state
                rewards[data_index, :] = reward
                data_index += 1
                #print("i: {}, state: {}, action: {}, next: {}, r: {}, rand: {}".format(i, state, action, next_state, reward, rand))
                state = np.array(next_state).reshape((1, 2))
                if is_done:
                    break
        success_rate = success_count / games
        return data, success_rate, data_index

    def train_step(self, alpha, data, batch_size=100, gamma=0.999):
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
            step = self.extract_features(states[i].reshape((1, 2)), actions[i]) * coeff
            update_step += step
        max_element = np.max(np.abs(update_step))
        return self.theta + alpha * update_step / (max_element or 1)

    def reset_random(self):
        init_state = (np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07))
        return self.reset(init_state)

    def train(self, init_epsilon=1, init_alpha=1, max_iterations=30, visualise=True, test_states=[]):
        alpha = init_alpha
        epsilon = init_epsilon
        success_rates = np.zeros((len(test_states), max_iterations))
        for i in range(max_iterations):
            data, win_pct, max_ind = self.gather_data(epsilon)

            data = (data[0][:max_ind, :],
                    data[1][:max_ind, :],
                    data[2][:max_ind, :],
                    data[3][:max_ind, :])
            old_theta = self.theta
            for j in range(10):
                self.theta = self.train_step(alpha, data)
            theta_diff = self.theta - old_theta

            diff_max = np.max(np.abs(theta_diff))
            theta_max = np.max(np.abs(self.theta))
            success_rates[:, i] = self.test_train_iteration(test_states)
            avg_rate = np.average(success_rates[:, i])
            print("Iter", i, "train_iters", max_ind, "train_win_pct", win_pct, "test_win_pct", avg_rate, "alpha", alpha, "ep", epsilon, "theta_new - theta (max) =", diff_max, "theta_max", theta_max)
            epsilon = 0.9 * epsilon
            alpha = 0.8 * alpha
            if visualise:
                self.visualize()
        return success_rates

    def play(self, init_state=None, render=True, max_iterations=1000):
        state = self.reset(init_state).reshape((1,2))
        done = False
        for i in range(max_iterations):
            action = int(self.next_a(state))
            next_state, reward, is_done, _ = self.game.step(action)
            if render:
                self.game.render()
            state = np.array(next_state).reshape((1,2))
            if is_done:
                done = True
                break
        self.game.close()
        return done

    def test_train_iteration(self, test_init_states):
        results = np.zeros(len(test_init_states))
        for state_idx, init_state in enumerate(test_init_states):
            result = self.play(init_state, render=False)
            results[state_idx] = int(result)

        return results

    def get_test_states(self, count=10):
        return [(np.random.uniform(low=-0.6, high=-0.4), 0) for i in range(count)]

    def test_model(self, training_cycles=5, test_states=None, **training_args):
        success_rates = None
        if test_states is None:
            test_states = self.get_test_states()
        for t in range(training_cycles):
            self.reset_theta()
            print("*** TRAINING EXPERIMENT {} ***".format(t))
            rates = self.train(test_states=test_states, **training_args)
            print("*** RESULT ***")
            print(rates)
            if success_rates is None:
                success_rates = rates
            else:
                success_rates += rates
        success_rates /= training_cycles

        return test_states, success_rates

    def plot_success_rates(self, success_rates):
        avg = np.average(success_rates, axis=0)
        plt.plot(avg)
        plt.title('Average success rate per iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Success rate')
        plt.show()

if __name__ == '__main__':
    q = QLearningAgent()
    states, rates = q.test_model(visualise=False)
    print("Training done")
    print("Training states:", states)
    print("Success rates:")
    print(rates)

    q.plot_success_rates(rates)
