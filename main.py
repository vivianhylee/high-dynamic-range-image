import numpy as np
import pickle
from cvxopt import matrix, solvers



class Qlearning:
    def __init__(self, num_states, num_actions, players=('A', 'B'), max_episode=1000000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_players = len(players)
        self.max_episode = max_episode

        self.player_index = {}
        for i in range(self.num_players):
            self.player_index[players[i]] = i


        self.action_delta = {'st': 0, 'n': -4, 's': 4, 'w': -1, 'e': 1}
        self.action_to_index = {'st': 0, 'n': 1, 's': 2, 'w': 3, 'e': 4}
        self.possible_moves = {0: ['st', 'e', 's'], 1: ['st', 'w', 'e', 's'], 2: ['st', 'w', 'e', 's'],
                               3: ['st', 'w', 's'], 4: ['st', 'e', 'n'], 5: ['st', 'w', 'e', 'n'],
                               6: ['st', 'w', 'e', 'n'], 7: ['st', 'w', 'n']}

        self.gamma = 0.9
        self.min_alpha = 0.001


        self.states_to_index = self.initialize_states()
        self.q_table = None

        self.target_state = 'B21'
        self.target_action = [2, 0]

        self.error = None


    def get_error(self):
        return self.error


    def initialize_states(self):
        holder = self.player_index.keys()
        out = {}
        id_q = 0
        for holder in holder:
            for a in range(self.num_states):
                for b in range(self.num_states):
                    if a != b:
                        out[holder + str(a) + str(b)] = id_q
                        id_q += 1
        return out


    def get_reward(self, state):
        holder = self.player_index[state[0]]
        loc_a = int(state[1])
        loc_b = int(state[2])

        holder = 0 if state[0] == 'A' else 1
        loc_a = int(state[1])
        loc_b = int(state[2])

        if holder == 0 and loc_a in [0, 4]:
            return [100, -100]
        elif holder == 0 and loc_a in [3, 7]:
            return [-100, 100]
        elif holder == 1 and loc_b in [3, 7]:
            return [-100, 100]
        elif holder == 1 and loc_b in [0, 4]:
            return [100, -100]
        else:
            return [0, 0]


    def get_next_moves(self, first_player, cnt_state):
        holder = self.player_index[cnt_state[0]]
        loc_a = int(cnt_state[1])
        loc_b = int(cnt_state[2])

        next_a = self.possible_moves[loc_a]
        next_b = self.possible_moves[loc_b]

        output = {}

        if first_player == self.player_index['A']:
            for a in next_a:
                for b in next_b:
                    new_loc_a = loc_a + self.action_delta[a]

                    if new_loc_a == loc_b:
                        new_loc_a = loc_a
                        new_loc_b = loc_b

                        state = 'A' + str(new_loc_a) + str(new_loc_b)

                    else:
                        new_loc_b = loc_b + self.action_delta[b]

                        if new_loc_b == new_loc_a:
                            new_loc_b = loc_b
                            state = 'A' + str(new_loc_a) + str(new_loc_b)
                        else:
                            state = 'A' + str(new_loc_a) + str(new_loc_b) if holder == 0 else 'B' + str(
                                new_loc_a) + str(new_loc_b)

                    output[(self.action_to_index[a], self.action_to_index[b])] = state

        else:
            for a in next_a:
                for b in next_b:
                    new_loc_b = loc_b + self.action_delta[b]

                    if new_loc_b == loc_a:
                        new_loc_b = loc_b
                        new_loc_a = loc_a
                        state = 'B' + str(new_loc_a) + str(new_loc_b)

                    else:
                        new_loc_a = loc_a + self.action_delta[a]
                        if new_loc_a == new_loc_b:
                            new_loc_a = loc_a
                            state = 'B' + str(new_loc_a) + str(new_loc_b)

                        else:
                            state = 'A' + str(new_loc_a) + str(new_loc_b) if holder == 1 else 'B' + str(
                                new_loc_a) + str(new_loc_b)

                    output[(self.action_to_index[a], self.action_to_index[b])] = state

        return output


    def solver(self, value_matrix):
        m, n = value_matrix.shape
        a = np.zeros((m * n, m * (m - 1) + m * n))

        col = 0
        for i in range(m):
            for j in range(m):
                if i != j:
                    a[i * n: (i + 1) * n, col] = value_matrix[j] - value_matrix[i]
                    col += 1

        np.fill_diagonal(a[:, col:], -1.0)

        A = matrix(a.transpose())
        b = matrix([0.] * (m * (m - 1) + m * n))
        c = matrix(-value_matrix.flatten())

        solvers.options['show_progress'] = False
        sol = solvers.lp(c, A, b)['x']
        probability_matrix = np.array(sol).reshape((m, n))

        return probability_matrix / np.sum(probability_matrix)


    def _q_learning(self, value_func, init_alpha, alpha_decay, init_q_value):
        alpha = init_alpha

        self.q_table = \
            np.ones((self.num_states * self.num_states * self.num_players,
                     self.num_actions, self.num_actions), dtype=np.float64) * init_q_value

        prv_q = 0
        self.error = np.empty((self.max_episode,), dtype=np.float64)

        for e in range(self.max_episode):
            state = np.random.choice(self.states_to_index.keys())
            state_index = self.states_to_index[state]
            done = False

            cnt_q = self.q_table[self.states_to_index[self.target_state], self.target_action[0], self.target_action[1]]
            self.error[e] = abs(cnt_q - prv_q)
            prv_q = cnt_q

            while not done:
                first_player = np.random.choice(self.player_index.values())

                available_action_states = self.get_next_moves(first_player, state)

                c = np.random.choice(len(available_action_states))
                action = available_action_states.keys()[c]

                new_state = available_action_states[action]
                reward = self.get_reward(new_state)[0]

                if reward != 0:
                    self.q_table[state_index, action[0], action[1]] = \
                        (1 - alpha) * self.q_table[state_index, action[0], action[1]] + alpha * ((1 - self.gamma) * reward)
                    done = True

                else:
                    value_a = value_func(new_state)

                    self.q_table[state_index, action[0], action[1]] = \
                        (1 - alpha) * self.q_table[state_index, action[0], action[1]] + alpha * (
                        (1 - self.gamma) * reward + self.gamma * value_a)

                state = new_state
                state_index = self.states_to_index[state]
                alpha = max(alpha * alpha_decay, self.min_alpha)


    def _friend_eq_func(self, state):
        state_index = self.states_to_index[state]

        loc_a = int(state[1])
        loc_b = int(state[2])

        rows = np.array([self.action_to_index[a] for a in self.possible_moves[loc_a]], dtype=np.intp)
        cols = np.array([self.action_to_index[b] for b in self.possible_moves[loc_b]], dtype=np.intp)

        possible_values = self.q_table[state_index][rows[:, np.newaxis], cols]
        return np.amax(possible_values)


    def _foe_eq_func(self, state):
        state_index = self.states_to_index[state]

        loc_a = int(state[1])
        loc_b = int(state[2])

        rows = np.array([self.action_to_index[a] for a in self.possible_moves[loc_a]], dtype=np.intp)
        cols = np.array([self.action_to_index[b] for b in self.possible_moves[loc_b]], dtype=np.intp)

        possible_values = self.q_table[state_index][rows[:, np.newaxis], cols]
        return np.amax(np.amin(possible_values, axis=0))


    def _uce_eq_func(self, state):
        state_index = self.states_to_index[state]

        loc_a = int(state[1])
        loc_b = int(state[2])

        rows = np.array([self.action_to_index[a] for a in self.possible_moves[loc_a]], dtype=np.intp)
        cols = np.array([self.action_to_index[b] for b in self.possible_moves[loc_b]], dtype=np.intp)

        possible_values = self.q_table[state_index][rows[:, np.newaxis], cols]

        if len(np.where(possible_values != 0)[0]) == 0:
            value = 0.

        else:
            probability_matrix = self.solver(possible_values)
            value = np.sum(possible_values * probability_matrix)

        return value


    def pure_Qlearning(self, init_alpha=0.9, alpha_decay=0.9999985, init_epsilon=1.0, epsilon_decay=0.999999, min_epsilon=0.001):
        epsilon = init_epsilon
        alpha = init_alpha

        self.q_table = np.zeros((self.num_states * self.num_states * self.num_players, self.num_actions), dtype=np.float64)

        prv_q = 0
        self.error = np.zeros((self.max_episode,), dtype=np.float64)

        for e in range(self.max_episode):
            state = np.random.choice(self.states_to_index.keys())
            state_index = self.states_to_index[state]
            done = False

            cnt_q = self.q_table[self.states_to_index[self.target_state], self.target_action[0]]
            self.error[e] = abs(cnt_q - prv_q)
            prv_q = cnt_q

            while not done:
                first_player = np.random.choice(self.player_index.values())
                available_action_states = self.get_next_moves(first_player, state)

                if np.random.random() < epsilon:
                    c = np.random.choice(len(available_action_states))
                    action = available_action_states.keys()[c]

                else:
                    loc_a = int(state[1])
                    loc_b = int(state[2])

                    moves_a = [self.action_to_index[a] for a in self.possible_moves[loc_a]]
                    moves_b = [self.action_to_index[b] for b in self.possible_moves[loc_b]]

                    v_a = max([(self.q_table[state_index][a], a) for a in moves_a])
                    v_b = min([(self.q_table[state_index][b], b) for b in moves_b])

                    action = (v_a[1], v_b[1])

                new_state = available_action_states[action]
                new_state_index = self.states_to_index[new_state]
                reward = self.get_reward(new_state)[0]

                value = np.max(self.q_table[new_state_index, :])

                done = reward != 0
                self.q_table[state_index, action[0]] = \
                    (1 - alpha) * self.q_table[state_index, action[0]] + alpha * ((1 - self.gamma) * reward + self.gamma * value * (not done))

                state = new_state
                alpha = max(alpha * alpha_decay, self.min_alpha)
                epsilon = max(epsilon * epsilon_decay, min_epsilon)


    def friend_Qlearning(self):
        return self._q_learning(self._friend_eq_func, 1.0, 0.99995, 100.)


    def foe_Qlearning(self):
        return self._q_learning(self._foe_eq_func, 1.0, 0.999997, 1.0)


    def uCE_Qlearning(self):
        return self._q_learning(self._uce_eq_func, 0.9, 0.999997, 1.0)








def visualization(x, y, filename):
    import matplotlib.pyplot as plt

    xs = np.where(y > 0.0)[0]
    ys = y[xs]
    plt.plot(xs, ys)
    plt.axis([0, len(y), 0, 0.5])
    plt.savefig(filename)
    plt.clf()


def get_data_from_pkl_file(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save_data_to_pkl_file(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
    print 'data have been saved to file: %s' % filename



def run_pure_Q(number_states=8, number_actions=5, save=False, plot=False, episode=1000000):
    agent = Qlearning(number_states, number_actions, max_episode=episode)
    agent.pure_Qlearning()
    error = agent.get_error()

    if save:
        save_data_to_pkl_file(y, 'pure_q.plk')
    if plot:
        visualization(range(len(error)), error, 'pure_q.png')


def run_friend_Q(number_states=8, number_actions=5, save=False, plot=False, episode=1000000):
    agent = Qlearning(number_states, number_actions, max_episode=episode)
    agent.friend_Qlearning()
    error = agent.get_error()

    if save:
        save_data_to_pkl_file(error, 'friend_q.plk')
    if plot:
        visualization(range(len(error)), error, 'friend_q.png')


def run_foe_Q(number_states=8, number_actions=5, save=False, plot=False, episode=1000000):
    agent = Qlearning(number_states, number_actions, max_episode=episode)
    agent.foe_Qlearning()
    error = agent.get_error()

    if save:
        save_data_to_pkl_file(error, 'foe_q.plk')
    if plot:
        visualization(range(len(error)), error, 'foe_q.png')


def run_ce_Q(number_states=8, number_actions=5, save=False, plot=False, episode=1000000):
    agent = Qlearning(number_states, number_actions, max_episode=episode)
    agent.uCE_Qlearning()
    error = agent.get_error()

    if save:
        save_data_to_pkl_file(error, 'ce_q.plk')
    if plot:
        visualization(range(len(error)), error, 'ce_q.png')



if __name__ == '__main__':
    #run_pure_Q()
    #run_friend_Q(plot=True)
    run_foe_Q(plot=True)
    #run_ce_Q()

