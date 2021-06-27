import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sim_data.ou import *
import os
import pickle
from collections import deque
import random
import sys

class InteractiveDQNAgentInvBS:
    def __init__(self, state_size, action_size, buffer_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_buy = deque(maxlen=buffer_size)
        self.memory_sell = deque(maxlen=buffer_size)
        self.max_exposure = 5
        self.gamma = .9
        self.epsilon = 1.
        self.epsilon_decay = 0.995
        self.epsilon_min = .01
        self.learning_rate = .0001
        self.batch_size = 32
        self.buy_agent = self._build_model()
        self.sell_agent = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        if action > 0:
            self.memory_buy.append((state, action, reward, next_state, done))
        elif action < 0:
            self.memory_sell.append((state, action, reward, next_state, done))
        elif state[0][-1] == self.max_exposure:
            self.memory_sell.append((state, action, reward, next_state, done))
        elif -1*state[0][-1] == self.max_exposure:
            self.memory_buy.append((state, action, reward, next_state, done))
        else:
            self.memory_sell.append((state, action, reward, next_state, done))
            self.memory_buy.append((state, action, reward, next_state, done))

    def act(self, state):
        rv = np.random.rand()
        to_action = 2

        if abs(state[0][-1]) != self.max_exposure and abs(state[0][-1]) != self.max_exposure - 1:
            if rv <= self.epsilon:
                return np.random.choice([-2, -1, 0, 1, 2]), 0
            else:
                act_values_b = self.buy_agent.predict(state)
                act_values_s = self.sell_agent.predict(state)
                if np.amax(act_values_b[0]) > np.amax(act_values_s[0]):
                    return np.argmax(act_values_b[0]), 1
                elif np.amax(act_values_b[0]) < np.amax(act_values_s[0]):
                    return np.argmax(act_values_s[0]) - to_action, 1
                else:
                    return 0, 1

        elif state[0][-1] == self.max_exposure:
            if rv <= self.epsilon:
                return np.random.choice([-2, -1, 0]), 0
            else:
                act_values_s = self.sell_agent.predict(state)
                return np.argmax(act_values_s[0]) - to_action, 1
        elif -1*state[0][-1] == self.max_exposure:
            if rv <= self.epsilon:
                return np.random.choice([0, 1, 2]), 0
            else:
                act_values_b = self.buy_agent.predict(state)
                return np.argmax(act_values_b[0]), 1
        elif state[0][-1] == self.max_exposure - 1:
            if rv <= self.epsilon:
                return np.random.choice([-2, -1, 0, 1]), 0
            else:
                act_values_b = self.buy_agent.predict(state)
                act_values_s = self.sell_agent.predict(state)
                if np.amax(act_values_b[0][:-1]) > np.amax(act_values_s[0]):
                    return np.argmax(act_values_b[0][:-1]), 1
                elif np.amax(act_values_b[0][:-1]) < np.amax(act_values_s[0]):
                    return np.argmax(act_values_s[0]) - to_action, 1
                else:
                    return 0, 1
        elif -1*state[0][-1] == self.max_exposure - 1:
            if rv <= self.epsilon:
                return np.random.choice([-1, 0, 1, 2]), 0
            else:
                act_values_b = self.buy_agent.predict(state)
                act_values_s = self.sell_agent.predict(state)
                if np.amax(act_values_b[0]) > np.amax(act_values_s[0][1:]):
                    return np.argmax(act_values_b[0]), 1
                elif np.amax(act_values_b[0]) < np.amax(act_values_s[0][1:]):
                    return np.argmax(act_values_s[0][1:]) - 1, 1
                else:
                    return 0, 1

    def replay(self):
        if len(self.memory_buy) > self.batch_size:
            minibatch_buy = random.sample(self.memory_buy, self.batch_size)
            for state, action, reward, next_state, done in minibatch_buy:
                target = reward
                if not done:
                    if abs(next_state[0][-1]) != self.max_exposure and abs(next_state[0][-1]) != self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0])
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * max(buy, sell))
                    elif next_state[0][-1] == self.max_exposure:
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * sell)
                    elif next_state[0][-1] == self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0][:-1])
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * max(buy, sell))
                    elif -1*next_state[0][-1] == self.max_exposure:
                        buy = np.amax(self.buy_agent.predict(next_state)[0][:-1])
                        target = (reward + self.gamma * buy)
                    elif -1*next_state[0][-1] == self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0])
                        sell = np.amax(self.sell_agent.predict(next_state)[0][1:])
                        target = (reward + self.gamma * max(buy, sell))
                    else:
                        print("Unknown exposure")

                target_f = self.buy_agent.predict(state)
                target_f[0][action] = target
                self.buy_agent.fit(state, target_f, epochs=1, verbose=0)

        if len(self.memory_sell) > self.batch_size:
            minibatch_sell = random.sample(self.memory_sell, self.batch_size)
            for state, action, reward, next_state, done in minibatch_sell:
                target = reward
                if not done:
                    if abs(next_state[0][-1]) != self.max_exposure and abs(next_state[0][-1]) != self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0])
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * max(buy, sell))
                    elif next_state[0][-1] == self.max_exposure:
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * sell)
                    elif next_state[0][-1] == self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0][:-1])
                        sell = np.amax(self.sell_agent.predict(next_state)[0])
                        target = (reward + self.gamma * max(buy, sell))
                    elif -1 * next_state[0][-1] == self.max_exposure:
                        buy = np.amax(self.buy_agent.predict(next_state)[0][:-1])
                        target = (reward + self.gamma * buy)
                    elif -1 * next_state[0][-1] == self.max_exposure - 1:
                        buy = np.amax(self.buy_agent.predict(next_state)[0])
                        sell = np.amax(self.sell_agent.predict(next_state)[0][1:])
                        target = (reward + self.gamma * max(buy, sell))
                    else:
                        print("Unknown exposure")

                target_f = self.sell_agent.predict(state)
                target_f[0][action + 2] = target
                self.sell_agent.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def interactive_dqn_learning_inventory_bs(theta, mu, sigma, ts, n_episodes):

    # DQN Agent Hyperparameters
    state_size = 3
    action_size = 3

    x0 = mu
    process = OrnsteinUhlenbeckProcess(theta, mu, sigma)
    agent = InteractiveDQNAgentInvBS(state_size, action_size)
    to_action = int(agent.action_size / 2)
    scores = []

    price = []
    exposure =[]
    bucket = []
    last_price = []
    states = []
    decision = []
    episode = []
    greedy = []
    epsilon_val = []

    for e in range(n_episodes):
        Xs = generate_ornstein_uhlenbeck_process(x0, ts, process)
        cur_exp = 0
        last_switch = Xs[0]
        state = (Xs[0], last_switch, cur_exp)
        state = np.reshape(state, (1, state_size))
        cash = 0.
        cashs = [cash]
        pnls = [cash]
        position = 0
        positions = [position]
        done = False
        for i, t in enumerate(ts[:-1]):
            action, greedy_dec = agent.act(state)
            a = int(action)

            next_exp = cur_exp + int(a)
            if abs(next_exp) > agent.max_exposure:
                print("Mistake")
                sys.exit(0)

            price.append(Xs[i])
            exposure.append(cur_exp)
            bucket.append(0)
            last_price.append(last_switch)
            states.append(0)
            decision.append(a)
            episode.append(e)
            greedy.append(greedy_dec)
            epsilon_val.append(agent.epsilon)

            if (next_exp <= 0 and cur_exp > 0) or (next_exp >= 0 and cur_exp > 0):
                last_switch = Xs[i]

            next_state = (Xs[i + 1], last_switch, int(next_exp))
            next_state = np.reshape(next_state, (1, state_size))

            reward = int(a) * (Xs[i + 1] - Xs[i])

            if t == ts[-2]:
                done = True

            cash -= int(a) * Xs[i]
            pnl = cash + next_exp * Xs[i + 1]

            cashs.append(cash)
            positions.append(cur_exp)
            pnls.append(pnl)

            cur_exp = next_exp

            agent.remember(state, int(action), reward, next_state, done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {}'.format(e, n_episodes, pnls[-1]))
                scores.append(pnls[-1])
                break
        if len(agent.memory_buy) > agent.batch_size or len(agent.memory_sell) > agent.batch_size:
            agent.replay()
    time = [i for i in range(len(price))]

    df_records = pd.DataFrame(dict(price=price, bucket=bucket, last_switch=last_price,
                                   states=states, decision=decision,
                                   episode=episode, greedy=greedy, epsilon=epsilon_val,
                                   time=time, exposure=exposure))

    output_dict = dict(scores=scores, data=df_records)
    return output_dict

if __name__ == '__main__':
    """
    Set up of OU-process
    """
    theta = 10.
    mu = 0.
    sigma = .20

    T = 10.
    N = 100
    ts = np.linspace(0., T, N)
    n_episodes = 2_500
    output_dict = interactive_dqn_learning_inventory_bs(theta, mu, sigma, ts, n_episodes)

    path_parent = os.path.dirname(os.getcwd())
    path_grandparent = os.path.dirname(path_parent)
    """Save data"""
    filename = path_grandparent + '\data\interactive_dqn\interactive_dqn_inv_bs_output.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
