import random
from collections import deque
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sim_data.ou import *
import os
import pickle

"""
    def act(self, state):
        rv = np.random.rand()
        if rv <= self.epsilon:
            return np.random.choice(self.actions), 0
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0]), 1
"""

class DQNAgentInv:
    def __init__(self, state_size, action_size, buffer_size=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.abs_pure_action_reward = deque(maxlen=buffer_size)
        self.abs_pure_action_reward.append(0)
        self.exposure_limit = 5
        self.max_exposure = 5
        self.gamma = .9
        self.epsilon = 1.
        self.epsilon_decay = 0.998
        self.epsilon_min = .01
        self.learning_rate = .0001
        self.batch_size = 32
        self.actions = [i for i in range(self.action_size)]
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        greedy_dec = np.random.choice([0, 1], p=[self.epsilon, 1 - self.epsilon])
        cur_exp = state[0][-1]
        max_exposure = self.max_exposure
        to_action = int(self.action_size / 2)
        if abs(cur_exp) != max_exposure and abs(cur_exp) != max_exposure - 1:
            if greedy_dec == 1:
                act_values = self.model.predict(state)
                return np.argmax(act_values[0]), 1
            else:
                return np.random.choice([i for i in range(0, self.action_size)]), 0
        elif cur_exp == max_exposure:
            if greedy_dec == 1:
                act_values = self.model.predict(state)
                return np.argmax(act_values[0][:-2]), 1
            else:
                return np.random.choice([i for i in range(0, self.action_size - to_action )]), 0
        elif -1 * cur_exp == max_exposure:
            if greedy_dec == 1:
                act_values = self.model.predict(state)
                return np.argmax(act_values[0][2:]) + 2, 1
            else:
                return np.random.choice([i for i in range( to_action, self.action_size)]), 0
        elif cur_exp == max_exposure - 1:
            if greedy_dec == 1:
                act_values = self.model.predict(state)
                return np.argmax(act_values[0][:-1]), 1
            else:
                return np.random.choice([i for i in range(0, self.action_size-1)]), 0
        elif -1 * cur_exp == max_exposure - 1:
            if greedy_dec == 1:
                act_values = self.model.predict(state)
                return np.argmax(act_values[0][1:]) + 1, 1
            else:
                return np.random.choice([i for i in range(1, self.action_size)]), 0
        else:
            print(cur_exp)
            print("Unknown exposure")


    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                if abs(next_state[0][-1]) != self.max_exposure and abs(next_state[0][-1]) != self.max_exposure - 1:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                elif next_state[0][-1] == self.max_exposure:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0][:-2]))
                elif -1*next_state[0][-1] == self.max_exposure:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0][2:]))
                elif next_state[0][-1] == self.max_exposure-1:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0][:-1]))
                elif -1*next_state[0][-1] == self.max_exposure-1:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0][1:]))
                else:
                    print("Unknown exposure")

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def dqn_learning_inventory(theta, mu, sigma, ts, n_episodes):

    # DQN Agent Hyperparameters
    state_size = 3
    action_size = 5

    x0 = mu
    process = OrnsteinUhlenbeckProcess(theta, mu, sigma)
    agent = DQNAgentInv(state_size, action_size)

    setting = dict(theta=theta, mu=mu, sigma=sigma, ts=ts, n_episodes=n_episodes,
                   gamma=agent.gamma, epsilon=agent.epsilon, epsilon_decay=agent.epsilon_decay,
                   epsilon_min=agent.epsilon_min, learning_rate=agent.learning_rate,
                   batch_size=agent.batch_size, max_exposure= agent.max_exposure,
                   layers=3, nodes="64,128,64")

    to_action = int(agent.action_size / 2)
    scores = []

    price = []
    exposure = []
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
            a = action - to_action

            next_exp = cur_exp + int(a)
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

            next_state = (Xs[i+1], last_switch, next_exp)
            next_state = np.reshape(next_state, (1, state_size))

            """
            fraction = abs(next_exp) / agent.exposure_limit
            mean_pure_reward = np.max(agent.abs_pure_action_reward)
            inventory_penalty = fraction * mean_pure_reward
            pure_reward = int(a) * (Xs[i + 1] - Xs[i])
            agent.abs_pure_action_reward.append(abs(pure_reward))
            reward = pure_reward - inventory_penalty
            """

            reward = int(a)*(Xs[i+1] - Xs[i])
            #reward = int(next_exp) * (Xs[i + 1] - Xs[i])

            if t == ts[-2]:
                done = True

            cash -= int(a) * Xs[i]
            pnl = cash + next_exp * Xs[i + 1]

            cashs.append(cash)
            positions.append(cur_exp)
            pnls.append(pnl)

            cur_exp = next_exp

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {}'.format(e, n_episodes, pnls[-1]))
                scores.append(pnls[-1])
                break
        if len(agent.memory) > agent.batch_size:
            agent.replay()

    time = [i for i in range(len(price))]
    df_records = pd.DataFrame(dict(price=price, bucket=bucket, last_switch=last_price,
                                   states=states, decision=decision,
                                   episode=episode, greedy=greedy, epsilon=epsilon_val,
                                   time=time, exposure=exposure))

    output_dict = dict(scores=scores, data=df_records, setting=setting)
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
    output_dict = dqn_learning_inventory(theta, mu, sigma, ts, n_episodes)

    path_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_grandparent = os.path.dirname(path_parent)
    """Save data"""
    filename = path_grandparent + '\data\dqn\dqn_inv_output_final_act.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
