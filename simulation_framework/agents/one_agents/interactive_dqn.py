import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sim_data.ou import *
import os
import pickle
from collections import deque
import random

class InteractiveDQNAgent:
    def __init__(self, state_size, action_size, buffer_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_buy = deque(maxlen=buffer_size)
        self.memory_sell = deque(maxlen=buffer_size)
        self.gamma = .9
        self.epsilon = 1.
        self.epsilon_decay = 0.998
        self.epsilon_min = .01
        self.learning_rate = .0001
        self.batch_size = 32
        self.buy_agent = self._build_model()
        self.sell_agent = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size-1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        if state[0][-1] == 1:
            self.memory_sell.append((state, action, reward, next_state, done))
        elif state[0][-1] == -1:
            self.memory_buy.append((state, action, reward, next_state, done))
        else:
            print(state[0][-1])
            print("Unknown exposure")

    def act(self, state):
        rv = np.random.rand()
        if state[0][-1] == 1:
            if rv <= self.epsilon:
                return np.random.choice([0, 1]), 0
            else:
                act_values = self.sell_agent.predict(state)
                return np.argmax(act_values[0]), 1
        elif state[0][-1] == -1:
            if rv <= self.epsilon:
                return np.random.choice([1, 2]), 0
            else:
                act_values = self.buy_agent.predict(state)
                return np.argmax(act_values[0]) + 1, 1
        else:
            print(state[0][-1])
            print("Unknown exposure")


    def replay(self):
        if len(self.memory_buy) > self.batch_size:
            minibatch_buy = random.sample(self.memory_buy, self.batch_size)
            for state, action, reward, next_state, done in minibatch_buy:
                target = reward
                if not done:
                    if action == 1:
                        target = (reward + self.gamma * np.amax(self.buy_agent.predict(next_state)[0]))
                    elif action == 2:
                        target = (reward + self.gamma * np.amax(self.sell_agent.predict(next_state)[0]))
                    else:
                        print(action)
                        print("Unknown action")

                target_f = self.buy_agent.predict(state)
                target_f[0][action-1] = target
                self.buy_agent.fit(state, target_f, epochs=1, verbose=0)

        if len(self.memory_sell) > self.batch_size:
            minibatch_sell = random.sample(self.memory_sell, self.batch_size)
            for state, action, reward, next_state, done in minibatch_sell:
                target = reward
                if not done:
                    if action == 0:
                        target = (reward + self.gamma * np.amax(self.buy_agent.predict(next_state)[0]))
                    elif action == 1:
                        target = (reward + self.gamma * np.amax(self.sell_agent.predict(next_state)[0]))
                    else:
                        print(action)
                        print("Unknown action")

                target_f = self.sell_agent.predict(state)
                target_f[0][action] = target
                self.sell_agent.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def interactive_dqn_learning(theta, mu, sigma, ts, n_episodes):

    # DQN Agent Hyperparameters
    state_size = 3
    action_size = 3
    x0 = mu
    process = OrnsteinUhlenbeckProcess(theta, mu, sigma)
    agent = InteractiveDQNAgent(state_size, action_size)

    setting = dict(theta=theta, mu=mu, sigma=sigma, ts=ts, n_episodes=n_episodes,
                   gamma=agent.gamma, epsilon=agent.epsilon, epsilon_decay=agent.epsilon_decay,
                   epsilon_min=agent.epsilon_min, learning_rate=agent.learning_rate,
                   batch_size=agent.batch_size,
                   layers=3, nodes=48)

    scores = []

    price = []
    bucket = []
    last_price = []
    states = []
    decision = []
    episode = []
    greedy = []
    epsilon_val = []

    for e in range(n_episodes):
        Xs = generate_ornstein_uhlenbeck_process(x0, ts, process)
        cur_exp = np.random.choice([-1, 1])
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
            a = action - 1

            price.append(Xs[i])
            bucket.append(0)
            last_price.append(last_switch)
            states.append(0)
            decision.append(a)
            episode.append(e)
            greedy.append(greedy_dec)
            epsilon_val.append(agent.epsilon)

            if a != 0:
                cur_exp = a
                last_switch = Xs[i]

            next_state = (Xs[i + 1], last_switch, cur_exp)
            next_state = np.reshape(next_state, (1, state_size))
            if t == ts[-2]:
                done = True

            if a == -1:
                cash += 2 * Xs[i]
                pnl = cash - Xs[i + 1]
            elif a == 1:
                cash -= 2 * Xs[i]
                pnl = cash + Xs[i + 1]
            else:
                if cur_exp == -1:
                    pnl = cash - Xs[i + 1] + Xs[i]
                elif cur_exp == 1:
                    pnl = cash + Xs[i + 1] - Xs[i]
                else:
                    pnl = -1000
                    print("Unknown exposure")

            cashs.append(cash)
            positions.append(cur_exp)
            pnls.append(pnl)


            reward = cur_exp*(Xs[i+1]-Xs[i])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {}'.format(e, n_episodes, pnls[-1]))
                scores.append(pnls[-1])
                break
        if len(agent.memory_buy) > agent.batch_size or len(agent.memory_sell) > agent.batch_size:
            agent.replay()
    time = [i for i in range(len(price))]
    for k in range(len(decision)):
        if decision[k] == 1:
            decision[k] = "b"
        elif decision[k] == -1:
            decision[k] = "s"
        elif decision[k] == 0:
            decision[k] = "n"
        else:
            print("Unknown exposure")

    df_records = pd.DataFrame(dict(price=price, bucket=bucket, last_switch=last_price,
                                   states=states, decision=decision,
                                   episode=episode, greedy=greedy, epsilon=epsilon_val, time=time))

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
    output_dict = interactive_dqn_learning(theta, mu, sigma, ts, n_episodes)

    path_parent = os.path.dirname(os.getcwd())
    path_grandparent = os.path.dirname(path_parent)
    """Save data"""
    filename = path_grandparent + '\data\interactive_dqn\interactive_dqn_output_final.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)