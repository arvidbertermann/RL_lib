from collections import deque
import pandas as pd
import numpy as np


class QAgent():

    def __init__(self, syms, config):
        self.__syms = syms
        self.__frequency = config['frequency']
        self.__alpha = config['alpha']
        self.__gamma = config['gamma']
        self.__eps = config['eps']
        self.__eps_decay = config['eps_decay']
        self.__eps_min = config['eps_min']
        self.__barriers = config['barriers']
        self.__memory = deque(maxlen=config['memory_length'])
        self.__tabular = self.build_tabular(config["buckets"])
        self.output_dict = dict(price=[], bucket=[],  last_switch=[], states=[], decision=[],
                                episode=[], greedy=[], epsilon_val=[])
        self.cash = 0
        self.pnl = 0
        self.pnls = []

    def prepare_start(self, signal):
        for j in range(len(signal)):
            self.append_memory(signal[j])

        self.output_dict["greedy"].append(0)
        self.__cur_exp = np.random.choice(["b", "s"])

        if self.__cur_exp == "b":
            return "44"
        else:
            return "-44"


    def build_tabular(self, buckets):
        q_tabular = dict()
        for price_bucket in range(1, buckets + 1):
            for exposure in [-1, 1]:
                for last_reversion in range(1, buckets + 1):
                    if exposure == -1:
                        q_tabular["-" + str(price_bucket) + str(last_reversion)] = dict(n=0, b=0)
                    else:
                        q_tabular[str(price_bucket) + str(last_reversion)] = dict(s=0, n=0)
        return q_tabular

    def act(self, state):
        greedy_dec = np.random.choice([0, 1], p=[self.__eps, 1 - self.__eps])
        self.output_dict["greedy"].append(greedy_dec)
        if self.__cur_exp == "s":
            if greedy_dec == 1:
                if self.__tabular[state]["n"] > self.__tabular[state]["b"]:
                    return "n"
                else:
                    return "b"
            else:
                return np.random.choice(["n", "b"])
        elif self.__cur_exp == "b":
            if greedy_dec == 1:
                if self.__tabular[state]["s"] > self.__tabular[state]["n"]:
                    return "s"
                else:
                    return "n"
            else:
                return np.random.choice(["s", "n"])


    def determine_next_state_and_reward(self, state, action, x_1, x_2):
        self.__memory.append(x_1)
        self.save_data(state, action, x_1)
        if action != "n":
            second_digit = state[-2]
            self.__cur_exp = action
        else:
            second_digit = state[-1]

        j = 0
        while j < len(self.__barriers) and x_2 > np.quantile(self.__memory, q=self.__barriers[j]):
            j += 1

        first_digit = j + 1

        if self.__cur_exp == "s":
            next_state = "-" + str(first_digit) + str(second_digit)
            reward = x_1 - x_2
        elif self.__cur_exp == "b":
            next_state = str(first_digit) + str(second_digit)
            reward = x_2 - x_1

        return next_state, reward

    def save_data(self, state, action, x_1):
        self.output_dict["price"].append(x_1)
        self.output_dict["states"].append(state)
        self.output_dict["bucket"].append(state[-2])
        self.output_dict["last_switch"].append(state[-1])
        self.output_dict["decision"].append(action)
        self.output_dict["epsilon_val"].append(self.__eps)


    def update(self, state, action, reward, next_state):
        if self.__cur_exp == "s":
            temp_list = ["n", "b"]
        elif self.__cur_exp == "b":
            temp_list = ["s", "n"]

        q_temp = self.__tabular[state][action]
        max_next_state = max(self.__tabular[next_state][temp_list[0]], self.__tabular[next_state][temp_list[1]])
        self.__tabular[state][action] = q_temp + self.__alpha * (reward + self.__gamma * max_next_state - q_temp)

    def update_eps(self):
        self.__eps = max(self.__eps_min, self.__eps * self.__eps_decay)


    def append_memory(self,x):
        self.__memory.append(x)

    def position_convert(self):
        self.to_df()

    def to_df(self):
        self.output_dict["time"] = [i for i in range(len(self.output_dict["price"]))]
        self.output_dict["episode"] = self.output_dict["time"]
        self.data = pd.DataFrame(self.output_dict)

    def get_tabular(self):
        return self.__tabular

    def update_signal_pnl(self, state, action, x_1, x_2):
        if action == "s":
            self.cash += 2 * x_1
            self.pnl = self.cash - x_2
        elif action == "b":
            self.cash -= 2 * x_1
            self.pnl = self.cash + x_2
        else:
            if int(state) < 0:
                self.pnl = self.cash - x_2 + x_1
            else:
                self.pnl = self.cash + x_2 - x_1

    def append_pnl_and_reset(self):
        self.pnls.append(self.pnl)
        self.cash = 0

    def get_exp(self):
        return self.__cur_exp