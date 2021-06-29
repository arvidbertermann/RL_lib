import pandas as pd
import datetime as dt
from lob_framework.agents.q_learning import QAgent
from lob_framework.data_providers import KDBDataProvider
from lob_framework.execution import MPExecution
import pickle

class Backtester():
    def __init__(self, agent, execution):
        self.agent = agent
        self.execution = execution
        self.data = self.execution.lob
        self._update_eps = 100
        self._next_episode = 1_000

    def prepare_data(self):
        markers = ['exch_time', 'px', 'qtype']
        df = self.data[markers]
        df = df[df['qtype'] == 1][markers[:-1]]

        df = df[self.filtering(df)]
        df = df.set_index('exch_time')
        return df

    def filtering(self, df, column_name='px'):
        '''
        Input:
        df - Data Frame, what we will filter,
        column_name - the column that is being filtered by
        Output:
        Bool indexes of the original Data Frame that survived filtering
        '''
        ret = df[column_name] - df[column_name].shift(1)
        ret_std = ret.ewm(span=10).std().fillna(0)
        return abs(ret_std) < 8 * ret_std.mean()

    def __prepare_inputs(self):
        df = self.prepare_data()
        ewma1 = df['px'].ewm(alpha=.1).mean()
        feature_1 = df['px'] - ewma1
        ewma2 = df['px'].ewm(alpha=.1 / 2.).mean()
        feature_2 = ewma1 - ewma2
        features = pd.DataFrame({
            'feature_1': feature_1,
            'feature_2': feature_2,
        })
        return df, features

    def run(self, q_learning=False):
        df, features = self.__prepare_inputs()
        signal = features["feature_1"]
        stop = df.index[-1]
        time = df.index

        first_state = self.agent.prepare_start(signal)
        state, _ = self.agent.determine_next_state_and_reward(first_state, "n", signal[0], signal[1])
        self.agent.first_pnl(signal[0], signal[1])
        exp = self.agent.get_exp()
        _, price, next_price = self.execution.execute_order(time[0], exp)
        self.execution.first_pnl(exp, price, next_price)

        episode = 1
        i = 1
        while time[i] < stop:
            action = self.agent.act(state)
            next_state, _ = self.agent.determine_next_state_and_reward(state, action, signal[i], signal[i + 1])
            self.agent.update_signal_pnl(state, action, signal[i], signal[i + 1])


            exp = self.agent.get_exp()
            reward, price, next_price = self.execution.execute_order(time[i], exp)
            self.execution.update_pnl(state, action, price, next_price)


            self.agent.update(state, action, reward, next_state)

            i += 1
            if i % self._update_eps == 0:
                self.agent.update_eps()
                if i % self._next_episode == 0:
                    self.agent.append_pnl()
                    self.execution.append_pnl(exp, next_price)
                    print("Episode " + str(episode) + ": " + str(self.execution.pnls[-1]))
                    episode += 1
            state = next_state


        self.agent.to_df()
        output_dict = dict(scores=self.agent.pnls, data=self.agent.data)
        if q_learning:
            output_dict["tabular"] = self.agent.get_tabular()

        return output_dict

if __name__ == '__main__':

    kdb_data_path = r'C:\Users\arvid\kdb_bmll_v1.0_snapshot'

    kdb_port = 9000
    date = dt.date(2021, 2, 19)
    end_date = dt.date(2021, 2, 19)
    date_str = dt.datetime.strftime(date, "%Y.%m.%d")
    end_date_str = dt.datetime.strftime(end_date, "%Y.%m.%d")
    syms = ("BBG00BFJSR16",)
    sym = "BBG00BFJSR16"
    dp = KDBDataProvider(kdb_data_path, kdb_port, sym, date_str, end_date_str)

    q_config_1 = dict(
        frequency='1S',
        alpha=0.01,
        gamma=0.9,
        eps=1,
        eps_decay=0.998,
        eps_min=0.01,
        barriers=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        memory_length=1000,
        buckets=7)

    agent = QAgent(sym, q_config_1)
    execution = MPExecution(dp.data)
    backtester = Backtester(agent, execution)
    output_dict = backtester.run()
    filename = str(date_str) + "_" + str(sym) + ".pickle"
    print(filename)
    path = r'C:\Users\arvid\PycharmProjects\RL_lib\data\backtester\_' + str(filename)
    with open(path, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)






