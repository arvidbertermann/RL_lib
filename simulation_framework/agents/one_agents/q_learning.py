from simulation_framework.sim_data.ou import *
from collections import deque
import pandas as pd
import numpy as np
import pickle
import os

"""
Q-Tabular:
Sign + 2 Digits:
Sign : Exposure, which is either + for long or - for short. Simple version here,
                it is not possible to have zero exposure.
First Digit: Price bucket of OU-process. 1 highest to buckets lowest
Second Digit: Last bucket of "reversion" of position (long to short (s) or short to long (b))
"""


def q_tabular(buckets=7):
    q_tabular = dict()
    for price_bucket in range(1, buckets + 1):
        for exposure in [-1, 1]:
            for last_reversion in range(1, buckets + 1):
                if exposure == -1:
                    q_tabular["-" + str(price_bucket) + str(last_reversion)] = dict(n=0, b=0)
                else:
                    q_tabular[str(price_bucket) + str(last_reversion)] = dict(s=0, n=0)
    return q_tabular


"""
q_tabular : Initialized Action-State_pair values
theta, mu, sigma : Parameters of OU-process
ts : Timeline
alpha : Stepsize parameter in Q-Learning updating rule
gamma : discount factor in expected reward
n_episodes : Number of episodes the agent will be trained
epsilon : Exploration factor in epsilon greedy strategy
epsilon_decay: After each episode epsilon *= epsilon_decay. Hence, balances exploration and
            exploitation over time.
barriers: Quantiles that divide price range into buckets
memory_len : Length of memory that determines quantiles
"""


def q_learning_agent(q_tabular, theta, mu, sigma, ts, alpha, gamma, n_episodes,
                     epsilon, epsilon_decay, barriers, memory_len):

    setting = dict(theta=theta, mu=mu, sigma=sigma, ts=ts, alpha=alpha, gamma=gamma, n_episodes=n_episodes,
                   epsilon=epsilon, epsilon_decay=epsilon_decay, barriers=barriers, memory_len=memory_len)
    """
    Initialize memory by sampling a OU-process of length 100
    """
    process = OrnsteinUhlenbeckProcess(theta, mu, sigma)
    Xs = generate_ornstein_uhlenbeck_process(mu, ts, process)
    memory = deque(maxlen=memory_len)
    for i in range(len(Xs)):
        memory.append(Xs[i])

    x0 = mu

    done = False
    scores = []

    """
    Empty list to record decisions of agent over time
    """
    price = []
    bucket = []
    last_switch = []
    states = []
    decision = []
    episode = []
    greedy = []
    epsilon_val = []

    for e in range(n_episodes):
        Xs = generate_ornstein_uhlenbeck_process(x0, ts, process)
        first_digit = 4
        second_digit = 4
        cash = 0.
        pnl = .0
        cashs = [cash]
        pnls = [cash]
        position = 0
        positions = [position]
        done = False
        if np.random.choice(["s", "b"]) == "s":
            cur_exp = "s"
            state = "-" + str(first_digit) + str(second_digit)
        else:
            cur_exp = "b"
            state = str(first_digit) + str(second_digit)

        for i, t in enumerate(ts[:-1]):
            greedy_dec = np.random.choice([0, 1], p=[epsilon, 1 - epsilon])
            if cur_exp == "s":
                if greedy_dec == 1:
                    if q_tabular[state]["n"] > q_tabular[state]["b"]:
                        action = "n"
                    else:
                        action = "b"
                else:
                    action = np.random.choice(["n", "b"])
            elif cur_exp == "b":
                if greedy_dec == 1:
                    if q_tabular[state]["s"] > q_tabular[state]["n"]:
                        action = "s"
                    else:
                        action = "n"
                else:
                    action = np.random.choice(["s", "n"])
            else:
                print("Unknown exposure")

            price.append(Xs[i])
            bucket.append(first_digit)
            last_switch.append(second_digit)
            states.append(state)
            decision.append(action)
            episode.append(e)
            greedy.append(greedy_dec)
            epsilon_val.append(epsilon)

            if action != "n":
                second_digit = first_digit
                cur_exp = action

            j = 0
            while j < len(barriers) and Xs[i + 1] > np.quantile(memory, q=barriers[j]):
                j += 1

            first_digit = j + 1

            if cur_exp == "s":
                next_state = "-" + str(first_digit) + str(second_digit)
                reward = Xs[i] - Xs[i + 1]
            elif cur_exp == "b":
                next_state = str(first_digit) + str(second_digit)
                reward = Xs[i + 1] - Xs[i]
            else:
                print("Unknown exposure")

            if t == ts[-2]:
                done = True

            if action == "s":
                cash += 2 * Xs[i]
                pnl = cash - Xs[i + 1]
            elif action == "b":
                cash -= 2 * Xs[i]
                pnl = cash + Xs[i + 1]
            else:
                if cur_exp == "s":
                    pnl = pnl - Xs[i + 1] + Xs[i] #mistake
                elif cur_exp == "b":
                    pnl = pnl + Xs[i + 1] - Xs[i] #mistake
                else:
                    print("Unknown exposure")

            cashs.append(cash)
            positions.append(cur_exp)
            pnls.append(pnl)

            if cur_exp == "s":
                temp_list = ["n", "b"]
            elif cur_exp == "b":
                temp_list = ["s", "n"]
            else:
                temp_list = []
                print("Unknown exposure")

            q_temp = q_tabular[state][action]
            max_next_state = max(q_tabular[next_state][temp_list[0]], q_tabular[next_state][temp_list[1]])
            q_tabular[state][action] = q_temp + alpha * (reward + gamma * max_next_state - q_temp)
            state = next_state
            memory.append(Xs[i])

            if done:
                print('episode: {}/{}, score: {}'.format(e, n_episodes, pnls[-1]))
                scores.append(pnls[-1])
                break

        epsilon = max(epsilon * epsilon_decay, 0.01)

    time = [i for i in range(len(price))]
    df_records = pd.DataFrame(dict(price=price, bucket=bucket, last_switch=last_switch,
                                   states=states, decision=decision,
                                   episode=episode, greedy=greedy, epsilon=epsilon_val, time=time))
    output_dict = dict(scores=scores, q_tabular=q_tabular, data=df_records, setting=setting)
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

    """
    Initialize Action-State_pair values
    """
    q_tabular = q_tabular(buckets=7)

    """
    Q Learning Parameters
    """
    alpha = .01
    gamma = 0.9
    n_episodes = 2_500
    epsilon = 1
    epsilon_decay = 0.998
    barriers = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    memory_len = 1_000

    output_dict = q_learning_agent(q_tabular, theta, mu, sigma, ts, alpha, gamma, n_episodes,
                                   epsilon, epsilon_decay, barriers, memory_len)

    path_parent = os.path.dirname(os.getcwd())
    path_grandparent = os.path.dirname(os.path.dirname(path_parent))
    """Save data"""
    filename = path_grandparent + '\data_new\q_learning\q_learning_output.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


