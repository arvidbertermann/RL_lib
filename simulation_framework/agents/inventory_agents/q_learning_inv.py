from simulation_framework.sim_data.ou import *
from collections import deque
import pandas as pd
import numpy as np
import pickle
import os

"""
Q-Tabular:
3 Digits:
Zero Digit: Exposure, which is either + for long or - for short. Range (-5, 5)
First Digit: Price bucket of OU-process. 1 highest to buckets lowest
Second Digit: Last bucket of "reversion" of position (long to short (s) or short to long (b))
"""


def q_tabular_inv(buckets=7, max_exposure=5):
    q_tabular = dict()
    for exposure in range(-max_exposure, max_exposure + 1):
        for price_bucket in range(1, buckets + 1):
            for last_reversion in range(1, buckets + 1):
                if abs(exposure) != max_exposure and abs(exposure) != max_exposure - 1:
                    q_tabular[str(exposure) + str(price_bucket) + str(last_reversion)] = {"-2": 0, "-1": 0, "0": 0, "1": 0, "2": 0}
                elif exposure == max_exposure - 1:
                    q_tabular[str(exposure) + str(price_bucket) + str(last_reversion)] = {"-2": 0, "-1": 0, "0": 0, "1": 0}
                elif -1*exposure == max_exposure - 1:
                    q_tabular[str(exposure) + str(price_bucket) + str(last_reversion)] = {"-1": 0, "0": 0, "1": 0, "2": 0}
                elif exposure == max_exposure:
                    q_tabular[str(exposure) + str(price_bucket) + str(last_reversion)] = {"-2": 0, "-1": 0, "0": 0}
                elif -1*exposure == max_exposure:
                    q_tabular[str(exposure) + str(price_bucket) + str(last_reversion)] = {"0": 0, "1": 0, "2": 0}
                else:
                    print("Warning: Undetected exposure")
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


def q_learning_agent_inv(max_exposure, q_tabular, theta, mu, sigma, ts, alpha, gamma, n_episodes,
                     epsilon, epsilon_decay, barriers, memory_len):
    """
    Initialize memory by sampling a OU-process of length 100
    """

    setting = dict(theta=theta, mu=mu, sigma=sigma, ts=ts, alpha=alpha, gamma=gamma, n_episodes=n_episodes,
                   epsilon=epsilon, epsilon_decay=epsilon_decay, barriers=barriers, memory_len=memory_len,
                   max_exposure=max_exposure)

    process = OrnsteinUhlenbeckProcess(theta, mu, sigma)
    Xs = generate_ornstein_uhlenbeck_process(mu, ts, process)
    memory = deque(maxlen=memory_len)
    for i in range(len(Xs)):
        memory.append(Xs[i])

    x0 = mu
    scores = []

    """
    Empty list to record decisions of agent over time
    """
    price = []
    exposure = []
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
        cashs = [cash]
        pnls = [cash]
        position = 0
        positions = [position]
        done = False

        cur_exp = 0 #np.random.choice([i for i in range(-max_exposure, max_exposure+1)])
        state = str(cur_exp) + str(first_digit) + str(second_digit)

        for i, t in enumerate(ts[:-1]):
            greedy_dec = np.random.choice([0, 1], p=[epsilon, 1 - epsilon])

            if abs(cur_exp) != max_exposure and abs(cur_exp) != max_exposure -1:
                if greedy_dec == 1:
                    action = find_greedy_action(q_tabular[state])
                else:
                    action = np.random.choice([-2, -1, 0, 1, 2])
            elif cur_exp == max_exposure:
                if greedy_dec == 1:
                    action = find_greedy_action(q_tabular[state])
                else:
                    action = np.random.choice([-2, -1, 0])
            elif -1*cur_exp == max_exposure:
                if greedy_dec == 1:
                    action = find_greedy_action(q_tabular[state])
                else:
                    action = np.random.choice([0, 1, 2])
            elif cur_exp == max_exposure - 1:
                if greedy_dec == 1:
                    action = find_greedy_action(q_tabular[state])
                else:
                    action = np.random.choice([-2, -1, 0, 1])
            elif -1*cur_exp == max_exposure - 1:
                if greedy_dec == 1:
                    action = find_greedy_action(q_tabular[state])
                else:
                    action = np.random.choice([-1, 0, 1, 2])
            else:
                print(cur_exp)
                print("Unknown exposure")



            next_exp = cur_exp + int(action)


            price.append(Xs[i])
            exposure.append(cur_exp)
            bucket.append(first_digit)
            last_switch.append(second_digit)
            states.append(state)
            decision.append(action)
            episode.append(e)
            greedy.append(greedy_dec)
            epsilon_val.append(epsilon)

            if (next_exp <= 0 and cur_exp > 0) or (next_exp >= 0 and cur_exp > 0):
                second_digit = first_digit

            j = 0
            while j < len(barriers) and Xs[i + 1] > np.quantile(memory, q=barriers[j]):
                j += 1

            first_digit = j + 1
            next_state = str(next_exp) + str(first_digit) + str(second_digit)
            #reward = cur_exp*(Xs[i + 1] - Xs[i]) #questionable
            #reward = next_exp * (Xs[i + 1] - Xs[i])
            reward = int(action) * (Xs[i + 1] - Xs[i])

            if t == ts[-2]:
                done = True

            cash -= int(action)*Xs[i]
            pnl = cash + next_exp*Xs[i + 1]

            cashs.append(cash)
            positions.append(cur_exp)
            pnls.append(pnl)
            cur_exp = next_exp

            q_temp = q_tabular[state][str(action)]
            max_next_state = max(q_tabular[next_state].values())
            q_tabular[state][str(action)] = q_temp + alpha * (reward + gamma * max_next_state - q_temp)
            state = next_state
            memory.append(Xs[i])

            if done:
                print('episode: {}/{}, score: {}'.format(e, n_episodes, pnls[-1]))
                scores.append(pnls[-1])
                break

        epsilon = max(epsilon * epsilon_decay, 0.1)

    time = [i for i in range(len(price))]
    df_records = pd.DataFrame(dict(price=price, bucket=bucket, last_switch=last_switch,
                                   states=states, decision=decision,
                                   episode=episode, greedy=greedy, epsilon=epsilon_val,
                                   time=time, exposure=exposure))
    output_dict = dict(scores=scores, q_tabular=q_tabular, data=df_records, setting=setting)
    return output_dict


def find_greedy_action(state_dict):
    max_actions = ["0"]
    cur_val = state_dict[max_actions[0]]
    for key in state_dict.keys():
        if state_dict[key] > cur_val:
            max_actions = [key]
            cur_val = state_dict[key]
        elif state_dict[key] == cur_val and key != "0":
            max_actions.append(key)

    return int(np.random.choice(max_actions))


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
    max_exposure = 5
    q_tabular = q_tabular_inv(buckets=7, max_exposure=max_exposure)
    print(q_tabular)

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

    output_dict = q_learning_agent_inv(max_exposure, q_tabular, theta, mu, sigma, ts, alpha, gamma, n_episodes,
                     epsilon, epsilon_decay, barriers, memory_len)

    path_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_grandparent = os.path.dirname(path_parent)
    """Save data"""
    filename = path_grandparent + '\data\q_learning\q_learning_inv_output_action.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
