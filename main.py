#bokeh serve --show PycharmProjects/RL_lib/main.py

from simulation_framework.plots.bokeh import *
from simulation_framework.plots.bokeh_inventory import *
from simulation_framework.functions import *
import numpy as np
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs, Panel
import os

"""
Set up of OU-process
"""
theta = 10.
mu = 0.
sigma = .20

T = 10.
N = 100
ts = np.linspace(0., T, N)



"""Set up Bollinger Bands strategy as benchmark"""
num_iter = 100
pnlss = bollinger_strategy(theta, mu, sigma, ts, num_iter)
bollinger = [pnls[-1] for pnls in pnlss]

[]
"""
path = os.path.dirname(os.getcwd())
#filename = path + '\data\q_learning\q_learning_output.pickle'
"""

path = os.getcwd()

"""

Simple one_agents - no inventory!
"""
#filename = path + '\PycharmProjects\RL_lib\data\q_learning\q_learning_output.pickle'
#filename = path + '\PycharmProjects\RL_lib\data\dqn\dqn_output.pickle'
#filename = path + '\PycharmProjects\RL_lib\data\interactive_dqn\interactive_dqn_output.pickle'
filename = path + r'\PycharmProjects\deepRL_lib\data\backtester\syms.pickle'
page = bokeh_page_simple(filename, bollinger, q_learning=False)

"""
Agents with inventory
"""

#filename = path + '\PycharmProjects\RL_lib\data\q_learning\q_learning_inv_output_exp.pickle'
#filename = path + '\PycharmProjects\RL_lib\data\dqn\dqn_inv_output.pickle'
#filename = path + '\PycharmProjects\RL_lib\data\interactive_dqn\interactive_dqn_inv_exp_output.pickle'
#filename = path + '\PycharmProjects\RL_lib\data\interactive_dqn\interactive_dqn_inv_bs_output.pickle'
#page = bokeh_page_inventory(filename, bollinger, q_learning=True)


menu = Panel(child=page, title=filename)
tabs = Tabs(tabs=[menu])

curdoc().add_root(tabs)
curdoc().title = 'Agent Simplicity'