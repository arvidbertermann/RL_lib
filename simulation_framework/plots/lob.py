import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


p_b = [97, 98, 99]
p_a = [100, 101, 102]
q_b_1 = np.random.randint(10, 100, size=3)
q_a_1 = np.random.randint(10, 100, size=3)
plt.bar(p_b, q_b_1, label="Bid Limit Orders", color="green", alpha=0.5, width=0.5)
plt.bar(p_a, q_a_1, label="Ask Limit Orders", color="red", alpha=0.5, width=0.5)
y_max = max(max(q_b_1), max(q_a_1))
plt.plot([99.5, 99.5], [0, y_max], label="mid price", color="black", linestyle="--", alpha=0.5)
plt.xticks(p_b+p_a, ["97", "98", "99", "100", "101", "102"], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Volume", fontsize=16)
plt.xlabel("Price", fontsize=16)
plt.title("Limit order Book at $t_0$", fontsize=18)
plt.legend(fontsize=14)
plt.show()

q_b_1[-2] *= 0.3
plt.bar(x=[98], height=[0.7*q_b_1[-2]/0.3], width=0.5, bottom=q_b_1[-2], label = "Sell Market Order", color="grey")
plt.bar(x=[99], height=[q_b_1[-1]], width=0.5, color="grey")
plt.bar(p_b[:-1], q_b_1[:-1], label="Bid Limit Orders", color="green", alpha=0.5, width=0.5)
plt.bar(p_a, q_a_1, label="Ask Limit Orders", color="red", alpha=0.5, width=0.5)
plt.plot([99.5, 99.5], [0, y_max], label="mid price", color="black", linestyle="--", alpha=0.5)
plt.xticks(p_b+p_a, ["97", "98", "99", "100", "101", "102"], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Volume", fontsize=16)
plt.xlabel("Price", fontsize=16)
plt.title("Limit order Book at $t_1$", fontsize=18)
plt.legend(fontsize=14)
plt.show()

plt.bar(p_b[:-1], q_b_1[:-1], label="Bid Limit Orders", color="green", alpha=0.5, width=0.5)
plt.bar(p_a, q_a_1, label="Ask Limit Orders", color="red", alpha=0.5, width=0.5)
plt.plot([99, 99], [0, y_max], label="mid price", color="black", linestyle="--", alpha=0.5)
plt.xticks(p_b+p_a, ["97", "98", "99", "100", "101", "102"], fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.ylabel("Volume", fontsize=16)
plt.xlabel("Price", fontsize=16)
plt.title("Limit order Book at $t_2$", fontsize=18)
plt.legend(fontsize=14)
plt.show()

