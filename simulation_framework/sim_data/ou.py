import numpy as np

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, theta, mu, sigma):
        self.__theta = theta
        self.__mu = mu
        self.__sigma = sigma

    def propagate(self, x0, delta_t):
        exp_factor = np.exp(-self.__theta * delta_t)
        return x0 * exp_factor + self.__mu * (1. - exp_factor) + \
               self.__sigma / np.sqrt(2. * self.__theta) * np.sqrt(1. - exp_factor * exp_factor) * np.random.normal()


def generate_ornstein_uhlenbeck_process(x0, ts, process):
    Xs = [x0]
    for i in range(1, len(ts)):
        delta_t = ts[i] - ts[i-1]
        Xs.append(process.propagate(Xs[-1], delta_t))
    return Xs

