import numpy as np
import torch
from torch import distributions as dist
from matplotlib import pyplot as plt

class Annealing:
    def __init__(self, type, n_iter, **kwargs):
        self.type = type
        self.n_iter = n_iter
        self.kwargs = kwargs

    def forward(self, iter):
        if (self.type is None) or (self.type == "none") or (self.type == "None"):
            return torch.tensor(1.0)
        elif self.type == "cyclical":
            return self.cyclical_annealing(iter, self.n_iter, self.kwargs["n_cycles"], self.kwargs["R"])
        elif self.type == "sigmoid":
            return self.sigmoid_annealing(iter, self.n_iter, self.kwargs["mu"], self.kwargs["cov"])
        else:
            raise ValueError(f"Invalid type {self.type}")

    @staticmethod
    def cyclical_annealing(iter, n_iter, n_cycles, R):
        """
        Cyclical annealing schedule
        https://aclanthology.org/N19-1021.pdf
        :param iter: Current iteration
        :param n_iter: Total iterations
        :param n_cycles: Number of cycles
        :param R: Proportion of the cycle during which the weight is increased
        :return:
        """
        tau = np.mod(iter, n_iter / n_cycles) / (n_iter / n_cycles)
        if tau <= R:
            beta_t = 1.0 * (tau / R)
        else:
            beta_t = 1.0
        return torch.tensor(beta_t)


    @staticmethod
    def sigmoid_annealing(iter, n_iter, mu, cov):
        """
        Sigmoid annealing schedule
        :param mu: Midpoint of the curve in terms of the normalized step in [0, 1]
        :param cov: Coefficient of variation
        :param t_max:
        :return:
        """
        mu_t = mu * n_iter
        sigma_t = mu_t * cov
        return dist.Normal(mu_t, sigma_t).cdf(torch.tensor(iter))

if __name__ == "__main__":
    n_iter = 30_000
    mu = 0.1
    cov = 0.15
    n_cycles = 5
    R = 0.5

    t = torch.arange(0, n_iter)

    cyclical_annealer = Annealing("cyclical", n_iter, n_cycles=5, R=0.5)
    sigmoid_annealer = Annealing("sigmoid", n_iter, mu=mu, cov=cov)

    beta_cyclical_list = []
    beta_sigmoid_list = []
    for iter in range(n_iter):
        beta_cyclical = cyclical_annealer.forward(iter)
        beta_cyclical_list.append(beta_cyclical)

        beta_sigmoid = sigmoid_annealer.forward(iter)
        beta_sigmoid_list.append(beta_sigmoid)

    plt.figure()
    plt.plot(t, beta_cyclical_list, label="cyclical")
    plt.plot(t, beta_sigmoid_list, label="sigmoid")
    plt.legend()
    plt.grid()
    plt.show()