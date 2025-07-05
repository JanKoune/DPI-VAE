import numpy as np
import torch
from scipy.integrate import solve_ivp

from utils import device


def mass_spring(z, t):
    """
    Simple harmonic oscillator defined in terms of mass, stiffness and initial conditions

    :param x_init: Tensor of shape (n_samples, 2) containing initial positions and velocities
    :param k: Tensor of shape (n_samples, 1) containing spring stiffnesses
    :param m: Tesnro fo shape(n_samples, 1) containing masses

    :return:
    """
    k = torch.tensor([1.0]).to(z.device.type)
    x0 = torch.tensor([1.0]).to(z.device.type)
    xd_init = torch.tensor([0.0]).to(z.device.type)

    m = z[..., 0].unsqueeze(-1)

    omega = torch.sqrt(k / m)

    B = xd_init / omega
    x = B * torch.sin(omega * t) + x0 * torch.cos(omega * t)
    return x


def mass_spring_dashpot(input, dt=0.01, Nt=100):
    """
    Function to generate synthetic data for the damped oscillator case study.
    Can include an external sinusoidal forcing term which is not used in the
    case study presented in the paper.

    Based on the implementation of N. Takeishi and A. Kalousis:
    https://github.com/n-takeishi/phys-vae/blob/main/data/pendulum/generate.py
    """

    # Constants
    k = 1.0
    omega_f = 4.0 * np.pi
    T0 = 20.0
    alpha_T = 0.01
    A = 0.0

    # Assemble inputs
    m = input[0]
    c = input[1]
    T = input[2]
    init_cond = [input[3], 0.0]

    # Temperature effect
    k_T = alpha_T * (T0 - T) + k

    omega_sq = k_T / m
    beta = c / m
    def fun(t, x):
        x1, x2 = x
        force = A * np.sin(omega_f * t)
        return [x2, -omega_sq * x1 - beta * x2 - force / m]

    # Solve IVP
    sol = solve_ivp(fun, (0.0, dt * (Nt - 1)), init_cond, dense_output=True, method="RK45")
    t = np.linspace(0.0, dt * (Nt - 1), Nt)
    return t, sol.sol(t).T


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use("TkAgg")
    from matplotlib import pyplot as plt

    # Domain
    Nt = 200
    dt = 0.05
    t = torch.linspace(0.0, dt * (Nt - 1), Nt).to(device)

    # Parameters
    x_init = torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    m = torch.tensor([1.0]).unsqueeze(0).repeat(5, 1)

    x = mass_spring(x_init, m, t)

    plt.figure()
    plt.plot(t, x.T.detach().numpy(), label="Position")
    plt.legend()
    plt.grid()
    plt.show()
