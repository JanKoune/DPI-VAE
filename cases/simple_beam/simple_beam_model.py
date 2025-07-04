import torch
from utils import device

def euler_bernoulli_point_load(z, I=2e-6, L=1.0, P=1.0, npts=200):
    """
    :param z: Tensor of shape `(n_batch, n_mc, 2)` with Young's moduli in the first column
        and load position in the second column
    :param I:
    :param L:
    :param P:
    :param npts:
    :return:
    """
    x = torch.linspace(0.0, L, npts).to(z.device.type)
    E = z[..., 0].unsqueeze(-1) * 1e+6
    a = z[..., 1].unsqueeze(-1)
    b = L - a

    # Check inputs
    if (torch.any(a < 0.0)) or (torch.any(a > L)):
        raise ValueError("Load position must be between 0 and L")

    # Mask
    mask = x > a

    # Compute deflection
    w = P * b * x * (L**2 - b**2 - x**2) / (6 * E * I * L)
    wb = P * ((x - a) ** 3) / (6 * E * I)
    w[mask] += wb[mask]

    return -1000.0 * w