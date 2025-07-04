import torch
from torch import nn
from torch import distributions as dist
from utils import device, gaussian_const

class FullCovarianceNN(nn.Module):
    def __init__(self, n_latent, n_input, layers):
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input
        self.layers = layers.copy()

        # Initialize nn
        self.layers.insert(0, self.n_input)

        self.mean_output = self.n_latent
        self.sigma_output = self.n_latent
        self.cov_output = self.n_latent * self.n_latent

        self.f_mean = nn.Linear(self.layers[-1], self.mean_output)
        self.f_sigma = nn.Linear(self.layers[-1], self.sigma_output)
        self.f_cov = nn.Linear(self.layers[-1], self.cov_output)

        # Mean network
        self.net = nn.Sequential()
        for i in range(len(self.layers) - 1):
            self.net.add_module(
                f"encoder_linear_{i}",
                nn.Linear(self.layers[i], self.layers[i + 1]),
            )
            self.net.add_module(f"encoder_nonlinear_{i}", nn.ReLU())

    def forward(self, x, jitter=1e-8):
        x_net = self.net(x)
        loc = self.f_mean(x_net).clamp(-50.0, 50.0)
        sigma = torch.exp(self.f_sigma(x_net).clamp(-7.0, 3.0))
        L = torch.tril(
            self.f_cov(x_net).clamp(-20.0, 20.0).reshape(-1, self.n_latent, self.n_latent), diagonal=-1
        )

        # return mu, sigma, L

        scale_tril = L + torch.diag_embed(sigma + jitter)
        return loc, scale_tril

class GaussianEncoder(nn.Module):
    """
    Multivariate Gaussian encoder

    TODO:
        * Replace the KL divergence Monte Carlo gradient estimate with analytical calculation
    """

    def __init__(self, net, input_transform=None, output_transform=None):
        super().__init__()
        self.net = net
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x, jitter=1e-8):
        """
        Returns the `(mu, L)` of `log q_phi(z|x^i)`
        """
        if self.input_transform is not None:
            x_t, _ = self.input_transform.forward(x)
        else:
            x_t = x

        # Encode
        loc, scale_tril = self.net(x_t)
        return loc, scale_tril

    def sample(self, loc, scale_tril, n=1):
        """
        Returns samples and log density `log q_phi(z|x^i)`
        """
        # # Log-density of full covariance Gaussian
        # eps = torch.randn((n, *list(loc.shape)), device=device)
        # z = loc + torch.matmul(scale_tril, eps.unsqueeze(-1)).squeeze(-1)
        # log_q = torch.sum(-0.5 * eps ** 2 + gaussian_const, dim=-1)
        # log_q -= torch.sum(torch.diagonal(scale_tril, dim1=-2, dim2=-1).log(), dim=-1)

        # Check: Using torch.dist
        mvn = dist.MultivariateNormal(loc, scale_tril=scale_tril)
        z = mvn.rsample((n,))
        log_q = mvn.log_prob(z)

        # Output transform
        log_det_z = torch.zeros(log_q.shape, device=log_q.device)
        if self.output_transform is not None:
            z, log_det_z = self.output_transform.forward(z)

        return z, log_q - log_det_z


class FactorizedNN(nn.Module):
    def __init__(self, n_latent, n_input, layers):
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input
        self.layers = layers.copy()

        # Initialize nn
        self.layers.insert(0, self.n_input)

        self.mean_output = self.n_latent
        self.sigma_output = self.n_latent

        self.f_mean = nn.Linear(self.layers[-1], self.mean_output)
        self.f_sigma = nn.Linear(self.layers[-1], self.sigma_output)

        # Mean network
        self.net = nn.Sequential()
        for i in range(len(self.layers) - 1):
            self.net.add_module(
                f"encoder_linear_{i}",
                nn.Linear(self.layers[i], self.layers[i + 1]),
            )
            self.net.add_module(f"encoder_nonlinear_{i}", nn.ReLU())

    def forward(self, x, jitter=1e-8):
        x_net = self.net(x)
        loc = self.f_mean(x_net).clamp(-50.0, 50.0)
        sigma = torch.exp(self.f_sigma(x_net).clamp(-7.0, 3.0))
        # return mu, sigma

        scale_tril = torch.diag_embed(sigma + jitter)
        return loc, scale_tril