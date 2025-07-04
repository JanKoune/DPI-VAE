from torch import nn
from utils import device

class Decoder(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        layers,
        nonlinear_last=None,
        nonlinearity=nn.ReLU,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.layers = layers.copy()
        self.nonlinear_last = nonlinear_last
        self.nonlinearity = nonlinearity

        self.layers.insert(0, self.n_input)
        self.layers.append(2 * self.n_output)

        self.net = nn.Sequential()
        for i in range(len(self.layers) - 1):
            self.net.add_module(
                f"linear_{i}",
                nn.Linear(self.layers[i], self.layers[i + 1]),
            )
            self.net.add_module(f"nonlinear_{i}", self.nonlinearity())
        self.net.pop(-1)

        if (self.nonlinear_last is not None) and (self.nonlinear_last is not False):
            self.net.add_module(f"output", self.nonlinear_last)

    def forward(self, z):
        """
        Evaluate the deterministic forward model

        :param z:
        :param c:
        :return:
        """

        # Regression
        net_out = self.net(z)
        y = net_out[..., : self.n_output]
        log_sigma_y = net_out[..., self.n_output :]
        return y, log_sigma_y


class GradRevAdditive(nn.Module):
    """
    Decoder with additive assumption on the physics-based and data-driven predictions
    """

    def __init__(
        self,
        model,
        nz_p,
        nz_d,
        n_output,
        hidden=128,
        grad_reverse=None,
    ):
        super().__init__()

        self.model = model
        self.nz_p = nz_p
        self.nz_d = nz_d
        self.n_output = n_output
        self.grad_reverse = grad_reverse

        self.fx0 = nn.Linear(self.nz_d, hidden)
        self.fx1 = nn.Linear(hidden, n_output)

        self.nonlinearity = nn.ReLU()

    def forward(self, z, z_rev):
        # Inputs for `x` decoder
        if self.grad_reverse is not None:
            z_d = self.grad_reverse(z_rev)
        else:
            z_d = z_rev

        # Data driven x prediction
        xh_d = self.fx1(self.nonlinearity(self.fx0(z_d)))

        # Evaluate physics model
        xh_p = self.model(z)

        return xh_p, xh_d
