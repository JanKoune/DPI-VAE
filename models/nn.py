from torch import nn
from utils.transforms import IdentityTransform


class LinearModel(nn.Module):
    def __init__(
        self, n_latent, n_dim, input_transform=IdentityTransform(), output_transform=IdentityTransform()
    ):
        super().__init__()

        self.n_latent = n_latent
        self.n_dim = n_dim
        self.input_transform = input_transform
        self.output_transform = output_transform

        self.model = nn.Sequential()
        self.model.add_module(
            f"linear",
            nn.Linear(self.n_latent, self.n_dim),
        )

    def forward(self, z):
        zt, _ = self.input_transform.forward(z)
        x, _ = self.output_transform.forward(self.model(zt))
        return x


class MLP(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        layers,
        input_transform=None,
        output_transform=None,
        nonlinear_last=None,
        grad_reverse=None,
        nonlinearity=nn.ReLU,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.layers = layers.copy()
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.nonlinear_last = nonlinear_last
        self.grad_reverse = grad_reverse
        self.nonlinearity = nonlinearity

        self.layers.insert(0, self.n_input)
        self.layers.append(self.n_output)

        self.net = nn.Sequential()

        for i in range(len(self.layers) - 1):
            self.net.add_module(
                f"linear_{i}",
                nn.Linear(self.layers[i], self.layers[i + 1]),
            )
            self.net.add_module(f"nonlinear_{i}", self.nonlinearity())
        self.net.pop(-1)

        if (self.nonlinear_last is not None) and (self.nonlinear_last is not False):
            self.net.add_module(f"output", self.nonlinear_last())

    def forward(self, z):
        if self.grad_reverse is not None:
            z = self.grad_reverse(z)

        if self.input_transform is not None:
            zt, _ = self.input_transform.forward(z)
        else:
            zt = z

        if self.output_transform is not None:
            x, _ = self.output_transform.forward(self.net(zt))
        else:
            x = self.net(zt)
        return x
