import torch
from torch import nn
from torch import distributions as dist
from utils import device
import pytorch_lightning as pl


class DPIVAE(pl.LightningModule):
    def __init__(
        self,
        prior_x,
        prior_net_c,
        prior_net_y,
        encoder,
        decoder_x,
        decoder_c,
        decoder_y,
        nz_x,
        nz_c,
        nz_y,
        nd_x,
        nd_c,
        nd_y,
        idx_c_phys,
        model_type=None,
        encoder_c=None,
        encoder_y=None,
        lambda_x=None,
        transform_x=None,
        transform_c=None,
        transform_y=None,
        jitter=1e-6,
    ):
        super().__init__()
        self.prior_x = prior_x
        self.prior_net_c = prior_net_c
        self.prior_net_y = prior_net_y
        self.encoder = encoder
        self.model_type = model_type
        self.encoder_c = encoder_c
        self.encoder_y = encoder_y
        self.decoder_x = decoder_x
        self.decoder_c = decoder_c
        self.decoder_y = decoder_y
        self.nz_x = nz_x
        self.nz_c = nz_c
        self.nz_y = nz_y
        self.nd_x = nd_x
        self.nd_c = nd_c
        self.nd_y = nd_y
        self.idx_c_phys = idx_c_phys
        self.lambda_x = lambda_x
        self.transform_x = transform_x
        self.transform_c = transform_c
        self.transform_y = transform_y
        self.jitter = jitter

        if (self.model_type != "P") and (self.model_type != "S"):
            raise ValueError(f"Invalid model_type {self.model_type}")

        if self.model_type == "S":
            if (self.encoder_c is not None) or (self.encoder_y is not None):
                raise ValueError("encoder_c and encoder_y must NOT be defined for model type S")

        if self.model_type == "P":
            if (self.encoder_c is None) or (self.encoder_y is None):
                raise ValueError("encoder_c and encoder_y must be defined for model type P")

        # Noise parameter
        self.log_sigma_x = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def transform_inputs(self, x=None, c=None, y=None):
        if x is not None:
            if self.transform_x is not None:
                x_t, _ = self.transform_x.forward(x)
            else:
                x_t = x
        else:
            x_t = torch.nan

        if c is not None:
            if self.transform_c is not None:
                c_t, _ = self.transform_c.forward(c)
            else:
                c_t = c
        else:
            c_t = torch.nan

        if y is not None:
            if self.transform_y is not None:
                y_t, _ = self.transform_y.forward(y)
            else:
                y_t = y
        else:
            y_t = torch.nan

        return x_t, c_t, y_t

    def prior_net(self, c, y=None):
        # Input transforms
        _, c_t, y_t = self.transform_inputs(x=None, c=c, y=y)

        # Priors
        prior_loc_c, prior_scale_tril_c = self.prior_net_c(c_t)
        if y is not None:
            prior_loc_y, prior_scale_tril_y = self.prior_net_y(y_t)
        else:
            prior_loc_y, prior_scale_tril_y = None, None

        return prior_loc_c, prior_scale_tril_c, prior_loc_y, prior_scale_tril_y

    def sample_prior(self, c, y, n=1):
        # Input transforms
        _, c_t, y_t = self.transform_inputs(x=None, c=c, y=y)

        # Priors
        loc_prior_c, scale_tril_prior_c = self.prior_net_c(c_t)
        zc_prior, dens_zc_prior = self.prior_net_c.sample(loc_prior_c, scale_tril_prior_c, n=n)

        loc_prior_y, scale_tril_prior_y = self.prior_net_y(y_t)
        zy_prior, dens_zy_prior = self.prior_net_y.sample(loc_prior_y, scale_tril_prior_y, n=n)

        return zc_prior, dens_zc_prior, zy_prior, dens_zy_prior

    def encode(self, x, n=1):
        """
        Notes:
            * `loc` and `scale_tril` are defined in the internal workspace. `z` and `dens_z` are
            defined in the latent space.
        """

        # Single latent space
        if self.model_type == "S":
            loc, scale_tril = self.encoder(x)
            z, dens_z = self.encoder.sample(loc, scale_tril, n=n)
            zx = z[..., : self.nz_x]
            zc = z[..., self.nz_x : self.nz_x + self.nz_c]
            zy = z[..., self.nz_x + self.nz_c : self.nz_x + self.nz_c + self.nz_y]

        # Partitioned latent space
        elif self.model_type == "P":
            loc_x, scale_tril_x = self.encoder(x)
            loc_c, scale_tril_c = self.encoder_c(x)
            loc_y, scale_tril_y = self.encoder_y(x)

            zx, dens_zx = self.encoder.sample(loc_x, scale_tril_x, n=n)
            zc, dens_zc = self.encoder_c.sample(loc_c, scale_tril_c, n=n)
            zy, dens_zy = self.encoder_y.sample(loc_y, scale_tril_y, n=n)
            dens_z = dens_zx + dens_zc + dens_zy

        return zx, zc, zy, dens_z

    def decode(self, zx, zc, zy):
        """
        Notes:
            * For classification problems the `decoder_y` is expected to return an array
                of shape `(N_mc, N_batch, N_o, N_c)` containing logits, where `N_o` is the number of outputs
                and `N_c` is the number of classes (assumed equal across outputs for now)
        """
        xh_p, xh_d = self.decoder_x(zx, torch.cat((zc, zy), dim=-1))
        yh, log_sigma_y = self.decoder_y(zy)
        ch, log_sigma_c = self.decoder_c(zc)

        return xh_p, xh_d, ch, log_sigma_c, yh, log_sigma_y

    def forward(self, x, c, cond=False, n=1):
        x_t, c_t, y_t = self.transform_inputs(x=x, c=c)
        zx, zc, zy, dens_z = self.encode(x_t, n=n)

        # Use prior net if `c` is provided as an input
        if cond == True:
            loc_c, scale_tril_c = self.prior_net_c(c_t)
            zc, _ = self.prior_net_c.sample(loc_c, scale_tril_c, n=n)

        # Concatenate `c_p` to `z_x`. If there is no `c_phys`, then `idx_c_phys` == []
        # and nothing is concatenated to `zx`
        c_phys = c[..., self.idx_c_phys].unsqueeze(0).repeat(n, 1, 1)
        zx_in = torch.cat((zx, c_phys), dim=-1)
        xh_p, xh_d, ch, log_sigma_c, yh, log_sigma_y = self.decode(zx_in, zc, zy)

        return xh_p, xh_d, ch, log_sigma_c, yh, log_sigma_y, zx, zc, zy, dens_z

    def loss(
        self,
        x,
        c,
        y,
        n=1,
        beta_x=1.0,
        beta_c=1.0,
        beta_y=1.0,
        alpha_x=1.0,
        alpha_c=1.0,
        alpha_y=1.0,
    ):
        """
        Returns the per-datapoint loss
        """
        # Forward through the VAE
        xh_p, xh_d, ch, log_sigma_c, yh, log_sigma_y, zx, zc, zy, dens_z = self.forward(
            x, c=c, cond=False, n=n
        )
        xh = xh_p + xh_d

        # Get prior distributions
        prior_loc_c, prior_scale_tril_c, prior_loc_y, prior_scale_tril_y = self.prior_net(c, y=y)
        log_prior_zx = self.prior_x.log_prob(zx).sum(dim=-1)
        log_prior_zc = dist.MultivariateNormal(prior_loc_c, scale_tril=prior_scale_tril_c).log_prob(zc)
        log_prior_zy = dist.MultivariateNormal(prior_loc_y, scale_tril=prior_scale_tril_y).log_prob(zy)
        log_prior_z = log_prior_zx + log_prior_zc + log_prior_zy

        # KL-div
        KL_x = torch.mean(dens_z - log_prior_z, dim=0)
        KL_c = torch.tensor(0.0)
        KL_y = torch.tensor(0.0)

        # Reconstruction errors
        R_x = dist.Normal(xh, self.log_sigma_x.exp()).log_prob(x).sum(dim=-1).mean(dim=0)
        R_c = dist.Normal(ch, log_sigma_c.exp()).log_prob(c).sum(dim=-1).mean(dim=0)
        R_y = dist.Normal(yh, log_sigma_y.exp()).log_prob(y).sum(dim=-1).mean(dim=0)

        # Regularization
        reg = torch.zeros(x.shape[0], device=device)
        if self.lambda_x is not None:
            reg += dist.Normal(torch.tensor(0.0).to(device), self.lambda_x).log_prob(xh_d).sum(dim=-1).mean(dim=0)

        # Full loss and components
        return (
            beta_x * KL_x - alpha_x * R_x - alpha_c * R_c - alpha_y * R_y - reg,
            KL_x,
            KL_c,
            KL_y,
            R_x,
            R_c,
            R_y,
            reg,
        )

    def sample(self, x, c, cond=False, n=1):
        """
        Sample the VAE prediction

        Notes:
            * Contrary to `loss`, here `c` is used in the forward call if provided. This means that if `c` is known
            the corresponding `zc` are generated by the prior net `p(z_c | c)`.

        :param mu:
        :param sigma:
        :param n:
        :return:
        """

        # Decoder prediction
        xh_p, xh_d, ch, log_sigma_c, yh, log_sigma_y, zx, zc, zy, dens_z = self.forward(x, c, cond=cond, n=n)

        # Noise
        x_sample = dist.Normal(xh_p + xh_d, self.log_sigma_x.exp()).sample()
        c_sample = dist.Normal(ch, log_sigma_c.exp()).sample()
        y_sample = dist.Normal(yh, log_sigma_y.exp()).sample()

        return x_sample, xh_p, xh_d, c_sample, y_sample, zx, zc, zy, dens_z
