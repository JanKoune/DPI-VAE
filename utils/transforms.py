"""
Implementation of various transforms and associated utilities.
"""

from abc import ABC, abstractmethod
import math
import torch
from torch import nn
from utils import device
from torch.autograd import Function


class Transform(ABC):
    """
    Abstract base class for the transforms. Derived classes must
    implement the `_forward` and `_inverse` methods.
    """

    def __init__(self, flip=False):
        self.flip = flip

    @abstractmethod
    def _forward(self, z):
        pass

    @abstractmethod
    def _inverse(self, z):
        pass

    def forward(self, z):
        if not self.flip:
            return self._forward(z)
        else:
            return self._inverse(z)

    def inverse(self, z):
        if not self.flip:
            return self._inverse(z)
        else:
            return self._forward(z)


class StandardScaler(Transform):
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.

    From: https://discuss.pytorch.org/t/advice-on-implementing-input-and-output-data-scaling/64369
    """

    def __init__(self, mean=None, scale=None, **kwargs):
        super().__init__(**kwargs)
        if mean is not None:
            mean = torch.FloatTensor(mean).to(device)
        if scale is not None:
            scale = torch.FloatTensor(scale).to(device)
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data."""
        self.mean_ = sample.mean(0, keepdim=True)
        self.scale_ = sample.std(0, unbiased=False, keepdim=True)
        return self

    def _forward(self, z):
        z = (z - self.mean_) / self.scale_
        log_det = -torch.log(self.scale_).sum() * torch.ones(z.shape[:-1]).to(device)
        return z, log_det

    def _inverse(self, z):
        """Scale the data back to the original representation"""
        z = z * self.scale_ + self.mean_
        log_det = torch.log(self.scale_).sum() * torch.ones(z.shape[:-1]).to(device)
        return z, log_det


class ShiftScale(Transform):
    """Shift and scale a coordinate to a [0, 1] box

    Args:
        lb (FloatTensor): 1D tensor of lower bounds
        ub (FloatTensor): 1D tensor of upper bounds
    """

    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub
        self.a = self.ub - self.lb
        self.b = self.lb

    def _forward(self, z):
        z = z * self.a + self.b
        log_det = torch.sum(torch.log(torch.abs(self.a)) * torch.ones(z.shape).to(device), dim=-1)
        return z, log_det

    def _inverse(self, z):
        z = z / self.a - self.b / self.a
        log_det = -torch.log(self.a).sum() * torch.ones(z.shape[:-1]).to(device)
        return z, log_det


class Logistic(Transform):
    """
    Based on the `Logit` transform of the `normflows` package.

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```

    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.k = k

    def _forward(self, z):
        log_det = self.jld_forward(z, self.k).sum(dim=-1)
        return self.sigmoid(self.k * z), log_det

    def _inverse(self, z):
        raise NotImplementedError("Inverse not implemented for this transform")

    @staticmethod
    def jld_forward(x, k):
        return k * x - 2 * nn.functional.softplus(k * x) + math.log(k)


class ChainTransform(Transform):
    """
    Class used to chain together a series of transforms
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.lst_transforms = list(args)

    def _forward(self, z):
        log_det = torch.zeros(*z.shape[:-1]).to(device)
        for transf in self.lst_transforms:
            z, log_det_i = transf.forward(z)
            log_det += log_det_i
        return z, log_det

    def _inverse(self, z):
        log_det = torch.zeros(*z.shape[:-1]).to(device)
        for transf in self.lst_transforms.__reversed__():
            z, log_det_i = transf.inverse(z)
            log_det += log_det_i
        return z, log_det


class ChainTransformMasked(Transform):
    """
    Class used to chain together a series of transforms
    """

    def __init__(self, mask, *args, **kwargs):
        super().__init__(**kwargs)
        self.lst_transforms = list(args)
        self.mask = mask

    def _forward(self, z):
        z_masked = z[..., self.mask]
        log_det = torch.zeros(*z_masked.shape[:-1]).to(device)
        for transf in self.lst_transforms:
            z_masked, log_det_i = transf.forward(z_masked)
            log_det += log_det_i
        z[..., self.mask] = z_masked
        return z, log_det

    def _inverse(self, z):
        z_masked = z[..., self.mask]
        log_det = torch.zeros(*z_masked.shape[:-1]).to(device)
        for transf in self.lst_transforms.__reversed__():
            z_masked, log_det_i = transf.inverse(z_masked)
            log_det += log_det_i
        z[..., self.mask] = z_masked
        return z, log_det


class IdentityTransform(Transform):
    """Dummy transform to be used as default transform in surrogate models"""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, z):
        return z, torch.zeros(z.shape[:-1])

    def _inverse(self, z):
        return z, torch.zeros(z.shape[:-1])


class FuncGradRev(Function):
    """
    Taken from https://github.com/janfreyberg/pytorch-revgrad
    """

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class LayerGradRev(nn.Module):
    def __init__(self, revgrad, alpha=1.0, *args, **kwargs):
        """
        Taken from https://github.com/janfreyberg/pytorch-revgrad

        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self.revgrad = revgrad
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return self.revgrad(input_, self._alpha)
