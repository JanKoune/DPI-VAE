"""
See `__init__.py` in the root of this folder for info on how to define a problem

Notes:
    * Paths are relative to the dpivae script, i.e. the project root folder
"""
import os
import torch
import torch.distributions as dist
import scipy

# Local imports
from utils import device
from models.nn import MLP
from utils.transforms import StandardScaler
from utils.priors import get_shapes_from_dict
from cases.simple_beam.simple_beam_model import euler_bernoulli_point_load

# ==================================================================
# Distributions
#
# Notes:
#   * Lower and upper bounds are used to define ranges for transforms
# ==================================================================
dict_gt = {
        "E": {
            "lb": 2.0,
            "ub": 6.0,
            "dist": dist.Uniform,
            "args": {"low": 2.5, "high": 4.5},
            "type": "x",
            "label": r"$E$ [MPa]",
            "val": 3.0,
            "phys": False,
        },
        "x_F": {
            "lb": 0.01,
            "ub": 0.99,
            "dist": dist.Uniform,
            "args": {"low": 0.3, "high": 0.7},
            "type": "x",
            "label": r"$x_F$ [m]",
            "val": 0.5,
            "phys": False,
        },
        "log_kv": {
            "lb": 5.0,
            "ub": 9.0,
            "dist": dist.Uniform,
            "args": {"low": 6.0, "high": 8.0},
            "type": "y",
            "label": r"$\log k_\mathrm{v}$ [N/m]",
            "val": 8.0,
            "phys": False,
        },
        "T": {
            "lb": -15.0,
            "ub": 15.0,
            "dist": dist.Uniform,
            "args": {"low": -11.0, "high": 5.0},
            "type": "c",
            "label": r"$T \ [\mathrm{C}^o]$",
            "val": 5.0,
            "phys": False,
        },
    }

dict_prior_x = {
        "E": {
            "lb": 2.0,
            "ub": 6.0,
            "dist": dist.Normal,
            "args": {"loc": 4.0, "scale": 1.0},
        },
        "x_F": {
            "lb": 0.01,
            "ub": 0.99,
            "dist": dist.Normal,
            "args": {"loc": 0.5, "scale": 0.2},
        },
    }
# ==================================================================
# Parameters
# ==================================================================
path = "./cases/simple_beam/"

# Input and output shapes
nd_x = 32
nz_x, nd_c, nd_y, nd_f, nd_p = get_shapes_from_dict(dict_gt)

# Domain
t_min = 0.00001
t_max = 1.0
t = torch.linspace(t_min, t_max, nd_x)

# Load data, interpolate y
X_full = torch.load(os.path.join(path, "X.pt"), weights_only=True).to(device).type(torch.float32)
y_interp = torch.load(os.path.join(path, "y.pt"), weights_only=True).to(device).type(torch.float32)
t_interp = torch.linspace(t_min, t_max, y_interp.shape[-1])


# Interpolate y
f_full = scipy.interpolate.interp1d(t_interp.cpu().numpy(), y_interp.cpu().numpy())
y_full = torch.tensor(f_full(t.cpu().numpy())).to(device)

# Scaling
X_scaler_full = StandardScaler()
X_scaler_full = X_scaler_full.fit(X_full)
# ==================================================================
# Models
# ==================================================================
# NN settings
layers_full = [256, 256]

# Models
full_model = MLP(
    nd_c + nd_y + nd_f + nz_x,
    nd_x,
    layers_full,
    input_transform=X_scaler_full,
    nonlinear_last=None,
    nonlinearity=torch.nn.Tanh
)

# Load a pre-trained surrogate model
full_model.load_state_dict(torch.load(os.path.join(path, "full_model"), weights_only=True))

# Set the models to eval mode and freeze the parameters
full_model.eval()

# Send models to device
full_model.to(device)

for param in full_model.parameters():
    param.requires_grad = False

# Partial model
part_model = lambda z: euler_bernoulli_point_load(z, npts=nd_x)

# ==================================================================
# Model presets
# ==================================================================
# Case presets
presets = {
    "vae": {
        "model_type": "P",
        "lambda_g0": -1.0,
        "lambda_x": None,
        "nz_c": 2,
        "nz_y": 2,
        # "alpha_x": 1.0,
    },
    "dpivae": {
        "model_type": "S",
        "lambda_g0": 1 / 256,
        "lambda_x": None,
        "nz_c": 2,
        "nz_y": 2,
    },
}

# ==================================================================
# Definition
# ==================================================================
definition = {

    # Latent variables and features
    "nd_x": nd_x,
    "nd_c": nd_c,
    "nd_y": nd_y,
    "nd_f": nd_f,
    "nd_p": nd_p,

    # Model settings
    "nz_x": nz_x,

    # Domain
    "t_min": t_min,
    "t_max": t_max,
    "t": t,

    # Physics-based latent variable prior
    "dict_prior_x": dict_prior_x,

    # Ground truth
    "dict_gt": dict_gt,

    # Noise
    "sigma_x": torch.tensor(0.02).to(device),
    "sigma_c": torch.tensor(0.02).to(device),
    "sigma_y": torch.tensor(0.02).to(device),

    # Categorical variables
    "n_classes": None,
    "bins_y": None,
    "nk_y": None,
    "logsoftmax_y": False,

    # Data
    "x_full": X_full,
    "x_scaler_full": X_scaler_full,
    "y_full": y_full,
    "x_part": None,
    "y_part": None,
    "x_unit": "Distance [m]",
    "y_unit": "[mm]",


    # Trained models
    "full_model": full_model,
    "part_model": part_model,

    # Plotting
    "ylim": (-25.0, 2.0),
}




