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
from cases.damped_oscillator.mass_spring import mass_spring

# ==================================================================
# Distributions
#
# Notes:
#   * Lower and upper bounds are used to define ranges for transforms
# ==================================================================
dict_gt = {
        "m": {
            "lb": 1.2,
            "ub": 1.8,
            "dist": dist.Uniform,
            "args": {"low": 1.2, "high": 1.8},
            "type": "x",
            "label": r"$m$ [kg]",
            "val": 1.5,
            "phys": False,
        },
        "zeta": {
            "lb": 0.0,
            "ub": 2.0,
            "dist": dist.Uniform,
            "args": {"low": 0.0, "high": 2.0},
            "type": "y",
            "label": r"$c_\mathrm{d}$ [kg/s]",
            "val": 0.0,
            "phys": False,
        },
        "T": {
            "lb": 0.01,
            "ub": 39.99,
            "dist": dist.Uniform,
            "args": {"low": 0.01, "high": 39.99},
            "type": "c",
            "label": r"$T [\mathrm{C}^o]$",
            "val": 20.0,
            "phys": False,
        },
        "x_0": {
            "lb": 0.9,
            "ub": 1.1,
            "dist": dist.Uniform,
            "args": {"low": 0.9, "high": 1.1},
            "type": "f",
            "label": r"$x_0$ [m]",
            "val": 1.0,
            "phys": False,
        },
    }

dict_prior_x = {
        "m": {
            "lb": 1.0,
            "ub": 2.0,
            "dist": dist.Uniform,
            "args": {"low": 1.0, "high": 2.0},
        },
    }

# ==================================================================
# Parameters
# ==================================================================
path = "./cases/damped_oscillator/"

# Input and output shapes
nd_x = 64
nz_x, nd_c, nd_y, nd_f, nd_p = get_shapes_from_dict(dict_gt)

# Domain
Nt = 200
dt = 0.05
t_interp = torch.linspace(0.0, dt * (Nt - 1), Nt).to(device)
t_min, t_max = t_interp.min(), t_interp.max()
t = torch.linspace(t_min, t_max, nd_x).to(device)

# Load data, interpolate y
X_full = torch.load(os.path.join(path, "X.pt"), weights_only=True).to(device).type(torch.float32)
y_interp = torch.load(os.path.join(path, "y.pt"), weights_only=True).to(device).type(torch.float32)

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
part_model = lambda z: mass_spring(z, t)

# ==================================================================
# Presets
# ==================================================================
presets = {
    "vae": {
        "model_type": "P",
        "lambda_g0": -1.0,
        "lambda_x": None,
        "nz_c": 4,
        "nz_y": 4,
    },

    "dpivae": {
        "model_type": "S",
        "lambda_g0": 1/128,
        "lambda_x": None,
        "nz_c": 4,
        "nz_y": 4,
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
    "sigma_x": torch.tensor(0.01).to(device),
    "sigma_c": torch.tensor(0.01).to(device),
    "sigma_y": torch.tensor(0.01).to(device),

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
    "x_unit": "Time [s]",
    "y_unit": "[m]",


    # Trained models
    "full_model": full_model,
    "part_model": part_model,

    # Plotting
    "ylim": (-2.0, 2.0),
}




