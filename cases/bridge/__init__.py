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
# ==================================================================
# Distributions
#
# Notes:
#   * Lower and upper bounds are used to define ranges for transforms
# ==================================================================

dict_gt = {
        "kv1": {
            "lb": 9.5,
            "ub": 11.5,
            "dist": dist.Uniform,
            "args": {"low": 9.5, "high": 11.5},
            "type": "x",
            "label": r"$\log_{10} k_{v,1}$",
            "val": 11.5,
            "phys": False,
        },
        "kv2": {
            "lb": 9.5,
            "ub": 11.5,
            "dist": dist.Uniform,
            "args": {"low": 9.5, "high": 11.5},
            "type": "x",
            "label": r"$\log_{10} k_{v,2}$",
            "val": 11.5,
            "phys": False,
        },
        "y1": {
            "lb": 0.0,
            "ub": 1.0,
            "dist": dist.Uniform,
            "args": {"low": 0.0, "high": 1.0},
            "type": "y",
            "label": r"$y_1$ [-]",
            "val": 0.1,
            "phys": False,
        },
        "y2": {
            "lb": 0.0,
            "ub": 1.0,
            "dist": dist.Uniform,
            "args": {"low": 0.0, "high": 1.0},
            "type": "y",
            "label": r"$y_2$ [-]",
            "val": 0.1,
            "phys": False,
        },
        "v": {
            "lb": 0.9,
            "ub": 1.1,
            "dist": dist.Uniform,
            "args": {"low": 0.9, "high": 1.1},
            "type": "c",
            "label": r"$\delta_{\mathrm{v}}$ [-]",
            "val": 1.0,
            "phys": False,
        },
        "delta_xs": {
            "lb": -1.0,
            "ub": 1.0,
            "dist": dist.Uniform,
            "args": {"low": -1.0, "high": 1.0},
            "type": "c",
            "label": r"$\delta_\mathrm{s}$ [m]",
            "val": 0.0,
            "phys": True,
        },
        "f": {
            "lb": 0.95,
            "ub": 1.05,
            "dist": dist.Uniform,
            "args": {"low": 0.95, "high": 1.05},
            "type": "f",
            "label": r"$\delta_{\mathrm{F}}$ [-]",
            "val": 1.0,
            "phys": False,
        },
    }

dict_prior_x = {
        "kv1": {
            "lb": 9.001,
            "ub": 11.999,
            "dist": dist.Uniform,
            "args": {"low": 9.001, "high": 11.999},
        },
        "kv2": {
            "lb": 9.001,
            "ub": 11.999,
            "dist": dist.Uniform,
            "args": {"low": 9.001, "high": 11.999},
        },
    }

# ==================================================================
# Parameters
# ==================================================================
path = "./cases/bridge/"

# Input and output shapes
nd_x = 64
nz_x, nd_c, nd_y, nd_f, nd_p = get_shapes_from_dict(dict_gt)

# Domain
t_min = 1.0
t_max = 21.0
t = torch.linspace(t_min, t_max, nd_x)

# ==================================================================
# Models
# ==================================================================
# NN settings
layers_full = [64, 32, 64]
layers_part = [64, 32, 64]

# Load data, interpolate y
X_full = torch.load(os.path.join(path, "X.pt"), weights_only=True).to(device).type(torch.float32)
y_interp_full = torch.load(os.path.join(path, "y.pt"), weights_only=True).to(device).type(torch.float32)
X_part = torch.load(os.path.join(path, "X_partial.pt"), weights_only=True).to(device).type(torch.float32)
y_interp_part = torch.load(os.path.join(path, "y_partial.pt"), weights_only=True).to(device).type(torch.float32)
t_interp = torch.linspace(t_min, t_max, y_interp_full.shape[-1])

# Interpolate y
f_full = scipy.interpolate.interp1d(t_interp.cpu().numpy(), y_interp_full.cpu().numpy())
f_part = scipy.interpolate.interp1d(t_interp.cpu().numpy(), y_interp_part.cpu().numpy())
y_full = torch.tensor(f_full(t.cpu().numpy())).to(device)
y_part = torch.tensor(f_part(t.cpu().numpy())).to(device)

# Scaling
X_scaler_full = StandardScaler()
X_scaler_full = X_scaler_full.fit(X_full)

X_scaler_part = StandardScaler()
X_scaler_part = X_scaler_part.fit(X_part)

# Models
full_model = MLP(
    nd_c + nd_y + nd_f + nz_x,
    nd_x,
    layers_full,
    input_transform=X_scaler_full,
    nonlinear_last=None,
    nonlinearity=torch.nn.Tanh
).to(device)

part_model = MLP(
    nd_p + nz_x,
    nd_x,
    layers_part,
    input_transform=X_scaler_part,
    nonlinear_last=None,
    nonlinearity=torch.nn.Tanh
).to(device)

# Load a pre-trained surrogate model
full_model.load_state_dict(torch.load(os.path.join(path, "full_model"), weights_only=True))
part_model.load_state_dict(torch.load(os.path.join(path, "part_model"), weights_only=True))

# Set the models to eval mode and freeze the parameters
full_model.eval()
part_model.eval()

# Send models to device
full_model.to(device)
part_model.to(device)

for param in full_model.parameters():
    param.requires_grad = False

for param in part_model.parameters():
    param.requires_grad = False

# ==================================================================
# Model presets
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
        "lambda_g0": 1/1024,
        "lambda_x": None,
        "nz_c": 4,
        "nz_y": 4,
    },
    "DPIVAE-A": {
        "name": "DPIVAE-A",
        "model_type": "P",
        "lambda_g0": -1.0,
        "lambda_x": None,
        "nz_c": 4,
        "nz_y": 4,
    },
    "DPIVAE-B": {
        "name": "DPIVAE-B",
        "model_type": "S",
        "lambda_g0": 1 / 1024,
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
    "sigma_x": torch.tensor(0.0001).to(device),
    "sigma_c": torch.tensor(0.0001).to(device),
    "sigma_y": torch.tensor(0.0001).to(device),

    # Categorical variables
    "n_classes": None,
    "bins_y": None,
    "nk_y": None,
    "logsoftmax_y": False,


    # Data
    "x_full": X_full,
    "y_full": y_full,
    "x_scaler_full": X_scaler_full,
    "x_part": X_part,
    "y_part": y_part,
    "x_scaler_part": X_scaler_part,
    "x_unit": "Time [s]",
    "y_unit": r"[$^o/_{oo}$]",


    # Trained models
    "full_model": full_model,
    "part_model": part_model,


    # Plotting
    "ylim": (-1.0, 2.0),
}




