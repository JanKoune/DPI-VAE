import torch
import argparse

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
neg_inf = torch.tensor([-torch.inf]).to(device)
pos_inf = torch.tensor([torch.inf]).to(device)
gaussian_const = -0.5 * torch.log(torch.tensor(2.0) * torch.pi).to(device)

# Plotting
cmap_name = "plasma"
alpha_interp = 0.01
cmap_vars = {"x": "tab:blue", "c": "tab:green", "y": "tab:orange", "f": "tab:red", "p": "tab:cyan"}

# The argparser is not really appropriate for this job, considering that we don't use
# the CLI. Dataclasses would be a better option. Consider switching in the future.
def make_parser():
    parser = argparse.ArgumentParser("")

    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--use_seed", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)

    # Models
    parser.add_argument("--encoder_x", type=str, default="NN")
    parser.add_argument("--encoder_c", type=str, default="NN")
    parser.add_argument("--encoder_y", type=str, default="NN")
    parser.add_argument("--full_cov_prior", action="store_true", default=False)

    # Data, training and validation
    parser.add_argument("--n_iter", type=int, default=20_000)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_val", type=int, default=512)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--n_ood", type=int, default=512)
    parser.add_argument("--n_batch", type=int, default=64)
    parser.add_argument("--n_mc_train", type=int, default=16)
    parser.add_argument("--n_mc_val", type=int, default=64)
    parser.add_argument("--n_mc_test", type=int, default=512)
    parser.add_argument("--val_freq", type=int, default=10)

    # Disentanglement
    parser.add_argument("--lambda_g0", type=float, default=1/256)
    parser.add_argument("--beta_x0", type=float, default=1.0)
    parser.add_argument("--beta_c0", type=float, default=1.0)
    parser.add_argument("--beta_y0", type=float, default=1.0)
    parser.add_argument("--lambda_x", type=float, default=None)
    parser.add_argument("--alpha_x", type=float, default=1.0)
    parser.add_argument("--alpha_c", type=float, default=1.0)
    parser.add_argument("--alpha_y", type=float, default=1.0)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_e", type=float, default=1e-3)
    parser.add_argument("--lr_ex", type=float, default=1e-3)
    parser.add_argument("--lr_ec", type=float, default=1e-3)
    parser.add_argument("--lr_ey", type=float, default=1e-3)
    parser.add_argument("--lr_p", type=float, default=1e-3)
    parser.add_argument("--lr_dx", type=float, default=1e-3)
    parser.add_argument("--lr_dc", type=float, default=1e-3)
    parser.add_argument("--lr_dy", type=float, default=1e-3)
    parser.add_argument("--lr_sigma", type=float, default=5e-3)
    parser.add_argument("--wd_e", type=float, default=0.0)
    parser.add_argument("--wd_p", type=float, default=0.0)
    parser.add_argument("--wd_dx", type=float, default=0.0)
    parser.add_argument("--wd_dc", type=float, default=0.0)
    parser.add_argument("--wd_dy", type=float, default=0.0)
    parser.add_argument("--wd_sigma", type=float, default=0.0)
    parser.add_argument("--clip_gradients", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=200) # Note: epochs to wait = patience * val_freq
    parser.add_argument("--min_delta", type=float, default=0.001)

    # Annealing
    parser.add_argument(
        "--lambda_annealing", type=str, default=None, choices=["cyclical", "sigmoid", "None", None]
    )
    parser.add_argument("--lambda_n_cycles", type=int, default=5)
    parser.add_argument("--lambda_R", type=float, default=0.5)
    parser.add_argument("--lambda_mu", type=float, default=0.15)
    parser.add_argument("--lambda_cov", type=float, default=0.15)
    parser.add_argument(
        "--beta_x_annealing", type=str, default=None, choices=["cyclical", "sigmoid", "None", None]
    )
    parser.add_argument("--beta_x_n_cycles", type=int, default=5)
    parser.add_argument("--beta_x_R", type=float, default=0.5)
    parser.add_argument("--beta_x_mu", type=float, default=0.15)
    parser.add_argument("--beta_x_cov", type=float, default=0.15)
    parser.add_argument(
        "--beta_c_annealing", type=str, default=None, choices=["cyclical", "sigmoid", "None", None]
    )
    parser.add_argument("--beta_c_n_cycles", type=int, default=5)
    parser.add_argument("--beta_c_R", type=float, default=0.5)
    parser.add_argument("--beta_c_mu", type=float, default=0.15)
    parser.add_argument("--beta_c_cov", type=float, default=0.15)
    parser.add_argument(
        "--beta_y_annealing", type=str, default=None, choices=["cyclical", "sigmoid", "None", None]
    )
    parser.add_argument("--beta_y_n_cycles", type=int, default=4)
    parser.add_argument("--beta_y_R", type=float, default=0.5)
    parser.add_argument("--beta_y_mu", type=float, default=0.2)
    parser.add_argument("--beta_y_cov", type=float, default=0.2)

    # Plotting
    parser.add_argument("--n_skip_plot_train", type=int, default=0)
    parser.add_argument("--n_skip_plot_val", type=int, default=0)
    parser.add_argument("--n_plot", type=int, default=2000)
    parser.add_argument("--n_interp", type=int, default=5)

    # Nets
    parser.add_argument("--ch_in", type=int, default=1)
    parser.add_argument("--ch_out", type=int, default=16)
    parser.add_argument("--ch_latent", type=int, default=64)

    return parser