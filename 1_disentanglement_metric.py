import os
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dpivae import setup_model, train_model, disentanglement_metric
from utils import make_parser, cmap_vars
from utils.visualization import visualize_training_loss, save_close_fig
from utils.data import sample_response
from utils.priors import get_prior_dist
from cases import (
    damped_oscillator,
    simple_beam
)

# =======================================================================================
# SETUP
# =======================================================================================
# Problem definition
name = ""
case = damped_oscillator
preset = "dpivae"
regressor = "linear"

# Options
plot_interpolation = True
plot_aggregated = True
plot_regression = True
show_plots = False
save_plots = True
cond = False
use_mean = False

# Parse presets
parser = make_parser()
args, unknown = parser.parse_known_args()
presets = case.presets[preset]
vars_args = vars(args)
for key, item in presets.items():
    vars_args[key] = item

# Output path
path_output = os.path.join("output", name)

# Set definition to case definition
definition = case.definition
# =======================================================================================
# Runs
# =======================================================================================
# Impact of dataset size
scale_lambda = 1e4
n_runs = 6
var = "lambda_g0"
var_list = np.array([1e4, 1e3, 1e2, 1e1, 1e0, 0.0, -1e0, -1e1, -1e2, -1e3, -1e4]) / scale_lambda
# var_list = np.array([1e0, 0.0, -1e0]) / scale_lambda
n_train_regressor = 2048
n_test_regressor = 2048

dict_gt = definition["dict_gt"]
gen_factor_labels = [item["label"] for item in dict_gt.values()]
gen_factor_types = [item["type"] for item in dict_gt.values()]
gen_factors = list(dict_gt.keys())
df_columns = ["set", "gen_factor", "score", "idx_var", "iter", "lambda"]
df = pd.DataFrame(columns=df_columns)
n_runs_tot = len(var_list) * n_runs


list_fail = []
run_idx = 0
for i, var_i in enumerate(var_list):
    for j in range(n_runs):
        print("=====================================================================")
        print(f"Run {run_idx+1} / {n_runs_tot}, var={var_i}")
        print("=====================================================================")

        # ============================================================
        # VAE train and validation data
        # ============================================================
        # Distributions
        dist_gt = get_prior_dist(dict_gt)
        data_train_model = sample_response(definition, args.n_train, sample_dist=dist_gt)
        data_val_model = sample_response(definition, args.n_val, sample_dist=dist_gt)

        # ============================================================
        # Regressor train and test data
        # ============================================================
        data_train_regressor = sample_response(definition, n_train_regressor, sample_dist=dist_gt)
        data_test_regressor = sample_response(definition, n_test_regressor, sample_dist=dist_gt)

        # ==================================================================
        # Run
        # ==================================================================
        path_metrics = os.path.join(path_output, str(run_idx), "metrics")
        path_figures = os.path.join(path_output, str(run_idx), "figures")
        path_settings = os.path.join(path_output, str(run_idx), "settings")
        path_models = os.path.join(path_output, str(run_idx), "models")

        if not os.path.exists(path_metrics):
            os.makedirs(path_metrics)
        if not os.path.exists(path_figures):
            os.makedirs(path_figures)
        if not os.path.exists(path_settings):
            os.makedirs(path_settings)
        if not os.path.exists(path_models):
            os.makedirs(path_models)

        # Edit args
        vars_args[var] = float(var_i)

        # Store run arguments
        with open(os.path.join(path_settings, "args.json"), "w") as outfile:
            json.dump(vars(args), outfile)

        # Setup and train model
        vae = setup_model(args, definition, data_train_model)
        vae, logger = train_model(
            args,
            vae,
            definition,
            data_train_model,
            data_val_model,
            path_metrics=path_metrics,
            path_figures=path_figures,
        )

        # Loss curve
        fig_loss, ax_loss = visualize_training_loss(
            logger, n_skip_train=args.n_skip_plot_train, n_skip_val=args.n_skip_plot_val
        )
        save_close_fig(fig_loss, os.path.join(path_figures, "loss_curve.png"), show=show_plots)

        # Compute and store disentanglement score
        try:
            score_disentanglement = disentanglement_metric(
                args,
                vae,
                definition,
                data_train_regressor,
                data_test_regressor,
                regressor=regressor,
                cond=cond,
                use_mean=use_mean,
            )
            df_list = [item + [i, j, float(var_i)] for item in score_disentanglement]
            df_new = pd.DataFrame(df_list, columns=df_columns)
            df = pd.concat([df, df_new], axis=0, ignore_index=True)
        except:
            list_fail.append([i, j, run_idx, var_i])

        run_idx += 1

# Store disentanglement score results
df["lambda"] = df["lambda"] * scale_lambda
df.to_csv(os.path.join(path_output, "disentanglement_score.csv"), index=False)

# Separate dataframes
df_x = df[df["set"] == "zx"]
df_c = df[df["set"] == "zc"]
df_y = df[df["set"] == "zy"]

# Plotting
colors = ["tab:blue", "tab:green", "tab:orange"]
fig, ax = plt.subplots(len(gen_factors), 1, sharex="col")
list_var_gt = list(dict_gt.values())
for i, factor_i in enumerate(gen_factors):
    var_gt = dict_gt[factor_i]

    df_i = df[df["gen_factor"] == factor_i]
    df_x_i = df_x[df_x["gen_factor"] == factor_i]
    df_c_i = df_c[df_c["gen_factor"] == factor_i]
    df_y_i = df_y[df_y["gen_factor"] == factor_i]

    df_x_i_mean = df_x_i[["lambda", "score"]].groupby(["lambda"]).mean()
    df_c_i_mean = df_c_i[["lambda", "score"]].groupby(["lambda"]).mean()
    df_y_i_mean = df_y_i[["lambda", "score"]].groupby(["lambda"]).mean()
    df_x_i_std = df_x_i[["lambda", "score"]].groupby(["lambda"]).std()
    df_c_i_std = df_c_i[["lambda", "score"]].groupby(["lambda"]).std()
    df_y_i_std = df_y_i[["lambda", "score"]].groupby(["lambda"]).std()

    ax[i].fill_between(
        df_x_i_std.index.values,
        df_x_i_mean["score"] - 1 * df_x_i_std["score"],
        df_x_i_mean["score"] + 1 * df_x_i_std["score"],
        alpha=0.4,
        color=colors[0],
    )
    ax[i].fill_between(
        df_c_i_std.index.values,
        df_c_i_mean["score"] - 1 * df_c_i_std["score"],
        df_c_i_mean["score"] + 1 * df_c_i_std["score"],
        alpha=0.4,
        color=colors[1],
    )
    ax[i].fill_between(
        df_y_i_std.index.values,
        df_y_i_mean["score"] - 1 * df_y_i_std["score"],
        df_y_i_mean["score"] + 1 * df_y_i_std["score"],
        alpha=0.4,
        color=colors[2],
    )

    ax[i].plot(
        df_x_i_mean.index.values, df_x_i_mean["score"], alpha=1.0, label=r"$z_\mathrm{x}$", color=colors[0]
    )
    ax[i].plot(
        df_c_i_mean.index.values, df_c_i_mean["score"], alpha=1.0, label=r"$z_\mathrm{c}$", color=colors[1]
    )
    ax[i].plot(
        df_y_i_mean.index.values, df_y_i_mean["score"], alpha=1.0, label=r"$z_\mathrm{y}$", color=colors[2]
    )

    ax[i].scatter(df_x_i["lambda"], df_x_i["score"], alpha=0.9, s=4.0, color=colors[0])
    ax[i].scatter(df_c_i["lambda"], df_c_i["score"], alpha=0.9, s=4.0, color=colors[1])
    ax[i].scatter(df_y_i["lambda"], df_y_i["score"], alpha=0.9, s=4.0, color=colors[2])

    ax[i].set_xscale("symlog", linthresh=1)
    ax[i].set_ylabel(gen_factor_labels[i], color=cmap_vars[var_gt["type"]])

ax[-1].legend(bbox_transform=fig.transFigure, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=3)
ax[-1].set_xlabel(r"$\lambda \cdot 10^4$")

fig.tight_layout()
plt.show()
