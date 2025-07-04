import os
import json
import numpy as np
import pandas as pd

from dpivae import setup_model, train_model, run_comparison, evaluate_model
from utils import make_parser
from utils.data import sample_response
from utils.priors import make_square_dist

from matplotlib import pyplot as plt
from utils.visualization import plot_regression_error, save_close_fig
from cases import bridge, simple_beam

# ============================================================
# SETUP
# ============================================================
# Problem definition
name = "transfer"
case = bridge
dist_type = "extrapolation"  # Can be `interpolation` or `extrapolation`
n_domains = 4
n_runs = 6

# Options
plot_interpolation = True
plot_aggregated = True
plot_regression = False
plot_domain = True
show_plots = True
save_plots = True
cond = False

# Parse definition but don't load presets
parser = make_parser()
args, unknown = parser.parse_known_args()

# Paths
path_output = os.path.join("output", name)
path_metrics = os.path.join(path_output, "metrics")
path_figures = os.path.join(path_output, "figures")
path_settings = os.path.join(path_output, "settings")
path_models = os.path.join(path_output, "models")

if not os.path.exists(path_metrics):
    os.makedirs(path_metrics)
if not os.path.exists(path_figures):
    os.makedirs(path_figures)
if not os.path.exists(path_settings):
    os.makedirs(path_settings)
if not os.path.exists(path_models):
    os.makedirs(path_models)

# Store run arguments
with open(os.path.join(path_settings, "args.json"), "w") as outfile:
    json.dump(vars(args), outfile)

# Set definition to case definition
definition = case.definition
# ============================================================
# DATA
# ============================================================
# Get list of distributions
if dist_type == "interpolation":
    dists_train, dists_test = make_square_dist(definition)
elif dist_type == "extrapolation":
    dists_test, dists_train = make_square_dist(definition)
else:
    raise ValueError(f"`dist_type` can be either `interpolation` or `extrapolation`, not {dist_type}")

# Get labels
labels_x = [item["label"] for item in definition["dict_gt"].values() if item["type"] == "x"]
labels_y = [item["label"] for item in definition["dict_gt"].values() if item["type"] == "y"]

if plot_domain == True:
    fig, ax = plt.subplots(1, n_domains, figsize=(12, 3), layout="compressed")
    for i in range(n_domains):
        dist_train_i = dists_train[i]
        dist_test_i = dists_test[i]

        _, _, _, z_train = sample_response(definition, args.n_train, sample_dist=dist_train_i)
        _, _, _, z_test = sample_response(definition, args.n_test, sample_dist=dist_test_i)

        z_train_plot = z_train.detach().cpu().numpy()
        z_test_plot = z_test.detach().cpu().numpy()

        xy_max_i = np.max(np.vstack((z_train_plot, z_test_plot)), axis=0)[[0, 1]]
        xy_min_i = np.min(np.vstack((z_train_plot, z_test_plot)), axis=0)[[0, 1]]
        xlim_i = (xy_min_i[0], xy_max_i[0])
        ylim_i = (xy_min_i[1], xy_max_i[1])

        ax[i].scatter(z_train_plot[:, 0], z_train_plot[:, 1], s=4.0)
        ax[i].scatter(z_test_plot[:, 0], z_test_plot[:, 1], s=4.0)

        ax[i].set_xlabel(labels_x[0], fontsize=14)
        ax[i].set_title(f"Sub-case {i + 1}")
        ax[i].set_xlim(xlim_i)
        ax[i].set_ylim(ylim_i)

        # Annotation
        Xc = np.sum(xlim_i) / 2
        Yc = np.sum(ylim_i) / 2
        ax[i].axvline(x=Xc, ymin=0.0, ymax=1.0, color="black")
        ax[i].axhline(y=Yc, xmin=0.0, xmax=1.0, color="black")

    ax[0].set_ylabel(labels_x[1], fontsize=14)
    plt.show()

# ============================================================
# OPTIMIZATION
# ============================================================
# Assemble train and test distributions: three quarters are used for training
# and one for testing. The quarters are used to do 4-fold cross validation

dict_run_metrics = {}
dict_run_pred = {}
for j in range(n_runs):

    dict_domain_metrics = {}
    dict_domain_pred = {}
    for i in range(n_domains):
        print("=====================================================================")
        print(f"Domain {i+1} / {n_domains}, run {j+1} / {n_runs}")
        print("=====================================================================")

        dist_train_i = dists_train[i]
        dist_val_i = dists_train[i]  # Validation dist same as training
        dist_test_i = dists_test[i]

        x_train, c_train, y_train, z_train = sample_response(
            definition, args.n_train, sample_dist=dist_train_i
        )
        x_val, c_val, y_val, z_val = sample_response(definition, args.n_val, sample_dist=dist_val_i)
        x_test, c_test, y_test, z_test = sample_response(definition, args.n_test, sample_dist=dist_test_i)

        data_train = (x_train, c_train, y_train)
        data_val = (x_val, c_val, y_val)
        data_test = (x_test, c_test, y_test)

        # ==================================================================
        # Run DVAE-A
        # ==================================================================
        # Load presets
        preset = "DPIVAE-A"
        presets = case.presets[preset]
        vars_args = vars(args)
        for key, item in presets.items():
            vars_args[key] = item

        dvae_a = setup_model(args, definition, data_train)
        dvae_a, logger_dvae_a = train_model(
            args,
            dvae_a,
            definition,
            data_train,
            data_val,
            path_metrics=path_metrics,
            path_figures=path_figures,
        )

        # Evaluate VAE
        dvae_a_metrics_test, dvae_a_pred_test = evaluate_model(args, definition, dvae_a, data_test, cond=cond)

        # ==================================================================
        # Run DVAE-B
        # ==================================================================
        # Load presets
        preset = "DPIVAE-B"
        presets = case.presets[preset]
        vars_args = vars(args)
        for key, item in presets.items():
            vars_args[key] = item

        dvae_b = setup_model(args, definition, data_train)
        dvae_b, logger_dvae_b = train_model(
            args,
            dvae_b,
            definition,
            data_train,
            data_val,
            path_metrics=path_metrics,
            path_figures=path_figures,
        )

        # Evaluate VAE
        dvae_b_metrics_test, dvae_b_pred_test = evaluate_model(args, definition, dvae_b, data_test, cond=cond)

        # ==================================================================
        # Run comparison
        # ==================================================================
        regression_metrics_test, regression_pred_test = run_comparison(
            args, definition, data_train, data_test
        )

        # Metrics dict
        dict_metrics_i = {}
        dict_metrics_i.update(regression_metrics_test)
        dict_metrics_i.update(dvae_a_metrics_test)
        dict_metrics_i.update(dvae_b_metrics_test)

        # Predictions dict
        dict_pred_i = {}
        dict_pred_i.update(regression_pred_test)
        dict_pred_i.update(dvae_a_pred_test)
        dict_pred_i.update(dvae_b_pred_test)

        # Group by domain
        dict_domain_metrics[i + 1] = dict_metrics_i
        dict_domain_pred[i + 1] = dict_pred_i

        # Plot
        if plot_regression == True:
            for name, item in dict_pred_i.items():
                fig_regression_error_test, ax_regression_error_test = plot_regression_error(
                    y_test, item, definition, metrics=dict_metrics_i[name], title=name + ": Test"
                )
                save_close_fig(
                    fig_regression_error_test,
                    os.path.join(path_figures, "regression_error_test.png"),
                    show=show_plots,
                )
    dict_run_metrics[j] = dict_domain_metrics
    dict_run_pred[j] = dict_domain_pred


# Check if table can be generated from dict with `tabulate` otherwise try `df.to_latex()`
# Dataframe
list_domains = list(dict_domain_metrics.keys())
list_models = list(dict_domain_metrics[list_domains[0]].keys())
list_runs = list(range(n_runs))
list_names = ["Run", "Domain", "Model"]
df_idx = pd.MultiIndex.from_product([list_runs, list_domains, list_models], names=list_names)
df_idx_avg = pd.MultiIndex.from_product([list_runs, ["Avg."], list_models], names=list_names)

# Dataframe index and labels
df_columns = ["R2", "MSE", "MAE"]
df_header = ["R" + r"$^2$" + r"$(\uparrow)$", "MSE" + r"$(\downarrow)$"]  # , "MAE" + r"$(\downarrow)$"]
df_caption = "Comparison of model performance metrics in extrapolation"
df_label = "tab:bridge_metrics"

# Create dataframes
df_dom = pd.DataFrame(index=df_idx, columns=df_columns)
df_avg = pd.DataFrame(index=df_idx_avg, columns=df_columns)

# Append domain results to dataframe
for run_i, item_i in dict_run_metrics.items():
    for domain_j, item_j in item_i.items():
        for model_k, item_k in item_j.items():
            list_metrics_l = []
            for metric_l, item_l in item_k.items():
                list_metrics_l.append(float(np.mean(item_l)))
            df_dom.loc[run_i, domain_j, model_k] = list_metrics_l

# Average over runs
df_run_agg = df_dom.groupby(level=["Domain", "Model"])[["R2", "MSE", "MAE"]].agg(["mean", "std"])
df_run_agg.columns = ["_".join(col) for col in df_run_agg.columns]
df_run_agg.reset_index(inplace=True)
df_run_agg["Domain"] = df_run_agg["Domain"].mask(df_run_agg["Domain"].duplicated(), "")

# Average over runs and domains
df_dom_agg = df_dom.groupby(level=["Model"])[["R2", "MSE", "MAE"]].agg(["mean", "std"])
df_dom_agg.columns = ["_".join(col) for col in df_dom_agg.columns]
df_dom_agg.reset_index(inplace=True)
df_dom_agg.insert(0, "Domain", "Avg.")
df_dom_agg["Domain"] = df_dom_agg["Domain"].mask(df_dom_agg["Domain"].duplicated(), "")

# Concatenate
df_agg = pd.concat([df_run_agg, df_dom_agg]).reset_index(drop=True)

# Format +- std. deviations
df_agg_f = pd.DataFrame(
    {
        "Domain": df_agg["Domain"],
        "Model": df_agg["Model"],
        "R2": df_agg["R2_mean"].map("{:.3f}".format) + " $\\pm$ " + df_agg["R2_std"].map("{:.3f}".format),
        "MSE": df_agg["MSE_mean"].map("{:.3f}".format) + " $\\pm$ " + df_agg["MSE_std"].map("{:.3f}".format),
        # 'MAE': df_agg['MAE_mean'].map('{:.3f}'.format) + ' $\\pm$ ' + df_agg['MAE_std'].map('{:.3f}'.format),
    },
    index=df_agg.index,
)

df_agg_f_index = df_agg_f[["Domain", "Model"]]
df_agg_f_values = df_agg_f.drop(columns=["Domain", "Model"])

# Export to latex table
latex_table_index = df_agg_f_index.to_latex(
    index=False,
    index_names=False,
    caption=df_caption,
    header=["Domain", "Model"],
    float_format="%.3f",
    position="htb!",
)
latex_table_values = df_agg_f_values.to_latex(
    index=False, index_names=False, caption=df_caption, header=df_header, float_format="%.3f", position="htb!"
)
