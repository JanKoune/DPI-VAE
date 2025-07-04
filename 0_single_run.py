import os
import json

from utils import make_parser
from utils.priors import get_prior_dist
from utils.data import sample_response
from utils.visualization import (
    save_close_fig,
    plot_pred,
    plot_interp_pred,
    plot_marginal_post,
    plot_marginal_prior,
    plot_regression_error,
    visualize_training_loss,
    plot_ground_truth_posterior,
    interp_corner_latent_space
)
from dpivae import setup_model, run_comparison, evaluate_model, train_model
from cases import simple_beam, damped_oscillator, bridge

# ============================================================
# SETUP
# ============================================================
# Problem definition
name = "single_run"
case = simple_beam
preset = "dpivae"

# Options
plot_interpolation = True
plot_prediction = True
plot_aggregated = True
plot_regression = True
show_plots = True
save_plots = True
cond = False

# Parse arguments and adjust for case presets
parser = make_parser()
args, unknown = parser.parse_known_args()
presets = case.presets[preset]
vars_args = vars(args)
for key, item in presets.items():
    vars_args[key] = item

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
# Distributions
dist_gt = get_prior_dist(definition["dict_gt"])

# Train
x_train, c_train, y_train, z_train = sample_response(
    definition, args.n_train, sample_dist=dist_gt
)

# Validation
x_val, c_val, y_val, z_val = sample_response(
    definition, args.n_val, sample_dist=dist_gt
)

# Test
x_test, c_test, y_test, z_test = sample_response(
    definition, args.n_test, sample_dist=dist_gt
)

data_train = (x_train, c_train, y_train, z_train)
data_val = (x_val, c_val, y_val, z_val)
data_test = (x_test, c_test, y_test, z_test)

# ============================================================
# TRAINING
# ============================================================
vae = setup_model(args, definition, data_train)
vae, logger = train_model(args, vae, definition, data_train, data_val, path_metrics=path_metrics,
                          path_figures=path_figures)

# Loss curve
fig_loss, ax_loss = visualize_training_loss(
    logger, n_skip_train=args.n_skip_plot_train, n_skip_val=args.n_skip_plot_val
)
save_close_fig(fig_loss, os.path.join(path_figures, "loss_curve.png"), show=show_plots)

# ============================================================
# DISENTANGLEMENT
# ============================================================
# score_disentanglement = disentanglement_metric(args, vae, definition, data_train, data_test)


# ============================================================
# EVALUATION
# ============================================================
if plot_regression == True:
    dict_metrics_test, dict_pred_test = run_comparison(args, definition, data_train, data_test)
    vae_metrics_test, vae_pred_test = evaluate_model(args, definition, vae, data_test, cond=cond)

    # Update dict
    dict_metrics_test.update(vae_metrics_test)
    dict_pred_test.update(vae_pred_test)

    for name, item in dict_pred_test.items():
        fig_regression_error_test, ax_regression_error_test = plot_regression_error(
            y_test, item, definition, metrics=dict_metrics_test[name], title=name + ": Test"
        )
        save_close_fig(
            fig_regression_error_test,
            os.path.join(path_figures, "regression_error_test.png"),
            show=show_plots,
        )

# ============================================================
# PLOTTING
# ============================================================
# Traversing the generative factors
if plot_prediction == True:
    for idx_var_gt, var_gt_i in enumerate(definition["dict_gt"].values()):
        fig_pred_x, ax_pred_x = plot_pred(
            vae, args.n_interp, idx_var_gt, definition, n_plot=args.n_plot, cond=cond
        )
        save_close_fig(fig_pred_x, os.path.join(path_figures, "fig_pred_x_" + str(idx_var_gt) + ".png"),
                       show=show_plots)

if plot_interpolation == True:
    # ------------------------------------
    # Interpolation in the predictions
    # ------------------------------------
    fig_pred_interp_x, ax_pred_interp_x = plot_interp_pred(
        vae, args.n_interp, definition, n_plot=args.n_plot, cond=cond
    )
    save_close_fig(fig_pred_interp_x, os.path.join(path_figures, "fig_pred_interp_x.png"), show=show_plots)

    # # ------------------------------------
    # # Interpolation in the latent space - Corner plots
    # # ------------------------------------
    # for idx_var_interp in range(len(definition["dict_gt"])):
    #     fig_interp_latent = interp_corner_latent_space(
    #         vae, idx_var_interp, args.n_interp, definition, n_plot=args.n_plot, cond=cond
    #     )
    #     save_close_fig(
    #         fig_interp_latent,
    #         os.path.join(path_figures, "fig_interp_latent_" + str(idx_var_interp) + ".png"),
    #         show=show_plots,
    #     )

    # ------------------------------------
    # Interpolation in the latent space - Posterior marginals
    # ------------------------------------
    fig_post_marginal_z, ax_post_marginal_z = plot_marginal_post(
        vae, args, definition, n_plot=args.n_plot, cond=cond, vars_interp=None
    )
    save_close_fig(fig_post_marginal_z, os.path.join(path_figures, "fig_post_marginal_z.png"), show=show_plots)

    vars_interp = [0, 2]
    fig_post_marginal_z, ax_post_marginal_z = plot_marginal_post(
        vae, args, definition, n_plot=args.n_plot, cond=cond, vars_interp=vars_interp
    )
    save_close_fig(fig_post_marginal_z, os.path.join(path_figures, "fig_post_marginal_z.png"), show=show_plots)

    vars_interp = [1, 2]
    fig_post_marginal_z, ax_post_marginal_z = plot_marginal_post(
        vae, args, definition, n_plot=args.n_plot, cond=cond, vars_interp=vars_interp
    )
    save_close_fig(fig_post_marginal_z, os.path.join(path_figures, "fig_post_marginal_z.png"), show=show_plots)
    # ------------------------------------
    # Interpolation in the latent space - Prior marginals
    # ------------------------------------
    fig_prior_marginal_z, ax_prior_marginal_z = plot_marginal_prior(
        vae, args, definition, n_plot=args.n_plot
    )
    save_close_fig(fig_prior_marginal_z, os.path.join(path_figures, "fig_prior_marginal_z.png"), show=show_plots)

if plot_aggregated == True:
    fig_posterior_ground_truth = plot_ground_truth_posterior(
        vae, dist_gt, definition, n_plot=args.n_plot, cond=cond
    )
    save_close_fig(
        fig_posterior_ground_truth,
        os.path.join(path_figures, "fig_posterior_ground_truth.png"),
        show=show_plots,
    )
