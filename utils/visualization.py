import numpy as np
import pandas as pd
import torch


# Plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
mpl.rcParams['figure.dpi'] = 300

# Local imports
from utils import cmap_name, cmap_vars, alpha_interp
from utils.loss import get_logger_training_curve
from utils.priors import get_prior_dist, MarginalDistribution
from utils.data import sample_response


def save_close_fig(fig, path, show=False):
    fig.savefig(path)
    if show is True:
        plt.show()
    else:
        plt.close(fig)


def visualize_training_loss(logger, n_skip_train=0, n_skip_val=0):

    # Get iters
    list_iters_train, _ = get_logger_training_curve(logger, "ELBO")
    list_iters_val, _ = get_logger_training_curve(logger, "ELBO_val")

    # Get training log
    _, list_elbo = get_logger_training_curve(logger, "ELBO")
    _, list_KLx = get_logger_training_curve(logger, "KLx")
    _, list_KLy = get_logger_training_curve(logger, "KLy")
    _, list_Rx = get_logger_training_curve(logger, "Rx")
    _, list_Rc = get_logger_training_curve(logger, "Rc")
    _, list_Ry = get_logger_training_curve(logger, "Ry")
    _, list_reg = get_logger_training_curve(logger, "reg")
    _, list_lambda_x = get_logger_training_curve(logger, "lambda_x")
    _, list_beta_x = get_logger_training_curve(logger, "beta_x")
    _, list_beta_y = get_logger_training_curve(logger, "beta_y")
    _, list_sigma_x = get_logger_training_curve(logger, "sigma_x")

    # Get validation log
    _, list_elbo_val = get_logger_training_curve(logger, "ELBO_val")
    _, list_KLx_val = get_logger_training_curve(logger, "KLx_val")
    _, list_KLy_val = get_logger_training_curve(logger, "KLy_val")
    _, list_Rx_val = get_logger_training_curve(logger, "Rx_val")
    _, list_Rc_val = get_logger_training_curve(logger, "Rc_val")
    _, list_Ry_val = get_logger_training_curve(logger, "Ry_val")
    _, list_reg_val = get_logger_training_curve(logger, "reg_val")

    fig, ax = plt.subplots(5, 1, figsize=(16, 9))

    # Train/validation ELBO
    ax[0].plot(
        list_iters_train[n_skip_train:], list_elbo[n_skip_train:], label="Training", c="blue", alpha=0.3
    )
    ax[0].scatter(list_iters_val[n_skip_val:], list_elbo_val[n_skip_val:], label="Validation", c="red")
    ax[0].grid()
    ax[0].set_ylabel("ELBO")

    # Train/validation Rx
    ax[1].plot(list_iters_train[n_skip_train:], list_Rx[n_skip_train:], label="Training", c="blue", alpha=0.8)
    ax_t1 = ax[1].twinx()
    ax_t1.plot(list_iters_val[n_skip_val:], list_Rx_val[n_skip_val:], label="Validation", color="red")
    ax[1].yaxis.label.set_color("blue")
    ax[1].tick_params(axis="y", colors="blue")
    ax_t1.yaxis.label.set_color("red")
    ax_t1.tick_params(axis="y", colors="red")
    ax[1].grid()
    ax[1].set_ylabel("Rx")
    ax_t1.set_ylabel("Rx_val")

    # Train/validation Ry
    ax[2].plot(list_iters_train[n_skip_train:], list_Ry[n_skip_train:], label="Training", c="blue", alpha=0.8)
    ax_t2 = ax[2].twinx()
    ax_t2.plot(list_iters_val[n_skip_val:], list_Ry_val[n_skip_val:], label="Validation", color="red")
    ax[2].yaxis.label.set_color("blue")
    ax[2].tick_params(axis="y", colors="blue")
    ax_t2.yaxis.label.set_color("red")
    ax_t2.tick_params(axis="y", colors="red")
    ax[2].grid()
    ax[2].set_ylabel("Ry")
    ax_t2.set_ylabel("Ry_val")

    # Train/validation Rc
    ax[3].plot(list_iters_train[n_skip_train:], list_Rc[n_skip_train:], label="Training", c="blue", alpha=0.8)
    ax_t3 = ax[3].twinx()
    ax_t3.plot(list_iters_val[n_skip_val:], list_Rc_val[n_skip_val:], label="Validation", color="red")
    ax[3].yaxis.label.set_color("blue")
    ax[3].tick_params(axis="y", colors="blue")
    ax_t3.yaxis.label.set_color("red")
    ax_t3.tick_params(axis="y", colors="red")
    ax[3].grid()
    ax[3].set_ylabel("Rc")
    ax_t3.set_ylabel("Rc_val")

    # Train/validation KL
    ax[4].plot(
        list_iters_train[n_skip_train:], list_KLx[n_skip_train:], label="Training", c="blue", alpha=0.8
    )
    ax_t4 = ax[4].twinx()
    ax_t4.plot(list_iters_val[n_skip_val:], list_KLx_val[n_skip_val:], label="Training", c="red")
    ax[4].yaxis.label.set_color("blue")
    ax[4].tick_params(axis="y", colors="blue")
    ax_t4.yaxis.label.set_color("red")
    ax_t4.tick_params(axis="y", colors="red")
    ax[4].grid()
    ax[4].set_ylabel("KL")
    ax_t4.set_ylabel("KL_val")

    return fig, ax


def plot_regression_error(y_test, y_pred, definition, metrics=None, title=None):
    """
    Parameters
    ----------
    y_test: Array or tensor with shape (n_test, nd_y)
    y_pred: Dict where each item is an array or tensor with shape (n_test, nd_y)
        corresponding to a different model
    definition: Dict containing the problem definition
    metrics: Optional dict of pre-computed metrics for each model

    Returns
    -------

    """
    dict_gt = definition["dict_gt"]
    nd_y = definition["nd_y"]
    z_idx_y = [idx for idx, val in enumerate(dict_gt.values()) if val["type"] == "y"]
    labels = [item["label"] for item in dict_gt.values()]

    y_test = y_test.detach().cpu() if torch.is_tensor(y_test) else y_test
    y_pred = y_pred.detach().cpu() if torch.is_tensor(y_pred) else y_pred

    y_test = np.expand_dims(y_test, -1) if y_test.ndim == 1 else y_test
    y_pred = np.expand_dims(y_pred, -1) if y_pred.ndim == 1 else y_pred

    fig, ax = plt.subplots(1, nd_y, figsize=(3 * nd_y, 4))
    ax = np.atleast_1d(ax)
    for i in range(nd_y):
        diag_min = np.min(y_pred[:, i])
        diag_max = np.max(y_pred[:, i])
        xy_diag = np.array([diag_min, diag_max])
        # ax[i].errorbar(y_test[..., i], y_pred[..., i].mean(dim=0).squeeze(), yerr=2 * y_pred[..., i].std(dim=0).squeeze(), fmt="o", alpha=0.2, markersize=0)
        ax[i].scatter(y_test[:, i], y_pred[:, i], c="red", s=3.0)
        ax[i].plot(xy_diag, xy_diag, linestyle="dashed", c="black", linewidth=2.0, alpha=0.5)

        # Add metrics if available
        if metrics is not None:
            for j, (metric_j, score_j) in enumerate(metrics.items()):
                ax[i].text(
                    0.1,
                    0.90 - j * 0.05,
                    metric_j + "=" + "{:.3f}".format(score_j[i]),
                    fontsize=12,
                    transform=ax[i].transAxes,
                )

        # Title, grid etc
        ax[i].set_title(labels[z_idx_y[i]])
        ax[i].grid()

    if title is not None:
        plt.suptitle(title)

    return fig, ax


def plot_ground_truth_posterior(model, sample_dist, definition, n_plot=1000, cond=False):
    # Get problem definition
    dict_gt = definition["dict_gt"]
    z_idx_x = [idx for idx, item in enumerate(dict_gt.values()) if item["type"] == "x"]

    # Priors over VAE latent space
    dict_prior_x = definition["dict_prior_x"]
    list_prior_x = [item["dist"](**item["args"]) for key, item in dict_prior_x.items()]
    prior_zx = MarginalDistribution(list_prior_x)

    # Generate ground truth data while interpolating over the latent space
    with torch.no_grad():
        x_gt_plot, c_gt_plot, y_gt_plot, z_gt_plot = sample_response(
            definition, n_plot, sample_dist=sample_dist
        )

        # Evaluate VAE interpolating across the latent space
        (
            xh_gt_plot,
            x_p_gt_plot,
            x_d_gt_plot,
            ch_gt_plot,
            yh_gt_plot,
            z_x_gt_plot,
            z_c_gt_plot,
            z_y_gt_plot,
            dens_z_gt_plot,
        ) = model.sample(x_gt_plot, c_gt_plot, cond=cond, n=1)

        # Convert from cuda
        z_gt_plot = z_gt_plot.detach().cpu()
        z_x_gt_plot = z_x_gt_plot.detach().cpu().squeeze(0)

        # Ground truth and aggregated posterior of physics-based latent variables
        df_gt_phys = pd.DataFrame(z_gt_plot[:, z_idx_x].detach().cpu().numpy())
        df_gt_phys.columns = [item["label"] for item in dict_gt.values() if item["type"] == "x"]
        df_gt_phys.insert(0, "type", ["Ground truth"] * n_plot)
        df_post_phys = pd.DataFrame(z_x_gt_plot.squeeze().detach().cpu().numpy())
        df_post_phys.columns = [item["label"] for item in dict_gt.values() if item["type"] == "x"]
        df_post_phys.insert(0, "type", ["Posterior Zp"] * n_plot)
        df_prior_phys = pd.DataFrame(prior_zx.sample((n_plot,)).detach().cpu().numpy())
        df_prior_phys.columns = [item["label"] for item in dict_gt.values() if item["type"] == "x"]
        df_prior_phys.insert(0, "type", ["Prior"] * n_plot)
        df_phys_plot = pd.concat([df_prior_phys, df_gt_phys, df_post_phys])
        plot_part_latent = sns.pairplot(df_phys_plot, hue="type", kind="hist")
        plot_part_latent.fig.suptitle("Ground truth and posterior " + r"$z_p$")

    return plot_part_latent.fig


def interp_corner_latent_space(model, idx_z_interp, n_interp, definition, n_plot=1000, cond=False):
    # Ground truth
    dict_gt = definition["dict_gt"]
    dist_gt = get_prior_dist(dict_gt)
    interp_vals = torch.tensor([item["val"] for item in dict_gt.values()])

    t = definition["t"]
    t_max = definition["t_max"]
    param_labels = [item["label"] for item in dict_gt.values()]
    dist_gt = get_prior_dist(definition["dict_gt"])

    interp_lb = dist_gt.icdf(torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_ub = dist_gt.icdf(1.0 - torch.ones(len(dict_gt)) * alpha_interp).squeeze()

    with torch.no_grad():

        # Generate ground truth data while interpolating over the latent space
        z_linsp = torch.linspace(interp_lb[idx_z_interp], interp_ub[idx_z_interp], n_interp)
        z_gt_interp = interp_vals.unsqueeze(0).repeat(n_interp, 1)
        z_gt_interp[:, idx_z_interp] = z_linsp
        x_interp, c_interp, y_interp, _ = sample_response(definition, n_plot, z=z_gt_interp)
        z_gt_interp = z_gt_interp.detach().cpu()

        df_interp_list = []
        # fig, ax = plt.figure(nz_x + nz_y, nz_x + nz_y)

        for idx_interp in range(n_interp):
            # Evaluate VAE interpolating across the latent space
            (
                xh_interp,
                x_p_interp,
                x_d_interp,
                ch_interp,
                yh_interp,
                z_x_interp,
                z_c_interp,
                z_y_interp,
                dens_z_interp,
            ) = model.sample(x_interp[:, idx_interp], c_interp[:, idx_interp], cond=cond)

            # Convert to cuda
            z_x_interp = z_x_interp.detach().cpu().squeeze(0)
            z_y_interp = z_y_interp.detach().cpu().squeeze(0)

            # Aggregated posterior of the data-driven latent variables
            z_p_interp_i = torch.hstack((z_x_interp, z_y_interp)).detach().numpy()

            # # Using corner
            # fig_i = corner.corner(z_p_interp_i)

            # Using seaborn
            df_post_interp_i = pd.DataFrame(z_p_interp_i)
            df_post_interp_i.insert(
                0,
                "type",
                [param_labels[idx_z_interp] + " = " + str(z_linsp[idx_interp].detach().numpy())] * n_plot,
            )
            df_interp_list.append(df_post_interp_i)

        df_post_interp = pd.concat(df_interp_list)
        plot_part_latent = sns.pairplot(
            df_post_interp, hue="type", kind="hist", diag_kind="kde", palette="plasma"
        )
        plot_part_latent.fig.suptitle("Posterior")

    return plot_part_latent.fig


def plot_marginal_prior(model, args, definition, n_plot=1000):
    # Ground truth
    dict_gt = definition["dict_gt"]
    interp_vals = torch.tensor([item["val"] for item in dict_gt.values()])

    n_interp = args.n_interp
    nz_c = args.nz_c
    nz_y = args.nz_y
    n_z = nz_c + nz_y

    param_labels = [item["label"] for item in dict_gt.values()]
    dist_gt = get_prior_dist(definition["dict_gt"])
    interp_lb = dist_gt.icdf(torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_ub = dist_gt.icdf(1.0 - torch.ones(len(dict_gt)) * alpha_interp).squeeze()

    cmap_interp = mpl.colormaps[cmap_name](np.linspace(0.0, 1.0, n_interp))
    fig, ax = plt.subplots(n_z, len(dict_gt), figsize=(12, 6), layout="compressed", sharey="row", sharex="row")

    with torch.no_grad():
        for idx_z_interp in range(len(dict_gt)):
            # Generate ground truth data while interpolating over the latent space
            vec_z = torch.linspace(interp_lb[idx_z_interp], interp_ub[idx_z_interp], n_interp)
            z_gt_interp = interp_vals.unsqueeze(0).repeat(n_interp, 1)
            z_gt_interp[:, idx_z_interp] = vec_z
            x_interp, c_interp, y_interp, _ = sample_response(definition, n_plot, z=z_gt_interp)

            # Create colorbar for the current latent variable
            norm_bar = Normalize(vmin=vec_z[0], vmax=vec_z[-1])
            cmap_bar = LinearSegmentedColormap.from_list(cmap_name, cmap_interp, N=n_interp)
            smap_bar = ScalarMappable(norm_bar, cmap=cmap_bar)

            # Get variable labels and initialize dataframe
            zc_labels = [r"$z_\mathrm{c}$" + r"$_{{{}}}$".format(_idx) for _idx in range(nz_c)]
            zy_labels = [r"$z_\mathrm{y}$" + r"$_{{{}}}$".format(_idx) for _idx in range(nz_y)]
            z_labels = zc_labels + zy_labels
            df = pd.DataFrame(columns=["type"] + z_labels)

            # Interpolate latent space for variable `i` and append to df
            for i in range(n_interp):
                # Evaluate VAE interpolating across the latent space
                (
                    zc_interp,
                    dens_zc_interp,
                    zy_interp,
                    dens_zy_interp,
                ) = model.sample_prior(c_interp[:, i], y_interp[:, i])

                zc_interp = zc_interp.squeeze(0).detach().cpu()
                zy_interp = zy_interp.squeeze(0).detach().cpu()
                z_interp = torch.hstack((zc_interp, zy_interp)).numpy()
                z_interp_dict = {key: value for key, value in zip(z_labels, z_interp.T)}
                z_interp_dict["type"] = [float(vec_z[i].detach().numpy())] * n_plot

                df_new = pd.DataFrame.from_dict(z_interp_dict)
                df = pd.concat([df.astype(df_new.dtypes), df_new])

            # Plot
            for j in range(n_z):
                ax_ij = ax[j, idx_z_interp]

                sns.kdeplot(
                    data=df, x=z_labels[j], hue="type", palette="plasma", ax=ax_ij, fill=True, legend=False
                )
                ax_ij.spines[["right", "top"]].set_visible(False)
                ax_ij.set(yticklabels=[])
                ax_ij.set_yticks([])
                ax_ij.set_ylabel(z_labels[j])
                ax_ij.set_xlabel(None)

            # Add colorbar
            cbar_post = fig.colorbar(
                smap_bar,
                ax=ax[0, idx_z_interp],
                orientation="horizontal",
                location="top",
                fraction=1.0,
                pad=0.2,
            )

            cbar_post.set_label(label=param_labels[idx_z_interp], size=14)
            cbar_post.ax.tick_params(labelsize=10)
    return fig, ax


def plot_marginal_post(model, args, definition, vars_interp=None, n_plot=1000, cond=False):
    # Ground truth
    dict_gt = definition["dict_gt"]
    dist_gt = get_prior_dist(dict_gt)
    interp_vals = torch.tensor([item["val"] for item in dict_gt.values()])

    nz_x = definition["nz_x"]
    nz_c = args.nz_c
    nz_y = args.nz_y
    n_z = nz_x + nz_c + nz_y

    interp_lb = dist_gt.icdf(torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_ub = dist_gt.icdf(1.0 - torch.ones(len(dict_gt)) * alpha_interp).squeeze()

    # Note: For the bridge case study, the height of the figure is increased
    # to properly plot the y labels
    if vars_interp is None:
        vars_interp = range(len(dict_gt))
        # figsize = (12, 6)     # For other case studies
        figsize = (15, 8)       # For bridge case study
    else:
        figsize = (3 * len(vars_interp), 0.8 * n_z)

    # Get labels
    list_var_gt = list(dict_gt.values())

    cmap_interp = mpl.colormaps[cmap_name](np.linspace(0.0, 1.0, args.n_interp))
    fig, ax = plt.subplots(n_z, len(vars_interp), figsize=figsize, layout="compressed", sharex="row")#, sharey="row", sharex="row")

    with torch.no_grad():
        for col, idx_z_interp in enumerate(vars_interp):
            # Generate ground truth data while interpolating over the latent space
            z_linsp_i = torch.linspace(interp_lb[idx_z_interp], interp_ub[idx_z_interp], args.n_interp)
            z_gt_interp = interp_vals.unsqueeze(0).repeat(args.n_interp, 1)
            z_gt_interp[:, idx_z_interp] = z_linsp_i
            var_gt = list_var_gt[idx_z_interp]
            x_interp, c_interp, y_interp, _ = sample_response(definition, n_plot, z=z_gt_interp)

            # Create colorbar for the current latent variable
            norm_bar = Normalize(vmin=z_linsp_i[0], vmax=z_linsp_i[-1])
            cmap_bar = LinearSegmentedColormap.from_list(cmap_name, cmap_interp, N=args.n_interp)
            smap_bar = ScalarMappable(norm_bar, cmap=cmap_bar)

            # Get variable labels and initialize dataframe
            zx_labels = [item["label"] for item in dict_gt.values() if item["type"] == "x"]
            zc_labels = [r"$z_\mathrm{c},$" + r"$_{{{}}}$".format(_idx + 1) for _idx in range(nz_c)]
            zy_labels = [r"$z_\mathrm{y},$" + r"$_{{{}}}$".format(_idx + 1) for _idx in range(nz_y)]
            z_labels = zx_labels + zc_labels + zy_labels
            z_types = ["x"] * nz_x + ["c"] * nz_c + ["y"] * nz_y
            df = pd.DataFrame(columns=["type"] + z_labels)

            # Interpolate latent space for variable `i` and append to df
            for i in range(args.n_interp):
                # Evaluate VAE interpolating across the latent space
                (
                    xh_interp,
                    x_p_interp,
                    x_d_interp,
                    ch_interp,
                    yh_interp,
                    z_x_interp,
                    z_c_interp,
                    z_y_interp,
                    dens_z_interp,
                ) = model.sample(x_interp[:, i], c_interp[:, i], cond=cond)

                z_x_interp = z_x_interp.squeeze(0).detach().cpu()
                z_c_interp = z_c_interp.squeeze(0).detach().cpu()
                z_y_interp = z_y_interp.squeeze(0).detach().cpu()
                z_interp = torch.hstack((z_x_interp, z_c_interp, z_y_interp)).numpy()
                z_interp_dict = {key: value for key, value in zip(z_labels, z_interp.T)}
                z_interp_dict["type"] = [float(z_linsp_i[i].detach().numpy())] * n_plot

                df_new = pd.DataFrame.from_dict(z_interp_dict)
                df = pd.concat([df.astype(df_new.dtypes), df_new])

                # TODO: Normalize dataframe

            # Plot
            for j in range(n_z):
                ax_ij = ax[j, col]

                sns.kdeplot(
                    data=df, x=z_labels[j], hue="type", palette="plasma", ax=ax_ij, fill=True, legend=False
                )
                ax_ij.spines[["right", "top"]].set_visible(False)
                ax_ij.set(yticklabels=[])
                ax_ij.set_yticks([])
                ax_ij.set_ylabel(z_labels[j], color=cmap_vars[z_types[j]], size=12)
                ax_ij.set_xlabel(None)

            # Add colorbar
            cbar_post = fig.colorbar(
                smap_bar,
                ax=ax[0, col],
                orientation="horizontal",
                location="top",
                fraction=1.0,
                pad=0.2,
            )

            cbar_post.set_label(label=var_gt["label"], size=14, color=cmap_vars[var_gt["type"]])
            cbar_post.ax.tick_params(labelsize=10)
    return fig, ax


def plot_interp_pred(model, n_interp, definition, n_plot=1000, cond=False):
    # Ground truth
    dict_gt = definition["dict_gt"]
    dist_gt = get_prior_dist(dict_gt)

    # Axis
    t = definition["t"].detach().cpu()
    ylim = definition["ylim"]
    x_unit = definition["x_unit"]
    y_unit = definition["y_unit"]

    # Interpolation
    interp_lb = dist_gt.icdf(torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_ub = dist_gt.icdf(1.0 - torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_vals = torch.tensor([item["val"] for item in dict_gt.values()])

    # Colors
    cmap_interp = mpl.colormaps[cmap_name](np.linspace(0.0, 1.0, n_interp))

    # Create plot
    fig_pred_x, ax_pred_x = plt.subplots(
        3, len(dict_gt), figsize=(16, 9), sharex="col", sharey="row", layout="compressed"
    )

    with torch.no_grad():
        for idx_z_interp, z_interp_i in enumerate(definition["dict_gt"].values()):

            # Generate ground truth data while interpolating over the latent space
            z_linsp_i = torch.linspace(interp_lb[idx_z_interp], interp_ub[idx_z_interp], n_interp)
            z_gt_interp = interp_vals.unsqueeze(0).repeat(n_interp, 1)
            z_gt_interp[:, idx_z_interp] = z_linsp_i
            x_interp, c_interp, y_interp, _ = sample_response(definition, n_plot, z=z_gt_interp)

            # Create colorbar for the current latent variable
            norm_bar = Normalize(vmin=z_linsp_i[0], vmax=z_linsp_i[-1])
            cmap_bar = LinearSegmentedColormap.from_list(cmap_name, cmap_interp, N=n_interp)
            smap_bar = ScalarMappable(norm_bar, cmap=cmap_bar)

            for i in range(n_interp):
                # Evaluate VAE interpolating across the latent space
                (
                    xh_interp,
                    x_p_interp,
                    x_d_interp,
                    ch_interp,
                    yh_interp,
                    z_x_interp,
                    z_c_interp,
                    z_y_interp,
                    dens_z_interp,
                ) = model.sample(x_interp[:, i], c_interp[:, i], cond=cond)

                x_p_interp = x_p_interp.detach().cpu()
                x_d_interp = x_d_interp.detach().cpu()
                xh_interp_mean = xh_interp.mean(dim=1).squeeze(0).detach().cpu()
                xh_interp_std = xh_interp.std(dim=1).squeeze(0).detach().cpu()
                x_p_interp_mean = x_p_interp.mean(dim=1).squeeze(0).detach().cpu()
                x_p_interp_std = x_p_interp.std(dim=1).squeeze(0).detach().cpu()
                x_d_interp_mean = x_d_interp.mean(dim=1).squeeze(0).detach().cpu()
                x_d_interp_std = x_d_interp.std(dim=1).squeeze(0).detach().cpu()

                # Physics-based model prediction
                ax_pred_x[0, idx_z_interp].fill_between(
                    t,
                    x_p_interp_mean - 2.0 * x_p_interp_std,
                    x_p_interp_mean + 2.0 * x_p_interp_std,
                    alpha=0.5,
                    color=cmap_interp[i],
                )
                ax_pred_x[0, idx_z_interp].plot(
                    t,
                    x_p_interp_mean,
                    alpha=0.5,
                    color=cmap_interp[i],
                    label=z_interp_i["label"] + r"$={:.3f}$".format(z_linsp_i[i]),
                )


                # Data-driven model prediction
                ax_pred_x[1, idx_z_interp].fill_between(
                    t,
                    x_d_interp_mean - 2.0 * x_d_interp_std,
                    x_d_interp_mean + 2.0 * x_d_interp_std,
                    alpha=0.3,
                    color=cmap_interp[i],
                )
                ax_pred_x[1, idx_z_interp].plot(t, x_d_interp_mean, alpha=0.5, color=cmap_interp[i])


                # Combined prediction
                ax_pred_x[2, idx_z_interp].fill_between(
                    t,
                    xh_interp_mean - 2 * xh_interp_std,
                    xh_interp_mean + 2 * xh_interp_std,
                    alpha=0.5,
                    color=cmap_interp[i],
                )
                ax_pred_x[2, idx_z_interp].scatter(
                    t, x_interp[:, i].mean(dim=0).detach().cpu().numpy(), color=cmap_interp[i]
                )
                ax_pred_x[2, idx_z_interp].plot(t, xh_interp_mean, alpha=0.5, color=cmap_interp[i])


                # Grids, limits etc
                # ax_pred_x[0, idx_z_interp].set_ylim(ylim)
                # ax_pred_x[1, idx_z_interp].set_ylim(ylim)
                # ax_pred_x[2, idx_z_interp].set_ylim(ylim)
                ax_pred_x[0, idx_z_interp].grid()
                ax_pred_x[1, idx_z_interp].grid()
                ax_pred_x[2, idx_z_interp].grid()

            ax_pred_x[2, idx_z_interp].set_xlabel(x_unit, fontsize=16)
            cbar_pred = fig_pred_x.colorbar(
                smap_bar, ax=ax_pred_x[0, idx_z_interp], orientation="horizontal", location="top"
            )
            cbar_pred.set_label(label=z_interp_i["label"], size=18, color=cmap_vars[z_interp_i["type"]])
            cbar_pred.ax.tick_params(labelsize=12)

        ax_pred_x[0, 0].set_ylabel(r"$\hat{x_\mathrm{p}}$ " + y_unit, fontsize=18)
        ax_pred_x[1, 0].set_ylabel(r"$\hat{x_\mathrm{d}}$ " + y_unit, fontsize=18)
        ax_pred_x[2, 0].set_ylabel(r"$\hat{x}$ " + y_unit, fontsize=18)
    return fig_pred_x, ax_pred_x


def plot_pred(model, n_interp, idx_var_gt, definition, n_plot=1000, cond=False):
    # Ground truth
    dict_gt = definition["dict_gt"]
    dist_gt = get_prior_dist(dict_gt)
    var_key = list(dict_gt.keys())[idx_var_gt]
    var_gt = dict_gt[var_key]

    # Axis
    t = definition["t"].detach().cpu()
    x_unit = definition["x_unit"]
    y_unit = definition["y_unit"]

    # Interpolation
    interp_vals = torch.tensor([item["val"] for item in dict_gt.values()])
    interp_lb = dist_gt.icdf(torch.ones(len(dict_gt)) * alpha_interp).squeeze()
    interp_ub = dist_gt.icdf(1.0 - torch.ones(len(dict_gt)) * alpha_interp).squeeze()

    # Colors
    cmap_interp = mpl.colormaps[cmap_name](np.linspace(0.0, 1.0, n_interp))

    # Figure
    fig_pred_x, ax_pred_x = plt.subplots(
        1, 3, figsize=(9, 3), layout="compressed"
    )

    with torch.no_grad():

        # Generate ground truth data while interpolating over the latent space
        z_linsp = torch.linspace(interp_lb[idx_var_gt], interp_ub[idx_var_gt], n_interp)
        z_gt_interp = interp_vals.unsqueeze(0).repeat(n_interp, 1)
        z_gt_interp[:, idx_var_gt] = z_linsp
        x_interp, c_interp, y_interp, _ = sample_response(definition, n_plot, z=z_gt_interp)

        # Create colorbar for the current latent variable
        norm_bar = Normalize(vmin=z_linsp[0], vmax=z_linsp[-1])
        cmap_bar = LinearSegmentedColormap.from_list(cmap_name, cmap_interp, N=n_interp)
        smap_bar = ScalarMappable(norm_bar, cmap=cmap_bar)

        for i in range(n_interp):
            # Evaluate VAE interpolating across the latent space
            (
                xh_interp,
                x_p_interp,
                x_d_interp,
                ch_interp,
                yh_interp,
                z_x_interp,
                z_c_interp,
                z_y_interp,
                dens_z_interp,
            ) = model.sample(x_interp[:, i], c_interp[:, i], cond=cond)

            x_p_interp = x_p_interp.detach().cpu()
            x_d_interp = x_d_interp.detach().cpu()
            xh_interp_mean = xh_interp.mean(dim=1).squeeze(0).detach().cpu()
            xh_interp_std = xh_interp.std(dim=1).squeeze(0).detach().cpu()
            x_p_interp_mean = x_p_interp.mean(dim=1).squeeze(0).detach().cpu()
            x_p_interp_std = x_p_interp.std(dim=1).squeeze(0).detach().cpu()
            x_d_interp_mean = x_d_interp.mean(dim=1).squeeze(0).detach().cpu()
            x_d_interp_std = x_d_interp.std(dim=1).squeeze(0).detach().cpu()

            # Physics-based model prediction
            ax_pred_x[0].fill_between(
                t,
                x_p_interp_mean - 2.0 * x_p_interp_std,
                x_p_interp_mean + 2.0 * x_p_interp_std,
                alpha=0.2,
                color=cmap_interp[i],
            )
            ax_pred_x[0].plot(
                t,
                x_p_interp_mean,
                alpha=0.5,
                color=cmap_interp[i],
                label=var_gt["label"] + r"$={:.3f}$".format(z_linsp[i]),
            )

            # Data-driven model prediction
            ax_pred_x[1].fill_between(
                t,
                x_d_interp_mean - 2.0 * x_d_interp_std,
                x_d_interp_mean + 2.0 * x_d_interp_std,
                alpha=0.2,
                color=cmap_interp[i],
            )
            ax_pred_x[1].plot(t, x_d_interp_mean, alpha=0.5, color=cmap_interp[i])


            # Combined prediction
            ax_pred_x[2].fill_between(
                t,
                xh_interp_mean - 2.0 * xh_interp_std,
                xh_interp_mean + 2.0 * xh_interp_std,
                alpha=0.2,
                color=cmap_interp[i],
            )
            ax_pred_x[2].plot(t, xh_interp_mean, alpha=1.0, linestyle="solid", color=cmap_interp[i])
            ax_pred_x[2].scatter(
                t, x_interp[:, i].mean(dim=0).detach().cpu().numpy(), alpha=1.0, s=8.0, color=cmap_interp[i]
            )


        # Grids, limits etc
        # ax_pred_x[0].set_ylim(ylim)
        # ax_pred_x[1].set_ylim(ylim)
        # ax_pred_x[2].set_ylim(ylim)
        ax_pred_x[0].grid()
        ax_pred_x[1].grid()
        ax_pred_x[2].grid()

        ax_pred_x[0].set_xlabel(x_unit, fontsize=16)
        ax_pred_x[1].set_xlabel(x_unit, fontsize=16)
        ax_pred_x[2].set_xlabel(x_unit, fontsize=16)

        cbar_pred = fig_pred_x.colorbar(
            smap_bar, ax=ax_pred_x[-1], orientation="vertical", location="right"
        )
        cbar_pred.set_label(label=var_gt["label"], size=18, color=cmap_vars[var_gt["type"]])
        cbar_pred.ax.tick_params(labelsize=12)

        ax_pred_x[0].set_ylabel(r"$\hat{x_\mathrm{p}}$ " + y_unit, fontsize=18)
        ax_pred_x[1].set_ylabel(r"$\hat{x_\mathrm{d}}$ " + y_unit, fontsize=18)
        ax_pred_x[2].set_ylabel(r"$\hat{x}$ " + y_unit, fontsize=18)
    return fig_pred_x, ax_pred_x
