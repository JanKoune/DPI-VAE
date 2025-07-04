"""
Structured variational autoencoder for SHM, using the Variational Information Bottleneck
theory to:

    * Extract and learn damage sensitive features from measurements of the structural response
    * Extract and learn features that are correlated to environmental influences
    * Regularize data-driven components to not override the physics-based components
    * Ground part of the latent space to the known physics for consistency, generalization
        and interpretability

The input data is composed of:

    * Measurements of the structural response
    * Observations of the state of a structure in terms of damage indices
    * Measurements of the relevant environmental parameters, which may or may not
        be included in the physics based model.


Evaluation:
    * The evaluation of the classification accuracy is done following the advice in Sohn et al. (2015)
    https://papers.nips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf. They
    propose using either:
        - Deterministic inference, i.e. y* = argmax_y p(y|x,z*), with z*=E[z|x]
        - Conditional likelihood of the test data p(y|x) = 1/N sum(p(y|x,z)), z~p(z|x)

TODO:
    * Make a new section in the paper Appendix, something like "Caveats", where I explain the
        caveats related to the interaction between the physics-based and data-driven latent spaces
        as it relates to conditional generation and label prediction. Explain that the components of the
        domain and class influences that can be explained by the physics will not be explained by
        the data-driven latent space when the GRL is active, and can result in reduced accuracy in
        conditional generation and label prediction. This issue is not as severe as it seems, since by
        definition it means that these components can be explained by the known physics, which is the
        purpose of including the known physics to begin with. E.g. if damage in a structural component
        is captured by a stiffness reduction in the physics-based latent space, then there is no need
        to predict the corresponding label. The stiffness reduction is enough to signal to an engineer
        that there is damage to that component.
    * Investigate the impact of cond=False, and how it relates to conditional generation
    * Consider a SDOF case based on: P. Franz et al. (2025) - On the potential of aerodynamic pressure measurements
        for Structural Damage Detection. The dataset is publically available and the physics can be simplified.
        We can also consider adding a physics-based latent variable that determines the phase of each signal
        segment such that we can consider timeseries measurements.
    * Also consider the bridge example and data: https://github.com/imcs-compsim/munich-bridge-data
    * Early stopping by monitoring the validation loss of the y prediction
    * Consider changing the regression example to use the logit-normal likelihood for the labels:
    https://arxiv.org/pdf/1505.05770. Then the labels can be a degradation percentage in (0, 1).
    * Test Linear Decoder
    * Test L2 constraint on the cross-decoder (with its own optimizer to not affect optimization of the other components)
    * Remove `param_labels` and `interp_c` keys in case definition dictionaries. Instead include this information
        in the parameter definition dict by adding a `label` and a `ref_value` key.
"""

import os
import json
import torch
from sklearn.preprocessing import StandardScaler as sklearn_scaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch import nn, optim
from torchrl.record import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from utils.loss import EarlyStopping

from tqdm import trange
from utils import device, make_parser
from models.encoders import GaussianEncoder, FullCovarianceNN, FactorizedNN
from models.decoders import GradRevAdditive, Decoder
from models.vae import DPIVAE
from utils.transforms import StandardScaler, ShiftScale, Logistic, ChainTransform, ChainTransformMasked, FuncGradRev, LayerGradRev
from utils.priors import MarginalDistribution, get_prior_dist
from utils.data import sample_response
from utils.annealing import Annealing
from utils.metrics import regression_metrics
from utils.visualization import (
    save_close_fig,
    plot_pred,
    plot_interp_pred,
    plot_marginal_post,
    plot_marginal_prior,
    plot_regression_error,
    visualize_training_loss,
    plot_ground_truth_posterior,
)
from cases import simple_beam


def setup_model(args, definition, data_train):
    # ==================================================================
    # Problem setup
    # ==================================================================
    # Seed
    if args.use_seed == True:
        torch.manual_seed(args.seed)

    print(args)
    # ==================================================================
    # Parse case definition
    # ==================================================================
    # Latents and features
    nz_c = args.nz_c
    nz_y = args.nz_y
    nz_x = definition["nz_x"]
    nd_x = definition["nd_x"]
    nd_c = definition["nd_c"]
    nd_y = definition["nd_y"]
    nd_p = definition["nd_p"]
    nk_y = definition["nk_y"]  # Number of classes in y, if class labels
    n_classes = definition["n_classes"]  # <- Make sure this is None and not 0 for regression

    # Models
    full_model = definition["full_model"]
    part_model = definition["part_model"]

    # Priors over VAE latent space
    dict_prior_x = definition["dict_prior_x"]
    list_prior_x = [item["dist"](**item["args"]) for key, item in dict_prior_x.items()]
    prior_zx = MarginalDistribution(list_prior_x)

    # Physics-based model input covariates
    z_c = [item for key, item in definition["dict_gt"].items() if item["type"] == "c"]
    idx_c_phys = [idx for idx, item in enumerate(z_c) if item["phys"] == True]

    # ==================================================================
    # Check
    # ==================================================================
    # Check that the number of latent variables agree between `dict_gt` and `dict_prior_x`
    if nz_x != len(dict_prior_x):
        raise ValueError("Prior distribution dimension mismatch with ground truth")

    # ==================================================================
    # Transforms
    # ==================================================================
    x_train, c_train, y_train = data_train[0], data_train[1], data_train[2]

    assert(x_train.shape[0] == args.n_train)
    assert(args.n_batch <= args.n_train)

    # Input transforms
    input_transform_x = StandardScaler()
    input_transform_c = StandardScaler()
    input_transform_y = StandardScaler()
    input_transform_x.fit(x_train)
    input_transform_c.fit(c_train)
    input_transform_y.fit(y_train)

    # ==================================================================
    # Prior networks
    # ==================================================================
    if args.full_cov_prior == True:
        prior_net_c = GaussianEncoder(FullCovarianceNN(nz_c, nd_c, [64]))
        prior_net_y = GaussianEncoder(FullCovarianceNN(nz_y, nd_y, [64]))

    elif args.full_cov_prior == False:
        prior_net_c = GaussianEncoder(FactorizedNN(nz_c, nd_c, [64]))
        prior_net_y = GaussianEncoder(FactorizedNN(nz_y, nd_y, [64]))

    else:
        raise ValueError(f"Unknown full_cov_prior argument: {args.full_cov_prior}")

    # ==================================================================
    # Decoders
    # ==================================================================
    func_gradrev_x = FuncGradRev.apply
    layer_gradrev_x = LayerGradRev(func_gradrev_x, alpha=args.lambda_g0)

    # Decoder for x
    decoder_x = GradRevAdditive(part_model, nz_x + nd_p, nz_c + nz_y, nd_x, hidden=128, grad_reverse=layer_gradrev_x)

    # Auxiliary decoder for c
    decoder_c = Decoder(nz_c, nd_c, [64])

    # Auxiliary decoder for y
    decoder_y = Decoder(nz_y, nd_y, [64])

    # ==================================================================
    # Encoders
    #
    # Internally the following transform is used:
    # Sampled z (-inf,+inf) -> Logistic [0, 1] -> ShiftScale [lb, ub]
    # ==================================================================
    # Encoder output transform
    transform_lb = torch.tensor([item["lb"] for key, item in dict_prior_x.items()], device=device)
    transform_ub = torch.tensor([item["ub"] for key, item in dict_prior_x.items()], device=device)
    logistic_transform = Logistic(k=1.0)
    shift_scale = ShiftScale(transform_lb, transform_ub)

    # ===============================================================================
    # Setup for P-DPIVAE
    # ===============================================================================
    if args.model_type == "P":
        output_transform_zx = ChainTransform(logistic_transform, shift_scale)

        # Encoder for x
        if args.encoder_x == "NN":
            encoder_model_x = FullCovarianceNN(nz_x, nd_x, [64])
        # elif args.encoder_x == "CNN":
        #     encoder_model_x = CNNEncoder(nz_x, nd_x, nd_c, ch_in=args.ch_in, ch_out=args.ch_out, ch_latent=args.ch_latent)
        else:
            raise ValueError(f"Unknown encoder x choice: {args.encoder_x}")

        # Encoder for c
        if args.encoder_c == "NN":
            encoder_model_c = FullCovarianceNN(nz_c, nd_x, [64])
        # elif args.encoder_c == "CNN":
        #     encoder_model_c = CNNEncoder(nz_c, nd_x, nd_c, ch_in=args.ch_in, ch_out=args.ch_out, ch_latent=args.ch_latent)
        else:
            raise ValueError(f"Unknown encoder x choice: {args.encoder_x}")

        # Encoder for y
        if args.encoder_y == "NN":
            encoder_model_y = FullCovarianceNN(nz_y, nd_x, [64])
        # elif args.encoder_y == "CNN":
        #     encoder_model_y = CNNEncoder(nz_y, nd_x, nd_c, ch_in=args.ch_in, ch_out=args.ch_out, ch_latent=args.ch_latent)
        else:
            raise ValueError(f"Unknown encoder y choice: {args.encoder_y}")

        # Physics-based encoder
        encoder = GaussianEncoder(
            encoder_model_x,
            output_transform=output_transform_zx,
        )

        # Domain encoder
        encoder_c = GaussianEncoder(
            encoder_model_c,
        )

        # Class encoder
        encoder_y = GaussianEncoder(encoder_model_y)

    # ===============================================================================
    # Setup for S-DPIVAE
    # ===============================================================================
    elif args.model_type == "S":
        z_idx_x = [idx for idx, val in enumerate(definition["dict_gt"].values()) if val["type"] == "x"]
        output_transform_zx = ChainTransformMasked(z_idx_x, logistic_transform, shift_scale)
        if args.encoder_x == "NN":
            encoder_model = FullCovarianceNN(nz_x + nz_c + nz_y, nd_x, [128])
        # elif args.encoder_x == "CNN":
        #     encoder_model = CNNEncoder(nz_x, nd_x, nd_c, ch_in=args.ch_in, ch_out=args.ch_out, ch_latent=args.ch_latent)
        else:
            raise ValueError(f"Unknown encoder choice: {args.encoder_x}")

        # Physics, domain and class encoder
        encoder = GaussianEncoder(
            encoder_model,
            output_transform=output_transform_zx,
        )

        # Set other encoders to None
        encoder_c = None
        encoder_y = None

    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    vae = DPIVAE(
        prior_zx,
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
        model_type=args.model_type,
        encoder_c=encoder_c,
        encoder_y=encoder_y,
        lambda_x=args.lambda_x,
        transform_x=input_transform_x,
        transform_c=input_transform_c,
        transform_y=input_transform_y,
    )

    return vae

def train_model(args, vae, definition, data_train, data_val, path_metrics=None, path_figures=None):
    # Move to device
    vae.to(device)
    vae.compile() # Doesn't seem to make a difference

    # Data
    x_train, c_train, y_train = data_train[0], data_train[1], data_train[2]
    x_val, c_val, y_val = data_val[0], data_val[1], data_val[2]

    # Data shape of last dimension
    nd_x = definition["nd_x"]
    nd_c = definition["nd_c"]
    nd_y = definition["nd_y"]

    # Annealing
    lambda_annealing = args.lambda_annealing
    kwargs_lambda_annealing = {
        "n_cycles": args.lambda_n_cycles,
        "R": args.lambda_R,
        "mu": args.lambda_mu,
        "cov": args.lambda_cov,
    }

    beta_x_annealing = args.beta_x_annealing
    kwargs_beta_x_annealing = {
        "n_cycles": args.beta_x_n_cycles,
        "R": args.beta_x_R,
        "mu": args.beta_x_mu,
        "cov": args.beta_x_cov,
    }

    beta_c_annealing = args.beta_c_annealing
    kwargs_beta_c_annealing = {
        "n_cycles": args.beta_c_n_cycles,
        "R": args.beta_c_R,
        "mu": args.beta_c_mu,
        "cov": args.beta_c_cov,
    }

    beta_y_annealing = args.beta_y_annealing
    kwargs_beta_y_annealing = {
        "n_cycles": args.beta_y_n_cycles,
        "R": args.beta_y_R,
        "mu": args.beta_y_mu,
        "cov": args.beta_y_cov,
    }

    # ==================================================================
    # Optimization
    # ==================================================================
    list_opt_params = []

    # Encoder
    if args.model_type == "P":
        list_opt_params.append({"params": vae.encoder.parameters(), "lr": args.lr_ex, "weight_decay": args.wd_e})
        list_opt_params.append({"params": vae.encoder_c.parameters(), "lr": args.lr_ec, "weight_decay": args.wd_e})
        list_opt_params.append({"params": vae.encoder_y.parameters(), "lr": args.lr_ey, "weight_decay": args.wd_e})
    elif args.model_type == "S":
        list_opt_params.append({"params": vae.encoder.parameters(), "lr": args.lr_e, "weight_decay": args.wd_e})
    else:
        raise ValueError(f"Unknown model type {args.model_type}")


    list_opt_params.append(
        {"params": vae.prior_net_c.parameters(), "lr": args.lr_p, "weight_decay": args.wd_p}
    )
    list_opt_params.append(
        {"params": vae.prior_net_y.parameters(), "lr": args.lr_p, "weight_decay": args.wd_p}
    )
    list_opt_params.append(
        {"params": vae.decoder_x.parameters(), "lr": args.lr_dx, "weight_decay": args.wd_dx}
    )
    list_opt_params.append(
        {"params": vae.decoder_c.parameters(), "lr": args.lr_dc, "weight_decay": args.wd_dc}
    )
    list_opt_params.append(
        {"params": vae.decoder_y.parameters(), "lr": args.lr_dy, "weight_decay": args.wd_dy}
    )
    list_opt_params.append({"params": [vae.log_sigma_x], "lr": args.lr_sigma, "weight_decay": args.wd_sigma})

    # Just to be safe
    try:
        for param in vae.decoder_x.model.parameters():
            param.requires_grad = False
    except:
        print("No parameters found for partial model")

    # Optimizers
    optimizer_vae = optim.Adam(list_opt_params, lr=args.lr)
    pbar = trange(args.n_iter)

    # Logger
    logger = CSVLogger(exp_name="", log_dir=path_metrics)

    # Annealing
    lambda_annealer = Annealing(lambda_annealing, args.n_iter, **kwargs_lambda_annealing)
    beta_x_annealer = Annealing(beta_x_annealing, args.n_iter, **kwargs_beta_x_annealing)
    beta_c_annealer = Annealing(beta_c_annealing, args.n_iter, **kwargs_beta_c_annealing)
    beta_y_annealer = Annealing(beta_y_annealing, args.n_iter, **kwargs_beta_y_annealing)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # Print model summary
    print(ModelSummary(vae, max_depth=1))
    for iter in pbar:
        optimizer_vae.zero_grad()

        # Annealing lambda
        lambda_x_i = lambda_annealer.forward(iter) * args.lambda_g0
        vae.decoder_x.grad_reverse.alpha_ = lambda_x_i

        # Annealing beta
        beta_x_i = args.beta_x0 * beta_x_annealer.forward(iter)
        beta_c_i = args.beta_c0 * beta_c_annealer.forward(iter)
        beta_y_i = args.beta_y0 * beta_y_annealer.forward(iter)

        # Random batch
        sample_idx = torch.multinomial(torch.ones(args.n_train), args.n_batch, replacement=False)
        x_i, c_i, y_i = x_train[sample_idx, :], c_train[sample_idx, :], y_train[sample_idx, :]

        # VAE loss and outputs
        loss_ELBO, loss_KL_x, loss_KL_c, loss_KL_y, loss_Rx, loss_Rc, loss_Ry, loss_reg = vae.loss(
            x_i,
            c_i,
            y_i,
            n=args.n_mc_train,
            beta_x=beta_x_i,
            beta_c=beta_c_i,
            beta_y=beta_y_i,
            alpha_x=args.alpha_x,
            alpha_c=args.alpha_c,
            alpha_y=args.alpha_y,
        )
        loss_ELBO = loss_ELBO.sum() / (args.n_batch * (nd_x + nd_y + nd_c))
        loss_KL_x = loss_KL_x.sum() / args.n_batch
        loss_KL_c = loss_KL_c.sum() / args.n_batch
        loss_KL_y = loss_KL_y.sum() / args.n_batch
        loss_Rx = loss_Rx.sum() / args.n_batch
        loss_Rc = loss_Rc.sum() / args.n_batch
        loss_Ry = loss_Ry.sum() / args.n_batch
        loss_reg = loss_reg.sum() / args.n_batch

        # Backward call
        loss_ELBO.backward()

        # Clip gradients
        if args.clip_gradients == True:
            nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

        # Step
        optimizer_vae.step()

        # Log
        logger.log_scalar("ELBO", loss_ELBO, iter)
        logger.log_scalar("KLx", loss_KL_x, iter)
        logger.log_scalar("KLc", loss_KL_c, iter)
        logger.log_scalar("KLy", loss_KL_y, iter)
        logger.log_scalar("Rx", loss_Rx, iter)
        logger.log_scalar("Rc", loss_Rc, iter)
        logger.log_scalar("Ry", loss_Ry, iter)
        logger.log_scalar("reg", loss_reg, iter)
        logger.log_scalar("lambda_x", lambda_x_i, iter)
        logger.log_scalar("beta_x", beta_x_i, iter)
        logger.log_scalar("beta_c", beta_c_i, iter)
        logger.log_scalar("beta_y", beta_y_i, iter)
        logger.log_scalar("sigma_x", vae.log_sigma_x.exp().detach(), iter)

        # Validation loss
        if iter % args.val_freq == 0:
            with torch.no_grad():
                (
                    loss_ELBO_val,
                    loss_KL_x_val,
                    loss_KL_c_val,
                    loss_KL_y_val,
                    loss_Rx_val,
                    loss_Rc_val,
                    loss_Ry_val,
                    loss_reg_val,
                ) = vae.loss(
                    x_val,
                    c_val,
                    y_val,
                    n=args.n_mc_val,
                    beta_x=beta_x_i,
                    beta_c=beta_c_i,
                    beta_y=beta_y_i,
                    alpha_x=args.alpha_x,
                    alpha_c=args.alpha_c,
                    alpha_y=args.alpha_y,
                )

                # Normalize
                loss_ELBO_val = loss_ELBO_val.sum() / (args.n_val * (nd_x + nd_y + nd_c))
                loss_KL_x_val = loss_KL_x_val.sum() / args.n_val
                loss_KL_c_val = loss_KL_c_val.sum() / args.n_val
                loss_KL_y_val = loss_KL_y_val.sum() / args.n_val
                loss_Rx_val = loss_Rx_val.sum() / args.n_val
                loss_Rc_val = loss_Rc_val.sum() / args.n_val
                loss_Ry_val = loss_Ry_val.sum() / args.n_val
                loss_reg_val = loss_reg_val.sum() / args.n_val

                # Log
                logger.log_scalar("ELBO_val", loss_ELBO_val, iter)
                logger.log_scalar("KLx_val", loss_KL_x_val, iter)
                logger.log_scalar("KLc_val", loss_KL_c_val, iter)
                logger.log_scalar("KLy_val", loss_KL_y_val, iter)
                logger.log_scalar("Rx_val", loss_Rx_val, iter)
                logger.log_scalar("Rc_val", loss_Rc_val, iter)
                logger.log_scalar("Ry_val", loss_Ry_val, iter)
                logger.log_scalar("reg_val", loss_reg_val, iter)

                # Check early stopping criterion
                # Rx and Ry must be passed with a minus sign
                if early_stopping.early_stop(loss_ELBO_val):
                    break

        if iter % 10 == 0:
            pbar.set_postfix(
                ELBO_loss=logger.experiment.scalars["ELBO"][-1][1],
                ELBO_val=logger.experiment.scalars["ELBO_val"][-1][1],
                KL_x=logger.experiment.scalars["KLx"][-1][1],
                Rx=logger.experiment.scalars["Rx"][-1][1],
                Rc=logger.experiment.scalars["Rc"][-1][1],
                Ry=logger.experiment.scalars["Ry"][-1][1],
                Rx_val=logger.experiment.scalars["Rx_val"][-1][1],
                Rc_val=logger.experiment.scalars["Rc_val"][-1][1],
                Ry_val=logger.experiment.scalars["Ry_val"][-1][1],
                reg=logger.experiment.scalars["reg"][-1][1],
                lambda_x_i=logger.experiment.scalars["lambda_x"][-1][1],
                beta_x=logger.experiment.scalars["beta_x"][-1][1],
                beta_c=logger.experiment.scalars["beta_c"][-1][1],
                beta_y=logger.experiment.scalars["beta_y"][-1][1],
                sigma_x=logger.experiment.scalars["sigma_x"][-1][1],
                counter=early_stopping.counter,
                refresh=True,
            )

    return vae, logger

    # Set to eval
def evaluate_model(args, definition, model, data_test, cond=False):
    x_test, c_test, y_test = data_test[0], data_test[1], data_test[2]
    model.eval()

    # =====================================================
    # Test set metrics
    # =====================================================
    dict_pred_test = {}
    dict_metrics_test = {}

    # Sampling realization of y and computing the mean
    with torch.no_grad():
        (
            xh_pred,
            x_p_pred,
            x_d_pred,
            c_pred,
            y_pred,
            z_x_pred,
            z_c_pred,
            z_y_pred,
            dens_z_pred,
        ) = model.sample(x_test, c_test, cond=cond, n=args.n_mc_test)
    y_pred = y_pred.mean(dim=0).detach().cpu().numpy()

    # Compute metrics
    metrics_test = regression_metrics(y_test, y_pred)

    # Append metrics to dictionary
    dict_metrics_test[args.name] = metrics_test
    dict_pred_test[args.name] = y_pred

    return dict_metrics_test, dict_pred_test

def run_comparison(args, definition, data_train, data_test):
    x_train, c_train, y_train = data_train[0], data_train[1], data_train[2]
    x_test, c_test, y_test = data_test[0], data_test[1], data_test[2]

    assert (x_train.shape[0] == args.n_train)
    assert (args.n_batch <= args.n_train)

    # =====================================================
    # Comparison with standard data-driven classification/regression
    # =====================================================
    # Input transforms
    input_transform_x = StandardScaler()
    input_transform_c = StandardScaler()
    input_transform_y = StandardScaler()
    input_transform_x.fit(x_train)
    input_transform_c.fit(c_train)
    input_transform_y.fit(y_train)

    # Processing: normalize and stack x and c for train, validation and test sets
    x_train_t, _ = input_transform_x.forward(x_train)
    c_train_t, _ = input_transform_c.forward(c_train)
    x_train_n = torch.cat((x_train_t, c_train_t), dim=-1).detach().cpu().numpy()

    # Test
    x_test_t, _ = input_transform_x.forward(x_test)
    c_test_t, _ = input_transform_c.forward(c_test)
    x_test_n = torch.cat((x_test_t, c_test_t), dim=-1).detach().cpu().numpy()

    # Targets
    y_train_n = y_train.detach().cpu().numpy()
    y_test_n = y_test.detach().cpu().numpy()

    # Model parameters
    gpr_kernel = RBF() + WhiteKernel()

    # List of regressors
    regressors = {
        "LIN": LinearRegression(),
        "GPR": GaussianProcessRegressor(gpr_kernel),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=10000),
    }

    dict_pred_test = {}
    dict_metrics_test = {}
    for rgr_name, rgr_model in regressors.items():
        print(f"Fitting {rgr_name}")

        # Fit
        rgr_model.fit(x_train_n, y_train_n)

        # Predict
        dict_pred_test[rgr_name] = rgr_model.predict(x_test_n)
        dict_metrics_test[rgr_name] = regression_metrics(y_test_n, dict_pred_test[rgr_name])

    return dict_metrics_test, dict_pred_test


def disentanglement_metric(args, model, definition, data_train, data_test, regressor="linear", cond=False, use_mean=False):
    gen_factors = list(definition["dict_gt"].keys())
    x_train, c_train, y_train, z_train = data_train[0], data_train[1], data_train[2], data_train[3]
    x_test, c_test, y_test, z_test = data_test[0], data_test[1], data_test[2], data_test[3]
    model.eval()

    if use_mean == True:
        n = args.n_mc_test
    else:
        n = 1

    # Get train data
    (
        _,
        _,
        _,
        _,
        _,
        z_x_train,
        z_c_train,
        z_y_train,
        dens_z_train,
    ) = model.sample(x_train, c_train, cond=cond, n=n)

    # Get test data
    (
        _,
        _,
        _,
        _,
        _,
        z_x_test,
        z_c_test,
        z_y_test,
        dens_z_test,
    ) = model.sample(x_test, c_test, cond=cond, n=n)

    # Train and test inputs
    z_x_train = z_x_train.mean(dim=0).detach().cpu()
    z_c_train = z_c_train.mean(dim=0).detach().cpu()
    z_y_train = z_y_train.mean(dim=0).detach().cpu()
    z_x_test = z_x_test.mean(dim=0).detach().cpu()
    z_c_test = z_c_test.mean(dim=0).detach().cpu()
    z_y_test = z_y_test.mean(dim=0).detach().cpu()

    z_train = z_train.squeeze(0).detach().cpu()
    z_test = z_test.squeeze(0).detach().cpu()

    # # Scaling
    # scaler_x = sklearn_scaler().fit(z_x_train)
    # scaler_c = sklearn_scaler().fit(z_c_train)
    # scaler_y = sklearn_scaler().fit(z_y_train)
    #
    # z_x_train_s = scaler_x.transform(z_x_train)
    # z_c_train_s = scaler_c.transform(z_c_train)
    # z_y_train_s = scaler_y.transform(z_y_train)
    # z_x_test_s = scaler_x.transform(z_x_test)
    # z_c_test_s = scaler_c.transform(z_c_test)
    # z_y_test_s = scaler_y.transform(z_y_test)

    # Initialize a model for each latent variable group and latent factor
    # score_test = {"zx": {}, "zc": {}, "zy": {}}
    score_test = []
    for i, factor_i in enumerate(gen_factors):

        # Fit
        if regressor == "linear":
            reg_x = LinearRegression().fit(z_x_train, z_train[:, i])
            reg_c = LinearRegression().fit(z_c_train, z_train[:, i])
            reg_y = LinearRegression().fit(z_y_train, z_train[:, i])
        elif regressor == "mlp":
            reg_x = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=20000).fit(z_x_train, z_train[:, i])
            reg_c = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=20000).fit(z_c_train, z_train[:, i])
            reg_y = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=20000).fit(z_y_train, z_train[:, i])
        else:
            raise ValueError(f"Unknown regressor type {regressor}")

        score_test.append(["zx", factor_i, reg_x.score(z_x_test, z_test[:, i])])
        score_test.append(["zc", factor_i, reg_c.score(z_c_test, z_test[:, i])])
        score_test.append(["zy", factor_i, reg_y.score(z_y_test, z_test[:, i])])

        # score_test["zx"][factor_i] = reg_x.score(z_x_test_s, z_test[:, i])
        # score_test["zc"][factor_i] = reg_c.score(z_c_test_s, z_test[:, i])
        # score_test["zy"][factor_i] = reg_y.score(z_y_test_s, z_test[:, i])

    return score_test

if __name__ == "__main__":
    # ============================================================
    # SETUP
    # ============================================================
    # Problem definition
    name = "single_run"
    case = simple_beam
    definition = case.definition
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
    vae, logger = train_model(args, vae, definition, data_train, data_val, path_metrics=path_metrics, path_figures=path_figures)

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
            save_close_fig(fig_pred_x, os.path.join(path_figures, "fig_pred_x_" + str(idx_var_gt) + ".png"), show=show_plots)


    if plot_interpolation == True:
        # ------------------------------------
        # Interpolation in the predictions
        # ------------------------------------
        fig_pred_interp_x, ax_pred_interp_x = plot_interp_pred(
            vae, args.n_interp, definition, n_plot=args.n_plot, cond=cond
        )
        save_close_fig(fig_pred_interp_x, os.path.join(path_figures, "fig_pred_interp_x.png"), show=show_plots)

        # # ------------------------------------
        # # Interpolation in the latent space - 3D
        # # ------------------------------------
        # fig_post_z, ax_post_z = plot_interp_latent_3D(
        #     vae, args.n_interp, definition, n_plot=args.n_plot, cond=cond,
        # )
        # save_close_fig(fig_post_z, os.path.join(path_figures, "fig_post_z.png"), show=show_plots)
        #
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

        # ------------------------------------
        # Interpolation in the latent space - Posterior marginals for specific dimensions
        # ------------------------------------
        vars_interp = [0, 1]
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
