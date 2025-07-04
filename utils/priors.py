import torch
from torch import distributions as dist
from utils import device
from scipy.linalg import circulant


class MarginalDistribution:
    """
    Convenience class for evaluating the log probability and sampling from
    independent RVs with specified distributions

    distributions: list of torch distributions
    """

    def __init__(self, distributions):
        self.n_z = len(distributions)
        self.distributions = distributions

    def log_prob(self, z):
        p_z = torch.zeros(z.shape)
        for i, dist_i in enumerate(self.distributions):
            p_z[..., i] = dist_i.log_prob(z[..., i])
        return p_z.to(device)

    def icdf(self, u):
        u = torch.atleast_2d(u)
        z = torch.zeros((u.shape[0], self.n_z))
        for i, dist_i in enumerate(self.distributions):
            z[..., i] = dist_i.icdf(u[..., i])
        return z.to(device)

    def sample(self, shape):
        z = torch.zeros((*shape, len(self.distributions)))
        for j, dist_j in enumerate(self.distributions):
            z[..., j] = dist_j.sample(shape).squeeze()
        return z.to(device)


def get_prior_dist(prior):
    list_prior_dist = [item["dist"](**item["args"]) for key, item in prior.items()]
    return MarginalDistribution(list_prior_dist)


def interp_ground_truth(dict_gt):
    interp_lb = []
    interp_ub = []
    for key, item in dict_gt.items():
        interp_lb.append(item["plb"])
        interp_ub.append(item["pub"])
    return interp_lb, interp_ub


def get_shapes_from_dict(dict_gt):
    z_idx_x = [idx for idx, item in enumerate(dict_gt.values()) if item["type"] == "x"]
    z_idx_c = [idx for idx, item in enumerate(dict_gt.values()) if item["type"] == "c"]
    z_idx_y = [idx for idx, item in enumerate(dict_gt.values()) if item["type"] == "y"]
    z_idx_f = [idx for idx, item in enumerate(dict_gt.values()) if item["type"] == "f"]
    z_idx_p = [
        idx for idx, item in enumerate(dict_gt.values()) if ((item["phys"] == True) and item["type"] == "c")
    ]
    return len(z_idx_x), len(z_idx_c), len(z_idx_y), len(z_idx_f), len(z_idx_p)


def make_square_dist(definition):
    """
    NOTES:
        * Assigns uniform distributions between the lower
        and upper bound for all variables
        * Assumes exactly two physics-based latent variables
    """

    # Get physics-based latent variable ground truth distributions
    dict_phys = {key: value for key, value in definition["dict_gt"].items() if value["type"] == "x"}
    assert len(dict_phys) == 2

    # Get bounds
    lb = torch.tensor([item["lb"] for key, item in definition["dict_gt"].items()]).to(device)
    ub = torch.tensor([item["ub"] for key, item in definition["dict_gt"].items()]).to(device)

    # Edit bounds for physics-based latent variables
    lb_x = torch.tensor([item["args"]["low"] for key, item in dict_phys.items()]).to(device)
    ub_x = torch.tensor([item["args"]["high"] for key, item in dict_phys.items()]).to(device)
    ce_x = lb_x + (ub_x - lb_x) / 2

    # Assign
    bounds_0 = torch.tensor([[lb_x[0], ce_x[0]], [ce_x[0], ub_x[0]], [ce_x[0], ub_x[0]], [lb_x[0], ce_x[0]]])
    bounds_1 = torch.tensor([[lb_x[1], ce_x[1]], [lb_x[1], ce_x[1]], [ce_x[1], ub_x[1]], [ce_x[1], ub_x[1]]])

    lb_new = lb.clone().repeat(4, 1)
    ub_new = ub.clone().repeat(4, 1)

    lb_new[:, 0], lb_new[:, 1] = bounds_0[:, 0], bounds_1[:, 0]
    ub_new[:, 0], ub_new[:, 1] = bounds_0[:, 1], bounds_1[:, 1]

    # Assemble MixtureSameFamily distributions
    mat_circ = circulant(torch.arange(4))
    dist_train = []
    dist_test = []
    for i in range(4):
        idx_train = mat_circ[:3, i]
        idx_test = mat_circ[3:, i]

        # Construct bounds
        lb_train = lb_new[idx_train, :]
        lb_test = lb_new[idx_test, :].squeeze()
        ub_train = ub_new[idx_train, :]
        ub_test = ub_new[idx_test, :].squeeze()

        # Make distributions
        dist_comp_i = dist.Independent(dist.Uniform(lb_train, ub_train), 1)
        dist_train_i = dist.MixtureSameFamily(dist.Categorical(torch.ones(3).to(device)), dist_comp_i)
        dist_test_i = dist.Uniform(lb_test, ub_test)

        dist_train.append(dist_train_i)
        dist_test.append(dist_test_i)
    return dist_train, dist_test
