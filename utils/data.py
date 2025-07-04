from torch import distributions as dist
from utils import device
from sklearn.model_selection import train_test_split

def test_train_split(n_train, n_test, data):
    return train_test_split(*data, test_size=int(n_test), train_size=int(n_train))


def sample_response(definition, n, sample_dist=None, z=None):
    """
    Utility function for sampling the structural response for a given problem

    :param definition:
    :param sample_dist:
    :param n:
    :param z:
    :return:
    """
    if (sample_dist == None) and (z == None):
        raise ValueError("At least one of `sample_dist` and `z` must not be `None`")
    if z == None:
        z_sample = sample_dist.sample((n,))
    elif sample_dist == None:
        z_sample = z.unsqueeze(0).repeat(n, 1, 1)

    # Send to device
    z_sample = z_sample.to(device)

    # Ground truth dictionary
    dict_gt = definition["dict_gt"]
    z_idx_c = [idx for idx, val in enumerate(dict_gt.values()) if val["type"] == "c"]
    z_idx_y = [idx for idx, val in enumerate(dict_gt.values()) if val["type"] == "y"]

    # Evaluate full model
    model = definition["full_model"]
    x_sample = model(z_sample)
    x_sample += dist.Normal(0.0, definition["sigma_x"]).sample(x_sample.shape)

    # Get environmental parameters
    c_sample = z_sample[..., z_idx_c].reshape((*z_sample.shape[:-1], len(z_idx_c)))
    c_sample += dist.Normal(0.0, definition["sigma_c"]).sample(c_sample.shape)

    # Get labels
    y_sample = z_sample[..., z_idx_y].reshape((*z_sample.shape[:-1], len(z_idx_y)))
    y_sample += dist.Normal(0.0, definition["sigma_y"]).sample(y_sample.shape)

    # Send to device
    x_sample = x_sample.to(device)
    c_sample = c_sample.to(device)
    y_sample = y_sample.to(device)

    return x_sample, c_sample, y_sample, z_sample