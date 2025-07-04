import torch

# Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def regression_metrics(y_test, y_pred):
    """
    Evaluate the specified metrics and return a dictionary of metric-value pairs

    Parameters
    ----------
    y_test
    y_pred

    Returns
    -------

    """
    y_test = y_test.detach().cpu().numpy() if torch.is_tensor(y_test) else y_test
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Add metrics to dictionary
    dict_metrics = dict()
    dict_metrics["R2"] = r2_score(y_test, y_pred, multioutput="raw_values")
    dict_metrics["MSE"] = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    dict_metrics["MAE"] = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    return dict_metrics
