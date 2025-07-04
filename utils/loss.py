def get_logger_training_curve(logger, label):
    list_iters = [item[0] for item in logger.experiment.scalars[label]]
    list_vals = [item[1] for item in logger.experiment.scalars[label]]
    return list_iters, list_vals

class EarlyStopping:
    """
    Modified version of the script from this stackoverflow answer:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False