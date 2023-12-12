import numpy as np


def accuracy_score(y_pred, y_true):
    """
    Compute the accuracy score between predicted and true labels.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y_true (numpy.ndarray): True labels.

    Returns:
    float: Accuracy score between 0 and 1.
    """
    if not (isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays.")

    if len(y_pred) != len(y_true):
        raise ValueError("Arrays must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Arrays must have at least one element.")

    correct = np.sum(y_pred == y_true)

    return float(correct / len(y_true))
