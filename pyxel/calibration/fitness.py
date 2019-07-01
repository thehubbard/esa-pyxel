"""Fitness functions for model fitting."""
import numpy as np


# HANS: refactor code to this.
# def sum_of_abs_residuals(
#         simulated: t.Union[float, np.array],
#         target: t.Union[float, np.array],
#         weighting: t.Optional[t.Union[float, np.array]] = None) -> float:
#     """TBW."""
#     diff = np.array(target).astype(float) - np.array(simulated).astype(float)
#     if weighting is not None:
#         diff *= weighting
#     return np.sum(np.abs(diff))


# FRED: Add typing information for all functions
def sum_of_abs_residuals(simulated, target, weighting=None):
    """TBW.

    :param simulated: np.array
    :param target: np.array
    :param weighting: np.array
    :return:
    """
    try:
        diff = target.astype(float) - simulated.astype(float)
    except AttributeError:
        diff = float(target) - float(simulated)

    if weighting is not None:
        diff *= weighting
    return np.sum(np.abs(diff))


def sum_of_squared_residuals(simulated, target, weighting=None):
    """TBW.

    :param simulated: np.array
    :param target: np.array
    :param weighting: np.array
    :return:
    """
    try:
        diff = target.astype(float) - simulated.astype(float)
    except AttributeError:
        diff = float(target) - float(simulated)
    diff_square = diff * diff

    if weighting is not None:
        diff_square *= weighting
    return np.sum(diff_square)
