"""Fitness functions for model fitting."""
import typing as t

import numpy as np


def sum_of_abs_residuals(
    simulated: np.ndarray, target: np.ndarray, weighting: t.Optional[float] = None
) -> np.ndarray:
    """TBW.

    Parameters
    ----------
    simulated
    target
    weighting

    Returns
    -------
    array
        TBW.
    """
    try:
        diff = target.astype(float) - simulated.astype(float)  # type: np.ndarray
    except AttributeError:
        diff = float(target) - float(simulated)

    if weighting is not None:
        diff *= weighting
    return np.sum(np.abs(diff))


def sum_of_squared_residuals(
    simulated: np.ndarray, target: np.ndarray, weighting: t.Optional[float] = None
) -> np.ndarray:
    """TBW.

    Parameters
    ----------
    simulated
    target
    weighting

    Returns
    -------
    array
        TBW.
    """
    try:
        diff = target.astype(float) - simulated.astype(float)
    except AttributeError:
        diff = float(target) - float(simulated)
    diff_square = diff * diff

    if weighting is not None:
        diff_square *= weighting
    return np.sum(diff_square)
