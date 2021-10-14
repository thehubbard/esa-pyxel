#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Fitness functions for model fitting."""

import typing as t
import numba

import numpy as np


@numba.njit()
def sum_of_abs_residuals(
    simulated: np.ndarray,
    target: np.ndarray,
    weighting: t.Optional[t.Union[np.ndarray, float]] = None,
) -> float:
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
    diff = target - simulated

    if weighting is not None:
        diff *= weighting

    result = np.sum(np.abs(diff))  # type: float
    return result


@numba.njit()
def sum_of_squared_residuals(
    simulated: np.ndarray,
    target: np.ndarray,
    weighting: t.Optional[t.Union[np.ndarray, float]] = None,
) -> float:
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
    diff = target - simulated

    diff_square = diff * diff

    if weighting is not None:
        diff_square *= weighting

    result = np.sum(diff_square)  # type: float
    return result
