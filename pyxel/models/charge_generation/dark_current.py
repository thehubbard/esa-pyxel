#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

"""Simple models to generate charge due to dark current process."""

import typing as t

import numpy as np

from pyxel.detectors import CCD
from pyxel.util import temporary_random_state


def calculate_dark_current(
    num_rows: int,
    num_cols: int,
    current: float,
    exposure_time: float,
    gain: float = 1.0,
) -> np.ndarray:
    """Simulate dark current in a CCD.

    Parameters
    ----------
    num_rows : int
        Number of rows for the generated image.
    num_cols : int
        Number of columns for the generated image.
    current : float
        Dark current, in electrons/pixel/second, which is the way
        manufacturers typically report it.
    exposure_time : float
        Length of the simulated exposure, in seconds.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.

    Returns
    -------
    ndarray
        An array the same shape and dtype as the input containing dark counts
        in units of ADU.
    """
    # dark current for every pixel
    base_current = current * exposure_time / gain

    # This random number generation should change on each call.
    dark_im_array_2d = np.random.poisson(base_current, size=(num_rows, num_cols))
    return dark_im_array_2d


@temporary_random_state
def dark_current(
    detector: CCD, dark_rate: float, gain: float = 1.0, seed: t.Optional[int] = None
) -> None:
    """Simulate dark current in a CCD.

    Parameters
    ----------
    detector : Detector
        CCD detector object.
    dark_rate : float
        Dark current, in electrons/pixel/second, which is the way
        manufacturers typically report it.
    gain : float, optional. Default: 1.0
        Gain of the camera, in units of electrons/ADU.
    seed: int, optional
    """
    if not isinstance(detector, CCD):
        raise TypeError("Expecting a CCD object for detector.")

    exposure_time = detector.time_step
    geo = detector.geometry

    dark_current_array = calculate_dark_current(
        num_rows=geo.row,
        num_cols=geo.col,
        current=dark_rate,
        exposure_time=exposure_time,
        gain=gain,
    )  # type: np.ndarray

    detector.charge.add_charge_array(dark_current_array)
