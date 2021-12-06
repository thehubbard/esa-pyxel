#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.


import numpy as np
import logging
from pyxel.detectors import CCD
from pyxel.util import temporary_random_state


@temporary_random_state
def calculate_dark_current(geo, current, exposure_time, gain=1.0) -> np.ndarray:
    # dark current for every pixel
    base_current = current * exposure_time / gain

    # This random number generation should change on each call.
    dark_im_array_2d = np.random.poisson(base_current, size=(geo.row, geo.col))
    return dark_im_array_2d


def dark_current(detector: CCD, dark_rate, gain) -> None:
    """
    Simulate dark current in a CCD
    ----------
    geo : numpy array
        Image whose shape the cosmic array should match.
    current : float
        Dark current, in electrons/pixel/second, which is the way
        manufacturers typically report it.
    exposure_time : float
        Length of the simulated exposure, in seconds.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    Returns
    -------
    numpy array
        An array the same shape and dtype as the input containing dark counts
        in units of ADU.
    """
    if not isinstance(detector, CCD):
        raise TypeError("Expecting a CCD object for detector.")
    logging.info("")

    exposure_time = detector.time_step
    geo = detector.geometry

    dark_current_array = calculate_dark_current(geo, dark_rate, exposure_time, gain)

    detector.charge.add_charge_array(dark_current_array)
