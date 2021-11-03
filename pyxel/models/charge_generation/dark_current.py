#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.


import numpy as np
import logging
from pyxel.detectors import Detector

def calculate_dark_current(geo, current, exposure_time,gain=1.0) -> np.ndarray:
    """
    Simulate dark current in a CCD, optionally including hot pixels.
    Parameters
    ----------
    image : numpy array
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

    # dark current for every pixel
    base_current = current * exposure_time / gain

    # This random number generation should change on each call.
    rng = np.random.default_rng()
    dark_im_array_2d = rng.poisson(base_current, size=(geo.row, geo.col))

    return dark_im_array_2d


def dark_current(detector, dark_rate, gain) -> None:
    """Adding dark current"""
    logging.info("")

    exposure_time = detector.time_step
    geo = detector.geometry

    dark_current_array = calculate_dark_current(geo, dark_rate, exposure_time, gain)

    dark_current_value = dark_current_array.flatten()
    where_non_zero = np.where(dark_current_value > 0)
    dark_current_value = dark_current_value[where_non_zero]
    size = dark_current_value.size

    init_hor_pix_position = geo.horizontal_pixel_center_pos_list()[where_non_zero]
    init_ver_pix_position = geo.vertical_pixel_center_pos_list()[where_non_zero]

    detector.charge.add_charge(
        particle_type="e",
        particles_per_cluster=dark_current_value,
        init_energy=np.zeros(size),
        init_ver_pix_position=init_ver_pix_position,
        init_hor_pix_position=init_hor_pix_position,
        init_z_position=np.zeros(size),
        init_ver_velocity=np.zeros(size),
        init_hor_velocity=np.zerps(size),
        init_z_velocity=np.zeros(size)
    )
