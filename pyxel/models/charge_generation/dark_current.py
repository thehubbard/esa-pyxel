#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

"""Simple models to generate charge due to dark current process."""

import typing as t

import numpy as np

from pyxel.detectors import APD, Detector
from pyxel.util import temporary_random_state


def calculate_simple_dark_current(
    num_rows: int, num_cols: int, current: float, exposure_time: float
) -> np.ndarray:
    """Simulate dark current in a :term:`CCD`.

    Parameters
    ----------
    num_rows : int
        Number of rows for the generated image.
    num_cols : int
        Number of columns for the generated image.
    current : float
        Dark current, in electrons/pixel/second
    exposure_time : float
        Length of the simulated exposure, in seconds.

    Returns
    -------
    dark_im_array_2d: ndarray
        An array the same shape and dtype as the input containing dark counts
        in units of charge (e-).
    """
    # dark current for every pixel
    mean_dark_charge = current * exposure_time

    # This random number generation should change on each call.
    dark_im_array_2d = np.random.poisson(mean_dark_charge, size=(num_rows, num_cols))
    return dark_im_array_2d


@temporary_random_state
def simple_dark_current(
    detector: Detector, dark_rate: float, seed: t.Optional[int] = None
) -> None:
    """Simulate dark current in a detector.

    Parameters
    ----------
    detector : Detector
        Any detector object.
    dark_rate : float
        Dark current, in electrons/pixel/second, which is the way
        manufacturers typically report it.
    seed: int, optional
    """

    exposure_time = detector.time_step
    geo = detector.geometry

    dark_current_array = calculate_simple_dark_current(
        num_rows=geo.row,
        num_cols=geo.col,
        current=dark_rate,
        exposure_time=exposure_time,
    ).astype(
        float
    )  # type: np.ndarray

    detector.charge.add_charge_array(dark_current_array)


def calculate_dark_current_saphira(
    temperature: float,
    avalanche_gain: float,
    shape: t.Tuple[int, int],
    exposure_time: float,
) -> np.ndarray:
    """Simulate dark current in a Saphira :term:`APD`.

    From: I. M. Baker et al., Linear-mode avalanche photodiode arrays in HgCdTe at Leonardo, UK: the
    current status, in Image Sensing Technologies: Materials, Devices, Systems, and Applications VI,
    2019, vol. 10980, no. May, p. 20.

    Parameters
    ----------
    temperature
    avalanche_gain
    shape
    exposure_time : float
        Length of the simulated exposure, in seconds.

    Returns
    -------
    dark_im_array_2d: ndarray
        An array the same shape and dtype as the input containing dark counts
        in units of charge (e-).
    """

    # We can split the dark current vs. gain vs. temp plot ([5] Fig. 3) into three linear
    # 'regimes': 1) low-gain, low dark current; 2) nominal; and 3) trap-assisted tunneling.
    # The following ignores (1) for now since this only applies at gains less than ~2.

    dark_nominal = (15.6e-12 * np.exp(0.2996 * temperature)) * (
        avalanche_gain
        ** ((-5.43e-4 * (temperature**2)) + (0.0669 * temperature) - 1.1577)
    )
    dark_tunnel = 3e-5 * (avalanche_gain**3.2896)

    # The value we want is the maximum of the two regimes (see the plot):
    dark = max(dark_nominal, dark_tunnel)

    # dark current for every pixel
    mean_dark_charge = dark * exposure_time

    # This random number generation should change on each call.
    dark_im_array_2d = np.random.poisson(mean_dark_charge, size=shape)

    return dark_im_array_2d


@temporary_random_state
def dark_current_saphira(detector: APD, seed: t.Optional[int] = None) -> None:
    """Simulate dark current in a Saphira APD detector.

    Reference: I. M. Baker et al., Linear-mode avalanche photodiode arrays in HgCdTe at Leonardo, UK: the
    current status, in Image Sensing Technologies: Materials, Devices, Systems, and Applications VI,
    2019, vol. 10980, no. May, p. 20.

    Parameters
    ----------
    detector : APD
        An APD detector object.
    seed: int, optional
    """

    if not isinstance(detector, APD):
        raise TypeError("Expecting an APD object for detector.")
    if detector.environment.temperature > 100:
        raise ValueError(
            "Dark current estimation is inaccurate for temperatures more than 100 K!"
        )
    if detector.characteristics.avalanche_gain < 2:
        raise ValueError("Dark current is inaccurate for avalanche gains less than 2!")

    exposure_time = detector.time_step

    dark_current_array = calculate_dark_current_saphira(
        temperature=detector.environment.temperature,
        avalanche_gain=detector.characteristics.avalanche_gain,
        shape=detector.geometry.shape,
        exposure_time=exposure_time,
    ).astype(
        float
    )  # type: np.ndarray

    detector.charge.add_charge_array(dark_current_array)
