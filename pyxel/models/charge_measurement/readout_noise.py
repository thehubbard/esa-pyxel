#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Readout noise model."""
import logging
import typing as t

import numpy as np

from pyxel.detectors import CMOS, Detector
from pyxel.util import temporary_random_state


def create_noise(signal_2d: np.ndarray, std_deviation: float) -> np.ndarray:
    """Create noise to signal array using normal random distribution.

    Parameters
    ----------
    signal_2d : ndarray
    std_deviation : float

    Returns
    -------
    ndarray
    """
    sigma_2d = std_deviation * np.ones_like(signal_2d)
    noise_2d = np.random.normal(loc=signal_2d, scale=sigma_2d)

    return noise_2d


@temporary_random_state
def output_node_noise(
    detector: Detector,
    std_deviation: float,
    seed: t.Optional[int] = None,
) -> None:
    """Add noise to signal array of detector output node using normal random distribution.

    Parameters
    ----------
    detector: Detector
        Pyxel detector object.
    std_deviation: float
        Standard deviation. Unit: V
    seed: int, optional
        Random seed.

    Raises
    ------
    ValueError
        Raised if 'std_deviation' is negative
    """
    if std_deviation < 0.0:
        raise ValueError("'std_deviation' must be positive.")

    noise_2d = create_noise(
        signal_2d=np.asarray(detector.signal.array, dtype=np.float64),
        std_deviation=std_deviation,
    )  # type: np.ndarray

    detector.signal.array = noise_2d


@temporary_random_state
def output_node_noise_cmos(
    detector: "CMOS",
    readout_noise: float,
    readout_noise_std: float,
    seed: t.Optional[int] = None,
) -> None:
    """Output node noise model for CMOS detectors where readout is statistically independent for each pixel.

    Parameters
    ----------
    detector: Detector
    readout_noise: Mean readout noise for the array in units of electrons.
    readout_noise_std: Readout noise standard deviation in units of electrons.
    seed: int, optional

    Returns
    -------
    None
    """
    logging.info("")

    # sv is charge readout sensitivity
    sv = detector.characteristics.sv

    signal_mean_array = detector.signal.array.astype("float64")
    sigma_array = np.random.normal(
        loc=readout_noise * sv,
        scale=readout_noise_std * sv,
        size=signal_mean_array.shape,
    )

    signal = np.random.normal(loc=signal_mean_array, scale=sigma_array)

    detector.signal.array = signal
