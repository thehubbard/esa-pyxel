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

# from astropy import units as u


# @validators.validate
# @config.argument(name='', label='', units='', validate=)


@temporary_random_state
def output_node_noise(
    detector: Detector, std_deviation: float, seed: t.Optional[int] = None
) -> None:
    """Add noise to signal array of detector output node using normal random distribution.

    Parameters
    ----------
    detector: Detector
        Pyxel detector object.
    std_deviation: float
        Standard deviation.
    seed: int, optional
        Random seed.
    """
    logging.info("")

    signal_mean_array = detector.signal.array.astype("float64")
    sigma_array = std_deviation * np.ones(signal_mean_array.shape)

    signal = np.random.normal(loc=signal_mean_array, scale=sigma_array)

    detector.signal.array = signal


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
