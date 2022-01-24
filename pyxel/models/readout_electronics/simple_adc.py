#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import numpy as np
from pyxel.detectors import Detector


def apply_simple_adc(signal: np.ndarray, bit_resolution: int, voltage_range: t.Tuple[float, float]) -> np.ndarray:
    """

    Parameters
    ----------
    signal
    bit_resolution
    voltage_range

    Returns
    -------
    ndarray
    """
    bins = np.linspace(start=voltage_range[0], stop=voltage_range[1], num=2 ** bit_resolution)
    output = np.digitize(x=signal, bins=bins[:-1], right=True)


def simple_adc(detector: Detector, bit_resolution: int, voltage_range: t.Tuple[float, float]) -> None:
    """TBW.

    Parameters
    ----------
    detector: Detector
    bit_resolution: int
    voltage_range: tuple of floats
    """
    detector.image.array = apply_simple_adc(signal=detector.signal.array, bit_resolution=bit_resolution, voltage_range=voltage_range)

